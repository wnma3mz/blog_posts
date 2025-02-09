---
title: together-LLM 跨机后端开发
date: 2024-11-15 22:05:06
tags: []
categories: []
mathjax: true
---

记录开发 tLLM 中后端相关的问题。

<!-- more -->

## RoadMap

使用 `torch.dist` 实现 张量并行，使用 `grpc` 实现流水并行

- [x] Web UI
    - [x] Node Status
        - [ ] Display Multi Model
    - [x] ChatWeb Demo by Gradio
        - [x] Parameters
        - [x] System
        - [x] Button
- [x] OpenAI API format
    - [x] Streaming Output
    - [x] chat completion(stream)
    - [x] chat completion(non-stream)
    - [x] using anythingLLM
- [x] Client Send Url and Port
- [ ] Auto Layer Split
    - [x] get free layer idx
    - [x] fix split layer pipeline
    - [x] calculate layer memory and recommend split
    - [ ] split model before load
- [x] Async Generation
    - [x] Multi-Sequence Batch=1
    - [x] Queuing mechanism
    - [x] Continuous Batch
    - [x] Test Cases
    - [x] Client Disconnect and Abort
    - [x] await Event
- [x] Communication
    - [x] Communication Time Benchmark
    - [x] Async GRPC
    - [x] Ring Communication
- [x] Auto Find Node
    - [x] WebSocket Communication
    - [x] Client Retry Connect
    - [x] Client auto update url 
    - [x] Master Exit
- [x] Auto Download

## 初始化方法

Master 和 Client 交互方式 http
- Master 先启动，已知模型名和层数
    - Client 启动 grpc，HTTP 发送可连接到地址信息（TODO 内存/显存大小/算力等信息）到 Master
    - Master 返回模型名，分配的起始和结束层数（同步操作，不需要状态）
    - Client 下载模型，加载模型，向 Master 发送 InitModel 信息完成
    - 之后 Master 会向 Client 定时发送心跳包，确保 Client 连接正常

- 如果 Master 重启，Master 会丢失所有的 Client 信息
    - Client 会有定时心跳检查，带着已有状态重新连接


## Engine 和 HTTP Server 架构分离

LLM 可以被视作一个独立的超级重**计算**的进程，所以跟 HTTP Server 放到一个进程中会导致 CPU 资源被抢占。

所以需要额外用一个进程来负责 Engine 的计算，HTTP Server 负责接收请求，将请求转发给 Engine，然后将 Engine 的结果返回给请求者。

但 tllm 本身是跨机器的，并不需要所有层都在一个进程中，所以这里分离的对象有所不同。这里的 Engine 和 HTTP Server 是在同一个机器上的。而把客户端的 Engine 独立了一个进程。

当然，对于多模态的情况，可能还是需要把 Engine 这部分分离处理，避免占用资源。但由于消息传递暂时不好处理，这部分暂未实现。

## CPU 死循环问题

Engine 本身是一个**死循环**的函数。哪怕在没有任务的时候，一直处于死循环状态，这样会导致 CPU 占用过高。如下所示

```python
    async def _generate(self):
        # 死循环，持续从队列中获取数据
        while True:
            # 做一些事情

    async def generate_stream(self, request_data: SequenceRequestData):
        # 数据进入到工作队列
        self.prefill_queue.put_nowait(request_data)

        # 等待队列完成
        ...
```

在 python 的异步队列中有一个 `asyncio.Event()`。这个事件对象可以用来在多个协程之间传递信号。当一个协程调用了 `set()` 方法，其他协程调用 `wait()` 方法就会立即返回。这样就可以实现协程之间的通信。

所以可以用这个信号来避免 CPU 过高占用。如下所示

```python

class AsyncEngine:
    def __init__(self, generator: Union[LLMGenerator, ImageGenerator], sleep_time: float = 0.0, limit_size: int = 5):
        ...
        self.queue_not_empty: asyncio.Event = asyncio.Event()

    async def _generate(self):
        while True:
            await self.queue_not_empty.wait()

    async def generate_stream(self, request_data: SequenceRequestData):
        self.prefill_queue.put_nowait(request_data)
        self.queue_not_empty.set()

        # 等待队列完成
        ...
```


## 请求排队问题

考虑到下面几个原因，在 Engine 中需要设计三个队列来处理不同的请求。

- LLM 对应的深度学习模型是一个计算密集型的任务，所以在处理请求的时候，可能会有多个请求同时进入。需要一个队列来处理这些请求。（传统叫攒 Batch，批量处理若干请求的计算效率最高）

- LLM 是 token by token 生成的，一个请求会反复被处理。

- 由于 LLM 生成时间过长，可能会被随时中断，所以需要一个队列来处理中断请求。

所以设计了三个队列，分别是 `prefill_queue`、`decoding_queue` 和 `abort_queue`。如下所示。

并且在每次进行计算前，都会通过 `fetch_data` 函数来控制哪些请求需要被处理。

```python

class AsyncEngine:
    def __init__(self, generator: Union[LLMGenerator, ImageGenerator], sleep_time: float = 0.0, limit_size: int = 5):
        ...
        self.prefill_queue: asyncio.Queue = asyncio.Queue()
        self.decoding_queue: asyncio.Queue = asyncio.Queue()
        self.abort_queue: asyncio.Queue = asyncio.Queue()
        ...

    async def fetch_data(self):
        aborting_request_ids = set()
        while not self.abort_queue.empty():
            request_id = self.abort_queue.get_nowait()
            aborting_request_ids.add(request_id)

        async def aborting_filter(request_data) -> bool:
            if request_data.request_id in aborting_request_ids:
                self.logger.debug(f"aborting generate request_id: {request_data.request_id}")
                request_data.is_stop = True
                request_data.finish_reason_list = ["abort"]
                aborting_request_ids.remove(request_data.request_id)
                return True
            return False

        # prefill 队列和 decoding 队列的调度逻辑
        request_data_list = []

        # 优先从 decoding_queue 取数据
        while not self.decoding_queue.empty() and len(request_data_list) < self.limit_size:
            request_data = self.decoding_queue.get_nowait()
            if await aborting_filter(request_data):
                continue
            request_data_list.append(request_data)

        # 从 prefill_queue 中取数据，直到达到限制
        while not self.prefill_queue.empty() and len(request_data_list) < self.limit_size:
            request_data = self.prefill_queue.get_nowait()
            if await aborting_filter(request_data):
                continue
            request_data_list.append(request_data)

        return request_data_list

    async def _generate(self):
        while True:
            request_data_list: List[SequenceRequestData] = await self.fetch_data()
            

    async def generate_stream(self, request_data: SequenceRequestData):
        self.prefill_queue.put_nowait(request_data)

        # 等待队列完成
        ...

    async def abort(self, request_id: str):
        # 从 prefill_queue 和 decoding_queue 中移除 request_id
        self.abort_queue.put_nowait(request_id)
```


## v0 性能测试

Mac Mini M4 (16G) 

|                                      | `mlx-community/Llama-3.2-1B-Instruct-4bit` | `mlx-community/Llama-3.2-1B-Instruct` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` |
| ------------------------------------ | -------------------------------------------- | --------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Engine, Baseline | 98.10 tok/s                                 | 35.45 tok/s                             | 20.68 tok/s                                       | No Memory                                         |
| Local    | 61.83 tok/s                                 | 34.54 tok/s                             | 14.91 tok/s                                       | No Memory                                         |
| Mac Mini M4 (16G) + M3 Pro (18G)     |                                              | 16.33 tok/s                             | 11.06 tok/s                                       | 5.64 tok/s                                        |


Q: Why Local is slower than Server+Client?

A:

- Local 只有一个进程，启动了 HTTP Serve， Engine 和 Model 都在一个进程中
- Server+Client 是两个进程，Server 中包含了 HTTP Serve 和 Engine，以及 Embedding 和 LM HEAD；Client 中只有 Model

但不清楚，为什么 `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` 这个不大一样，暂时归因到内存压力上。

Q: Mac Mini M4 (16G) + M3 Pro (18G) 这一列速度为什么慢？

A：理想情况下会等于 Mac Mini M4 (16G) (Server+Client)，但由于需要进行通信，通信开销占了主要部分，其中主要是延迟问题导致每个 token 生成都需要花费一定时间，哪怕在局域网内。
