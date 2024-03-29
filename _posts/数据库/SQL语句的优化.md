---
title: SQL语句的优化
date: 2018-04-07 12:32:12
tags: [SQL, 数据库]
categories: [Database]
---
这篇文章主要介绍了关于SQL语句（主要是Oracle数据库）的优化方案。
<!-- more -->

[数据库性能优化之SQL语句优化](https://juejin.im/post/5ab0f39df265da23866fba55)

## 操作符优化

### IN 操作符

优点：编写容易，清晰易懂

缺点：性能较低。

Oracle执行步骤：

1. 将其转换多表连接
2. 如果转换不成功则先执行IN里面的子查询，再执行外层表记录
3. 如果转换成功则直接采用多表连接查询

所以用IN的SQL至少会多一个转换过程。

替代方案：在业务密集的SQL用EXISTS替代

### NOT IN操作符

不能应用表的索引，不建议使用。

替代方案：用NOT EXISTS方案

### IS NULL或IS NOT NULL操作

判断字段为空一般是不会用索引的（索引不索引空值）。如果某列存在空值，即使对该列建索引也不会提高性能。

替代方案：将`a is not null`改为`a>0`或`a>' '`等。不允许字段为空，而用一个缺省值代替空值。

### < 及 > 操作符

一般情况下是不需要调整的，在某些情况下可以进行优化。

如：一个表有100万记录，一个数值型字段A，有30万的A=0，39万A=2， 1万A=3。在这种情况下执行A>2与A>=3有很大区别，`A>2`在Oracle中会先找出为2的记录索引再进行比较，但是`A>=3`会直接找到等于3的记录索引。

### LIKE操作符

LIKE医用通配符查询，基本上可以达到任一查询。但是不同语句可能会产生性能上的很大差异。如`LIKE '%5400%'`这种情况下不会引用索引，`LIKE 'X5400%'`会引用范围索引。即前者会产生全表扫描，后者会进行范围内的查询，后者相对于前者的性能大大提高了。

### UNION操作符

UNION在进行表连接之后会筛选掉重复的记录，所以在表连接之后会对产生的结果集进行排序运算，删除重复记录再返回结果。但是实际上大部分应用中是不会产生重复的记录，最常见的是过程表于历史表UNION。在这种情况下，替代方案就是使用UNION ALL代替UNION。UNION ALL操作只是简单的将两个结果合并后就返回

### 联接列

对于有联接的列，优化器是不会使用索引的。

```sql
# 使用联接查询
select * from employss where first_name||''||last_name ='Beill Cliton';
# 不使用联接查询，基于last_name创建索引
where first_name ='Beill' and last_name ='Cliton';
```

### Order By语句

将返回的查询结果进行排序。任何在Order by语句的非索引项或者有计算表达式都将降低查询速度。解决方案就是为所使用的列建立另一个索引，同时也绝对避免在order by子句中使用表达式。

### NOT 语句

NOT可用来对任何逻辑运算符号取反。

```sql
# 如果要使用NOT，则应在取反的短语前面机上括号，并在短语前面机上NOT运算符。
where not (status = 'VALID')
# NOT运算符包含在另外一个逻辑运算符<>运算符中。即使不在查询的where子句中显式地加入NOT词，NOT仍在运算符中
where status <> 'INVALID';
# 这种情况运行建立salary的索引，在速度上会优于上面的方案
select * from employee where salary < 3000 or salary > 3000;
```

## 书写的影响

### 同一功能同一性能不同写法SQL的影响

尽可能避免不同写法的出现，根据Oracle共享内存SGA原理，如果语句不同则进行分析，占用共享内存。如果SQL字符串及格式完全相同，则Oracle只会分析一次，共享内存也只会留下一次的分析结果

### WHERE后面的条件顺序影响

下面两个语句，在没有建立索引的情况下，执行都是全表扫描。

`dy_dj = '1KV以下' `这个条件在记录集内比率为99%, `xh_bz=1`而这个条件只有0.5%。所以第二条SQL的CPU占用率明显低于第一条。

```sql
Select * from zl_yhjbqk where dy_dj = '1KV以下' and xh_bz=1
Select * from zl_yhjbqk where xh_bz=1 and dy_dj = '1KV以下'
```

### 查询表顺序的影响

在FROM后面的表中的列表顺序会对SQL执行性能影响，在没有索引及ORACLE没有对表进行统计分析的情况下，ORACLE会按**表出现的顺序**进行连接，所以如果表的顺序不对时会产生十分消耗服务器资源的数据交叉（如果对表进行了统计分析，Oracle自动先进小表的连接，再进行大表的连接）

## SQL语句索引的利用

### 条件字段的优化

函数处理的字段不能利用索引

```sql
原语句：substr(hbs_bh,1,4)='5400'
优化处理：hbs_bh like '5400%'

原语句：trunc(sk_rq)=trunc(sysdate)
优化处理：sk_rq>=trunc(sysdate) and sk_rq<trunc(sysdate+1)
```

进行了显示或隐式的运算的字段不能进行索引。

```sql
原语句：'X' || hbs_bh>'X5400021452'
优化处理：hbs_bh>'5400021542'

原语句：sk_rq+5=sysdate
优化处理：sk_rq=sysdate-5
```

条件内包括的多个本表的字段运算时不能进行索引

```sql
原语句：ys_df>cx_df
无法进行优化

原语句：qc_bh || kh_bh='5400250000'
优化处理：qc_bh='5400' and kh_bh='250000'
```

## 其他优化方法

1. 选择最有效率的表名顺序（只在基于规则的优化器中有效）

   `ORACLE` 的解析器按照从右到左的顺序处理FROM子句中的表名，FROM子句中写在最后的表(基础表 driving table)将被最先处理，在FROM子句中包含多个表的情况下,你必须选择记录条数最少的表作为基础表。如果有3个以上的表连接查询, 那就需要选择交叉表(intersection table)作为基础表, 交叉表是指那个被其他表所引用的表

2. `WHERE`子句中的连接顺序

   `ORACLE`采用自下而上的顺序解析WHERE子句,根据这个原理,表之间的连接必须写在其他`WHERE`条件之前, 那些可以过滤掉最大数量记录的条件必须写在WHERE子句的末尾.

3. `SELECT`子句中避免使用 '*'

   `ORACLE`在解析的过程中, 会将'*'依次转换成所有的列名, 这个工作是通过查询数据字典完成的, 这意味着将耗费更多的时间。

4. 减少访问数据库的次数

   `ORACLE`在内部执行了许多工作: 解析SQL语句, 估算索引的利用率, 绑定变量 , 读数据块等。

5. 在`SQL*Plus` , `SQL*Forms`和`Pro*C`中重新设置`ARRAYSIZE`参数, 可以增加每次数据库访问的检索数据量 ,建议值为200

6. 使用`DECODE`函数来减少处理时间

   使用`DECODE`函数可以避免重复扫描相同记录或重复连接相同的表.

7.  整合简单,无关联的数据库访问

   如果你有几个简单的数据库查询语句,你可以把它们整合到一个查询中(即使它们之间没有关系) 。

8.  删除重复记录：

> 最高效的删除重复记录方法(因为使用了`ROWID`)

```sql
DELETE FROM EMP E WHERE E.ROWID > (SELECT MIN(X.ROWID) FROM EMP X WHERE X.EMP_NO = E.EMP_NO)
```
9. 用`TRUNCATE`替代`DELETE`

   当删除表中的记录时,在通常情况下, 回滚段(rollback segments ) 用来存放可以被恢复的信息. 如果你没有`COMMIT`事务,`ORACLE`会将数据恢复到删除之前的状态(准确地说是恢复到执行删除命令之前的状况) 而当运用`TRUNCATE`时, 回滚段不再存放任何可被恢复的信息.当命令运行后,数据不能被恢复.因此很少的资源被调用,执行时间也会很短. (译者按: `TRUNCATE`只在删除全表适用,`TRUNCATE`是DDL不是DML) 。

10. 尽量多使用`COMMIT`

    只要有可能,在程序中尽量多使用`COMMIT`, 这样程序的性能得到提高,需求也会因为`COMMIT`所释放的资源而减少，`COMMIT`所释放的资源:

    a. 回滚段上用于恢复数据的信息.

    b. 被程序语句获得的锁

    c. redo log buffer 中的空间

    d. `ORACLE`为管理上述3种资源中的内部花费

11.  用`Where`子句替换`HAVING`子句

    避免使用`HAVING`子句, `HAVING` 只会在检索出所有记录之后才对结果集进行过滤. 这个处理需要排序,总计等操作. 如果能通过`WHERE`子句限制记录的数目,那就能减少这方面的开销. (非`oracle`中)`on`、`where`、`having`这三个都可以加条件的子句中，`on`是最先执行，`where`次之，`having`最后，因为`on`是先把不符合条件的记录过滤后才进行统计，它就可以减少中间运算要处理的数据，按理说应该速度是最快的，`where`也应该比`having`快点的，因为它过滤数据后才进行`sum`，在两个表联接时才用`on`的，所以在一个表的时候，就剩下`where`跟`having`比较了。在这单表查询统计的情况下，如果要过滤的条件没有涉及到要计算字段，那它们的结果是一样的，只是`where`可以使用`rushmore`技术，而`having`就不能，在速度上后者要慢如果要涉及到计算的字 段，就表示在没计算之前，这个字段的值是不确定的，根据上篇写的工作流程，`where`的作用时间是在计算之前就完成的，而`having`就是在计算后才起作 用的，所以在这种情况下，两者的结果会不同。在多表联接查询时，`on`比`where`更早起作用。系统首先根据各个表之间的联接条件，把多个表合成一个临时表 后，再由`where`进行过滤，然后再计算，计算完后再由`having`进行过滤。由此可见，要想过滤条件起到正确的作用，首先要明白这个条件应该在什么时候起作用，然后再决定放在那里。

12. 减少对表的查询

> 在含有子查询的SQL语句中,要特别注意减少对表的查询

```sql
SELECT TAB_NAME FROM TABLES WHERE (TAB_NAME,DB_VER) = ( SELECT TAB_NAME,DB_VER FROM TAB_COLUMNS WHERE VERSION = 604)
```

13. 通过内部函数提高SQL效率

    复杂的SQL往往牺牲了执行效率. 能够掌握上面的运用函数解决问题的方法在实际工作中是非常有意义的。

14.  使用表的别名(`Alias`)：

    当在SQL语句中连接多个表时, 请使用表的别名并把别名前缀于每个Column上.这样一来,就可以减少解析的时间并减少那些由Column歧义引起的语法错误。

15.  识别’低效执行’的SQL语句

    虽然目前各种关于SQL优化的图形化工具层出不穷,但是写出自己的SQL工具来解决问题始终是一个最好的方法。

​```sql
SELECT EXECUTIONS, DISK_READS, BUFFER_GETS,
ROUND((BUFFER_GETS-DISK_READS)/BUFFER_GETS,2) Hit_radio,
ROUND(DISK_READS/EXECUTIONS,2) Reads_per_run,
SQL_TEXT
FROM V$SQLAREA
WHERE EXECUTIONS>0
AND BUFFER_GETS > 0
AND (BUFFER_GETS-DISK_READS)/BUFFER_GETS < 0.8
ORDER BY 4 DESC;
​```

16. 用索引提高效率

    索引是表的一个概念部分,用来提高检索数据的效率，ORACLE使用了一个复杂的自平衡B-tree结构. 通常,通过索引查询数据比全表扫描要快. 当ORACLE找出执行查询和Update语句的最佳路径时, ORACLE优化器将使用索引. 同样在联结多个表时使用索引也可以提高效率. 另一个使用索引的好处是,它提供了主键(primary key)的唯一性验证.。那些LONG或LONG RAW数据类型, 你可以索引几乎所有的列. 通常, 在大型表中使用索引特别有效. 当然,你也会发现, 在扫描小表时,使用索引同样能提高效率. 虽然使用索引能得到查询效率的提高,但是我们也必须注意到它的代价. 索引需要空间来存储,也需要定期维护, 每当有记录在表中增减或索引列被修改时, 索引本身也会被修改. 这意味着每条记录的INSERT , DELETE , UPDATE将为此多付出4 , 5 次的磁盘I/O . 因为索引需要额外的存储空间和处理,那些不必要的索引反而会使查询反应时间变慢.。定期的重构索引是有必要的

```sql
ALTER INDEX <INDEXNAME> REBUILD <TABLESPACENAME>
```

17. 用`EXISTS`替换`DISTINCT`

    当提交一个包含一对多表信息(比如部门表和雇员表)的查询时,避免在`SELECT`子句中使用`DISTINCT`. 一般可以考虑用`EXIST`替换, `EXISTS` 使查询更为迅速,因为RDBMS核心模块将在子查询的条件一旦满足后,立刻返回结果. 例子：

```sql
(低效):
SELECT DISTINCT DEPT_NO, DEPT_NAME FROM DEPT D, EMP E WHERE D.DEPT_NO = E.DEPT_NO
(高效):
SELECT DEPT_NO, DEPT_NAME FROM DEPT D WHERE EXISTS ( SELECT 'X' FROM EMP E WHERE E.DEPT_NO = D.DEPT_NO);
```

18. sql语句用大写的

    因为oracle总是先解析sql语句，把小写的字母转换成大写的再执行

19. 在java代码中尽量少用连接符“＋”连接字符串！

20. 避免在索引列上使用NOT，通常我们要避免在索引列上使用NOT, NOT会产生在和在索引列上使用函数相同的影响. 当ORACLE”遇到”NOT,他就会停止使用索引转而执行全表扫描。

21. 避免在索引列上使用计算

    WHERE子句中，如果索引列是函数的一部分．优化器将不使用索引而使用全表扫描．

```sql
低效：
SELECT * FROM DEPT WHERE SAL * 12 > 25000;
高效:
SELECT * FROM DEPT WHERE SAL > 25000/12;
```

22. 用>=替代>

```sql
高效:
SELECT * FROM EMP WHERE DEPTNO >= 4
低效:
SELECT * FROM EMP WHERE DEPTNO > 3
```

23. 用UNION替换OR (适用于索引列)

    通常情况下, 用UNION替换WHERE子句中的OR将会起到较好的效果. 对索引列使用OR将造成全表扫描. 注意, 以上规则只针对多个索引列有效. 如果有column没有被索引, 查询效率可能会因为你没有选择OR而降低. 在下面的例子中, LOC_ID 和REGION上都建有索引.

```sql
高效:
SELECT LOC_ID , LOC_DESC , REGION
FROM LOCATION
WHERE LOC_ID = 10
UNION
SELECT LOC_ID , LOC_DESC , REGION
FROM LOCATION
WHERE REGION = "MELBOURNE"
低效:
SELECT LOC_ID , LOC_DESC , REGION
FROM LOCATION
WHERE LOC_ID = 10 OR REGION = "MELBOURNE"
```

24. 用IN来替换OR

    这是一条简单易记的规则，但是实际的执行效果还须检验，在ORACLE8下，两者的执行路径似乎是相同的．

```sql
低效:
SELECT…. FROM LOCATION WHERE LOC_ID = 10 OR LOC_ID = 20 OR LOC_ID = 30
高效
SELECT… FROM LOCATION WHERE LOC_IN IN (10,20,30);
```

25. 避免在索引列上使用IS NULL和IS NOT NULL

    避免在索引中使用任何可以为空的列，ORACLE将无法使用该索引．对于单列索引，如果列包含空值，索引中将不存在此记录. 对于复合索引，如果每个列都为空，索引中同样不存在此记录. 如果至少有一个列不为空，则记录存在于索引中．举例: 如果唯一性索引建立在表的A列和B列上, 并且表中存在一条记录的A,B值为(123,null) , ORACLE将不接受下一条具有相同A,B值（123,null）的记录(插入). 然而如果所有的索引列都为空，ORACLE将认为整个键值为空而空不等于空. 因此你可以插入1000 条具有相同键值的记录,当然它们都是空! 因为空值不存在于索引列中,所以WHERE子句中对索引列进行空值比较将使ORACLE停用该索引.

```sql
低效: (索引失效)
SELECT * FROM DEPARTMENT WHERE DEPT_CODE IS NOT NULL;
高效: (索引有效)
SELECT * FROM DEPARTMENT WHERE DEPT_CODE >=0;
```

26. 总是使用索引的第一个列

    如果索引是建立在多个列上, 只有在它的第一个列(leading column)被where子句引用时,优化器才会选择使用该索引. 这也是一条简单而重要的规则，当仅引用索引的第二个列时,优化器使用了全表扫描而忽略了索引。

27. 用UNION-ALL 替换UNION ( 如果有可能的话)

    当SQL 语句需要UNION两个查询结果集合时,这两个结果集合会以UNION-ALL的方式被合并, 然后在输出最终结果前进行排序. 如果用UNION ALL替代UNION, 这样排序就不是必要了. 效率就会因此得到提高. 需要注意的是，UNION ALL 将重复输出两个结果集合中相同记录. 因此各位还是要从业务需求分析使用UNION ALL的可行性. UNION 将对结果集合排序,这个操作会使用到SORT_AREA_SIZE这块内存. 对于这块内存的优化也是相当重要的. 下面的SQL可以用来查询排序的消耗量

```sql
低效：
SELECT ACCT_NUM, BALANCE_AMT
FROM DEBIT_TRANSACTIONS
WHERE TRAN_DATE = '31-DEC-95'
UNION
SELECT ACCT_NUM, BALANCE_AMT
FROM DEBIT_TRANSACTIONS
WHERE TRAN_DATE = '31-DEC-95'
高效:
SELECT ACCT_NUM, BALANCE_AMT
FROM DEBIT_TRANSACTIONS
WHERE TRAN_DATE = '31-DEC-95'
UNION ALL
SELECT ACCT_NUM, BALANCE_AMT
FROM DEBIT_TRANSACTIONS
WHERE TRAN_DATE = '31-DEC-95'
```

28. 用WHERE替代ORDER BY

    ORDER BY 子句只在两种严格的条件下使用索引.

    ORDER BY中所有的列必须包含在相同的索引中并保持在索引中的排列顺序.

    ORDER BY中所有的列必须定义为非空.

    WHERE子句使用的索引和ORDER BY子句中所使用的索引不能并列.

```sql
低效: (索引不被使用)
SELECT DEPT_CODE FROM DEPT ORDER BY DEPT_TYPE
高效: (使用索引)
SELECT DEPT_CODE FROM DEPT WHERE DEPT_TYPE > 0
```

29. 避免改变索引列的类型

    当比较不同数据类型的数据时, ORACLE自动对列进行简单的类型转换.

```sql
# 假设 EMPNO是一个数值类型的索引列.
# 低效
SELECT … FROM EMP WHERE EMPNO = '123'
# 高效
SELECT … FROM EMP WHERE EMPNO = 123
```

30. 需要当心的WHERE子句

    某些SELECT 语句中的WHERE子句不使用索引. 这里有一些例子.

    1. `!=` 将不使用索引. 记住, 索引只能告诉你什么存在于表中, 而不能告诉你什么不存在于表中.
    2. `¦ ¦`是字符连接函数. 就象其他函数那样, 停用了索引.
    3. `+`是数学函数. 就象其他数学函数那样, 停用了索引.
    4. 相同的索引列不能互相比较,这将会启用全表扫描.

31. a. 如果检索数据量超过30%的表中记录数.使用索引将没有显著的效率提高.

    b. 在特定情况下, 使用索引也许会比全表扫描慢, 但这是同一个数量级上的区别. 而通常情况下,使用索引比全表扫描要块几倍乃至几千倍!

32. 避免使用耗费资源的操作

    带有DISTINCT,UNION,MINUS,INTERSECT,ORDER BY的SQL语句会启动SQL引擎执行耗费资源的排序(SORT)功能. DISTINCT需要一次排序操作, 而其他的至少需要执行两次排序. 通常, 带有UNION, MINUS , INTERSECT的SQL语句都可以用其他方式重写. 如果你的数据库的SORT_AREA_SIZE调配得好, 使用UNION , MINUS, INTERSECT也是可以考虑的, 毕竟它们的可读性很强。

33. 优化GROUP BY

    提高GROUP BY 语句的效率, 可以通过将不需要的记录在GROUP BY 之前过滤掉.下面两个查询返回相同结果但第二个明显就快了许多.

```sql
低效:
SELECT JOB, AVG(SAL)
FROM EMP
GROUP by JOB
HAVING JOB = 'PRESIDENT'
OR JOB = 'MANAGER'
高效:
SELECT JOB, AVG(SAL)
FROM EMP
WHERE JOB = 'PRESIDENT'
OR JOB = 'MANAGER'
GROUP by JOB
```

    ​

