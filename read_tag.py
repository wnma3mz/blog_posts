import os
import glob
import re
if __name__ == "__main__":
    root_dir = "_posts"
    tag_lst, category_lst = [], []
    for f in os.listdir(f"./{root_dir}"):
        with open(os.path.join(root_dir, f), "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("tags"):
                tags = re.findall(r'tags:\s*\[(.*?)\]', line)[0]
            if line.startswith("categories"):
                categories = re.findall(r'categories:\s*\[(.*?)\]', line)[0]
                break
        tag_lst += tags.split(",")
        category_lst += categories.split(",")
    print("tag", set(tag_lst))
    print("category", set(category_lst))
