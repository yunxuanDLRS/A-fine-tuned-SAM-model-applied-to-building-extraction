# -*- coding:utf8 -*-

#读取一个文件夹下的图片名字并保存到一个新建的txt文件中

import os
from PIL import Image
import numpy as np
import datetime

# 获取当前日期
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")

# 设置文件名和后缀
filename = f"{date}.txt"
suffix = 0

# 如果文件已经存在，则增加一个数字后缀
while os.path.isfile(filename):
    suffix += 1
    filename = f"{date}_{suffix}.txt"


# path = 'train/image/'
path = 'label_sin_1024_val/'
filelist = os.listdir(path)
num = 0
for item in filelist:
    print(item)
    num+=1
    with open(filename, "a") as f:
        f.write(item + "\n")
    
print("total num:",num)
