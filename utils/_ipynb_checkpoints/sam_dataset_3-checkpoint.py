 # from .functions import *
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing as mp
from torchvision.transforms import CenterCrop
import random
from PIL import Image
import numpy as np
import random

path_label = r'label_sin_1024_val/'
path_img = r'image_sin_1024_val/'


def readlist(path):
    filelist = []
    for item in os.listdir(path):
        if item[-3:] == "tif":
            label = Image.open(path + item)
            label = np.array(label)
            # print(np.sum(label > 0))
            if np.sum(label > 0) > 50:
                filelist.append(item)
    return filelist


filelist = readlist(path=path_label)
filelist_train = random.sample(filelist, 700)
print('filelist_train.lenth:', len(filelist_train))
filelist_test = [item for item in filelist if item not in filelist_train]
filelist_test = filelist_test[0:70]
print('filelist_test.lenth:', len(filelist_test))
crop_obj = CenterCrop((1000, 1000))


def loadlabel(path_label=path_label, filelist=filelist, index=0):
    patch_label = np.ones([len(filelist), 1024, 1024])
    for item in filelist:
        # print(item)
        label = Image.open(path_label + item.split('.')[0] + '.tif')
        label = np.array(label)
        # print(label.shape)
        if len(label.shape) == 3:
            label = label[:,:,0]
            label = np.array(label,dtype=bool)
        elif len(label.shape) == 2:
            label = np.array(label,dtype=bool)
        # print(label.shape)
        # label = label[np.newaxis, :]
        # label = torch.from_numpy(label)
        # label = crop_obj(label)
        # label = label.numpy()
        # print(label)
        # label[label > 0] = 1
        # print(label.shape)
        # print(label)
        patch_label[index,] = label
        index += 1
    return patch_label


def loadimg(path_img=path_img, filelist=filelist, index=0):
    patch_img = np.ones([len(filelist), 1024, 1024, 3])
    for item in filelist:
        img = Image.open(path_img + item.split('.')[0] + '.tif')
        img = np.array(img)
        img = img[np.newaxis, :]
        img = img.transpose(0, 3, 1, 2)
        # print(img.shape)
        img = torch.from_numpy(img)
        # img = crop_obj(img)
        img = img.numpy()
        img = img.transpose(0, 2, 3, 1)
        # print("img.shape:",img.shape)
        patch_img[index,] = img
        index += 1
    return patch_img


class SamDataset_train(torch.utils.data.Dataset):
    def __init__(self):
        Xtrain = loadimg(path_img=path_img, filelist=filelist_train, index=0)
        ytrain = loadlabel(path_label=path_label, filelist=filelist_train, index=0)
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

class SamDataset_test(torch.utils.data.Dataset):
    def __init__(self):
        Xtest = loadimg(path_img=path_img, filelist=filelist_test, index=0)
        ytest = loadlabel(path_label=path_label, filelist=filelist_test, index=0)
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

