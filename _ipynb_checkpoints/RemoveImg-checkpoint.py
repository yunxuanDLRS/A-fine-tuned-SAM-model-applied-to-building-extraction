
# 按照一定条件删除某个文件夹下的一些图片

import os.path

#在员名称的基础上改名
def rename(img_folder,num):
    for img_name in os.listdir(img_folder):  # os.listdir()： 列出路径下所有的文件
        #os.path.join() 拼接文件路径

        if '-' in img_name :
            os.remove(os.path.join(img_folder, img_name))
            num+=1
    print("删除重复文件{}张".format(num))

def main():
    img_folder0 = r'label_BJ/' #图片的文件夹路径
    num=0
    n=rename(img_folder0,num)

if __name__=="__main__":
    main()


# import os.path

# #
# def rename(img_folder,num):
#     for img_name in os.listdir(img_folder):  # os.listdir()： 列出路径下所有的文件
#         #os.path.join() 拼接文件路径
#         if (img_name == '736091_2_1.jpg.tmp'):
#             os.remove(os.path.join(img_folder, img_name))
#             num+=1
#     print("删除错误文件{}张".format(num))

# def main():
#     img_folder0 = r'img4/' #图片的文件夹路径
#     num=0
#     n=rename(img_folder0,num)

# if __name__=="__main__":
#     main()
