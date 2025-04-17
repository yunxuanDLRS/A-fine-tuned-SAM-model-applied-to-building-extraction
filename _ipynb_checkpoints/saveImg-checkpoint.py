import os
import shutil

def copy_tiff_images(src_folder, dest_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 获取源文件夹中的所有文件
    files = os.listdir(src_folder)

    # 遍历文件夹中的文件
    for file_name in files:
        # 检查文件是否是TIFF文件
        if file_name.lower().endswith('.tif'):
            # 构建源文件和目标文件的路径
            src_path = os.path.join(src_folder, file_name)
            dest_path = os.path.join(dest_folder, file_name)

            # 使用shutil复制文件
            shutil.copy(src_path, dest_path)

if __name__ == "__main__":
    # 指定源文件夹和目标文件夹的路径
    source_folder = "image_sin_1225_1024"
    destination_folder = "image_sin_1024_val"

    # 调用函数复制TIFF格式的图片
    copy_tiff_images(source_folder, destination_folder)