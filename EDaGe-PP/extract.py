import os
import shutil

old_path = r'/home/long/dataset/planning_origin'  # 要复制的文件所在目录
new_path = r'/home/long/dataset/planning_500'  #新路径

folder = [cla for cla in os.listdir(old_path) if os.path.isdir(os.path.join(old_path, cla))]
folder.sort()
print(folder)
supported = [".jpg", ".JPG", ".png", ".PNG"]
for cla in folder:
    cla_path = os.path.join(old_path, cla)
    images = [i for i in os.listdir(cla_path)
            if os.path.splitext(i)[-1] in supported]
    images_copy = images.copy()
    for i in images_copy:
        images[int(i[4:-4])] = i
    images = [os.path.join(old_path, cla, i) for i in images]
    images = images[0:500]
    data = os.path.join(old_path, cla, 'data')
    GMM = os.path.join(old_path, cla, 'GMM')
    print(data, GMM)
    print(images)
    path = os.path.join(new_path, cla)
    if not os.path.exists(path):
        os.mkdir(path)
    print(path)
    shutil.copytree(data, os.path.join(path, 'data'))
    shutil.copytree(GMM, os.path.join(path, 'GMM'))
    for item in images:
        shutil.copy(item, path)