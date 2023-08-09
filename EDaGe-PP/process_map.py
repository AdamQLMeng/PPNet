import datetime
import json
import os
import shutil
import time

import cv2
import imgviz
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from scipy.spatial import distance

from matplotlib import cm
import matplotlib.pyplot as plt  # 可视化的matplotlib库
from matplotlib.ticker import LinearLocator, FormatStrFormatter
torch.set_printoptions(threshold=np.inf)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_PER_FOLDER = 400


def time_synchronized():
    # torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def process_map(root: str = './', mode: str = 'SOD', is_record_init_end: bool = True):
    image_folder = os.path.join(root, 'original_data')
    folders = [cla for cla in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, cla))]
    print('There are {} folders are waiting to parse'.format(len(folders)))
    # 遍历每个文件夹下的文件
    for i in range(len(folders)):
        folder_index = str(i)
        cla_path = os.path.join(image_folder, folder_index)
        if not os.path.exists(cla_path):
            print("folder {} doesn't exist".format(cla_path))
            continue
        images, labels, spaces_obs = read_folder(cla_path, is_read_path=True)
        if len(images) != NUM_PER_FOLDER:
            print('Error: (read_folder) Incorrect length:', cla_path, len(images))
            continue

        rotation = []
        translation = []
        init_end = []
        path_point = []
        for item in labels:
            rotation.append(item[1])
            translation.append(item[2])
            init_end.append([item[3][0], item[3][10]])
            path_point.append(item[4])

        if is_record_init_end:
            print('recording init point and end point')
            record_init_end(images, init_end, folder_index, root)

        if mode == 'Gen_path':
            print('generating path generation masks from path point', len(spaces_obs))
            generate_gen_path(path_point, folder_index, root='{}/mask_path/'.format(root))
        elif mode == 'Seg_space':
            print('generating space Segmentation masks from path space', len(spaces_obs))
            generate_seg_space(spaces_obs, path_point, rotation, translation,  folder_index, root='{}/mask_space/'.format(root))
        elif mode == 'All':
            print('generating both two types of masks from path point', len(spaces_obs))
            generate_gen_path(path_point, folder_index, root='{}/mask_path/'.format(root))
            generate_seg_space(spaces_obs, path_point, rotation, translation, folder_index,
                               root='{}/mask_space/'.format(root))


def read_folder(root: str = './', is_read_path: bool = True):
    spaces_obs = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    print('reading the data in:', root)
    labels = torch.load(r'{}/data/MapLabel'.format(root))
    if is_read_path:
        spaces_folder = r'{}/data'.format(root)
        images = [i for i in os.listdir(spaces_folder)
                  if os.path.splitext(i)[-1] in supported]
        images_copy = images.copy()
        for i in images_copy:
            images[int(i[0:-4])] = i
        images = [os.path.join(spaces_folder, i) for i in images]
        for im in images:
            im = np.asarray(Image.open(im)).copy()
            spaces_obs.append(im)
    # 遍历获取supported支持的所有文件路径 并按数字索引排序
    images = [i for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    if len(images) != NUM_PER_FOLDER:
        return [os.path.join(root, i) for i in images], labels, spaces_obs
    images_copy = images.copy()
    for i in images_copy:
        images[int(i[0:-4])] = i
    images = [os.path.join(root, i) for i in images]
    # 获取样本标签
    labels = labels[0:len(images)]
    return images, labels, spaces_obs


def add_init_end(root: str = './', images=None, init_end=None):
    red = [255, 0, 0]
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    i = 0
    for init, end in init_end:
        im = np.asarray(Image.open(images[i])).copy()
        im = add_init_end_single(im, init, end)
        Image.fromarray(im).save(images[i])
        # plt.imshow(im)
        # plt.show()
        i = i + 1


def add_init_end_single(image, init, end):
    red = torch.tensor([255, 0, 0])
    assert len(image.shape) == 3, "Image shape incorrect"
    assert init is not None, "Init is None"
    assert end is not None, "End is None"

    # image = np.array(image).copy().reshape([image.shape[1], image.shape[2], image.shape[0]])
    resolution = image.shape[1]
    for j in range(4):
        for k in range(4):
            if 0 <= int(np.round(init[0]) + j) < resolution and 0 <= int(np.round(init[1]) + k) < resolution:
                image[:, int(np.round(init[0]) + j), int(np.round(init[1]) + k)] = red
            if 0 <= int(np.round(init[0]) + j) < resolution and 0 <= int(np.round(init[1]) - k) < resolution:
                image[:, int(np.round(init[0]) + j), int(np.round(init[1]) - k)] = red
            if 0 <= int(np.round(init[0]) - j) < resolution and 0 <= int(np.round(init[1]) + k) < resolution:
                image[:, int(np.round(init[0]) - j), int(np.round(init[1]) + k)] = red
            if 0 <= int(np.round(init[0]) - j) < resolution and 0 <= int(np.round(init[1]) - k) < resolution:
                image[:, int(np.round(init[0]) - j), int(np.round(init[1]) - k)] = red
            if 0 <= int(np.round(end[0]) + j) < resolution and 0 <= int(np.round(end[1]) + k) < resolution:
                image[:, int(np.round(end[0]) + j), int(np.round(end[1]) + k)] = red
            if 0 <= int(np.round(end[0]) + j) < resolution and 0 <= int(np.round(end[1]) - k) < resolution:
                image[:, int(np.round(end[0]) + j), int(np.round(end[1]) - k)] = red
            if 0 <= int(np.round(end[0]) - j) < resolution and 0 <= int(np.round(end[1]) + k) < resolution:
                image[:, int(np.round(end[0]) - j), int(np.round(end[1]) + k)] = red
            if 0 <= int(np.round(end[0]) - j) < resolution and 0 <= int(np.round(end[1]) - k) < resolution:
                image[:, int(np.round(end[0]) - j), int(np.round(end[1]) - k)] = red
    return image


def generate_gen_path(path_point, folder_index, root: str = './'):
    bias = 50
    if not os.path.exists(root):
        os.mkdir(root)
    for i in range(len(path_point)):
        point = path_point[i]
        space4seg = torch.zeros([3, 224, 224])
        space4seg = np.array((space4seg[0]))
        for step, p in enumerate(point):
            if step % 5 == 0 and 0 < int(np.round(p[0])) < 224 and 0 < int(np.round(p[1])) < 224:
                # if i < 255 - bias or i > len(point) - (255 - bias):
                #     space4seg[int(np.round(p[0])), int(np.round(p[1]))] = bias + i
                # else:
                space4seg[int(np.round(p[0])), int(np.round(p[1]))] = 255
        space4seg = torchvision.transforms.ToPILImage()(space4seg).convert(mode='L')
        space4seg.save('{}/{}.png'.format(root, int(folder_index)*NUM_PER_FOLDER+i))


def generate_seg_space(spaces, path_point, rotation, translation, folder_index, root: str = './'):
    if not os.path.exists(root):
        os.mkdir(root)
    colormap = imgviz.label_colormap()
    for i in range(len(path_point)):
        point = path_point[i]
        r = rotation[i]
        t = translation[i]
        rotation_f = T.RandomRotation(degrees=(-r, -r))
        space4seg = spaces[int(i/(len(path_point)/len(spaces)))]

        space4seg = torchvision.transforms.ToTensor()(space4seg).to(device)
        space4seg = rotation_f(space4seg)
        space4seg = T.functional.affine(space4seg, translate=t, angle=0, scale=1, shear=0)
        space4seg = np.array((space4seg[0]).cpu())
        _, space4seg = cv2.threshold(src=space4seg,  # 要二值化的图片
                                      thresh=0.5,  # 全局阈值
                                      maxval=1,  # 大于全局阈值后设定的值
                                      type=cv2.THRESH_BINARY)  # 设定的二值化类型，THRESH_BINARY：表示小于阈值置0，大于阈值置填充色
        # space4seg = space4seg * 2
        # for p in point:
        #     if 0 < int(np.round(p[0])) < 224 and 0 < int(np.round(p[1])) < 224:
        #         space4seg[int(np.round(p[0])), int(np.round(p[1]))] = 2
        space4seg = torchvision.transforms.ToPILImage()(space4seg).convert(mode='P')
        space4seg.putpalette(colormap.flatten())
        space4seg.save('{}/{}.png'.format(root, int(folder_index)*NUM_PER_FOLDER+i))


def coordi_rotation(x, radians):
    rotation = np.reshape([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]], [2, 2])
    return np.dot(rotation, x)


def coordi_euclidean2image(x, mapoffset):
    x = np.reshape(x, [-1, 2])
    coordi = []
    for point in x:
        step_len = 1 / 224 * 50
        coordi_x = int(np.round(point[0] / step_len + mapoffset))
        coordi_y = int(np.round(point[1] / step_len + mapoffset))
        coordi.append([coordi_x, coordi_y])
    return np.array(coordi)


def move_data(root: str = './'):
    image_path = os.path.join(root, 'original_data')
    assert os.path.exists(image_path), f"data {image_path} not found."
    image_des_path = os.path.join(root, 'map')
    if not os.path.exists(image_des_path):
        os.mkdir(image_des_path)

    folders = [cla for cla in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, cla))]
    print('There are {} folders are waiting to parse'.format(len(folders)))
    # 遍历每个文件夹下的文件
    for f in folders:
        folder_index = f.split('/')[-1]
        cla_path = os.path.join(image_path, folder_index)
        if not os.path.exists(cla_path):
            print("folder {} doesn't exist".format(cla_path))
            continue
        images, _, _ = read_folder(cla_path, is_read_path=False)
        if len(images) != NUM_PER_FOLDER:
            print('Error: (read_folder) Incorrect length:', cla_path, len(images))
            # continue
        for im in images:
            index = im.split('/')[-1].split('.')[0]
            dst = os.path.join(image_des_path, '{}.jpg'.format(int(folder_index)*NUM_PER_FOLDER+int(index)))
            shutil.copy(im, dst)


def record_init_end(file_names, init_end, folder_index, txt_root: str='./'):
    info_dict = {'image': None, 'init': None, 'end': None}

    txt_file = "{}/init_end.txt".format(txt_root)
    with open(txt_file, "a") as f:
        for img, i_e in zip(file_names, init_end):
            info_dict['image'] = '{}/{}.jpg'.format(txt_root,
                                                    int(folder_index) * NUM_PER_FOLDER + int(img.split('/')[-1].split('.')[0]))
            info_dict['init'] = str(i_e[0])
            info_dict['end'] = str(i_e[1])
            f.write(str(info_dict) + "\n")


def generate_txt(root: str='./'):
    data_root = root + '/map'
    txt_root = root + '/ImageSets/Segmentation'
    if not os.path.exists(txt_root):
        os.mkdir(txt_root)
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    images = [str(i.split('.')[0]) for i in os.listdir(data_root)
              if os.path.splitext(i)[-1] in supported]
    print("There is {} file in folder {}".format(len(images), root))
    images.sort()

    txt_file = "{}/test.txt".format(txt_root)
    if os.path.exists(txt_file):
        os.remove(txt_file)
    with open(txt_file, "a") as f:
        for item in images:
            f.write(item + "\n")
    if len(images) > 50000:
        txt_file = "{}/train.txt".format(txt_root)
        with open(txt_file, "a") as f:
            for item in images:
                f.write(item + "\n")
        txt_file = "{}/val.txt".format(txt_root)
        with open(txt_file, "a") as f:
            for item in images[-5000:]:
                f.write(item + "\n")


def plot_3D_heatmap(picture):
    assert len(picture.shape) == 2
    x = np.arange(0, picture.shape[0], 1)
    y = np.arange(0, picture.shape[1], 1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x, y, picture, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_zlim(picture.min(), picture.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def extract_path(mask, init_state, end_state, down_sample_rate=8):
    motions = [
                      [0, 1], [0, -1], [1, 0], [-1, 0],
                      [1, 1], [1, -1], [-1, 1], [-1, -1],
                      ]
    init = init_state / down_sample_rate
    end = end_state / down_sample_rate
    mask_orig = mask
    mask = mask_orig.resize((int(mask_orig.size[0]/down_sample_rate), int(mask_orig.size[1]/down_sample_rate)), Image.BILINEAR)
    mask = T.ToTensor()(mask).squeeze()
    inference_path = []
    dist = []
    next_point = init
    t_start = time_synchronized()
    while True:
        t = time.time()
        if t-t_start > 1:
            print("Time out!")
            return False, None
        candidate = []
        for m in motions:
            candidate.append(torch.tensor(m) + next_point)
        candidate_v = []
        for c in candidate:
            if 0 <= int(np.round(c[0].item())) < int(mask_orig.size[0]/down_sample_rate) and 0 <= int(np.round(c[1].item())) < int(mask_orig.size[1]/down_sample_rate):
                candidate_v.append(mask[int(np.round(c[0].item())), int(np.round(c[1].item()))])
            else:
                candidate_v.append(0)
        while max(candidate_v) > 0:
            candidate_i = candidate_v.index(max(candidate_v))
            next_point = candidate[candidate_i]
            not_existence = True
            for i, p in enumerate(inference_path):
                if torch.equal(next_point, p) or (distance.euclidean(next_point, p) <= 1.5 and i < len(inference_path) - 2) or (distance.euclidean(next_point, p) <= 1.5 and i < len(inference_path) - 3):
                    # print(torch.equal(next_point, p), distance.euclidean(next_point, p) <= 2, distance.euclidean(next_point, p), i)
                    candidate_v[candidate_i] = 0
                    not_existence = False
                    break
                else:
                    not_existence = True
            if not_existence:
                break
        if max(candidate_v) == 0:
            print('inference failed {} {}'.format(len(inference_path), candidate_v))
            # plt.imshow(T.ToPILImage()(mask))
            # plt.plot(init[1], init[0], 'ro')
            # plt.plot(end[1], end[0], 'go')
            # for p in inference_path:
            #     plt.plot(p[1], p[0], 'bo')
            # plt.show()
            return False, None
        inference_path.append(next_point)
        dist.append(distance.euclidean(next_point, end))
        if distance.euclidean(next_point, end) <= 2.5:
            print('Extract success!', 'Number of Waypoint:', len(inference_path))
            t_end = time_synchronized()
            print('time consumption:', (t_end - t_start)*1000, 'ms')
            # inference_path.append(end_state)
            # for p in inference_path[1:-1]:  # skip init and end
            #     p *= down_sample_rate
            #     p = p.reshape([1, 2])
            for i in range(len(inference_path)):
                inference_path[i] *= down_sample_rate
                inference_path[i] = inference_path[i].reshape([1, 2])
            inference_path.append(torch.tensor(end_state).reshape([1, 2]))
            inference_path = torch.cat(inference_path, dim=0)
            inference_path = torch.cat([torch.tensor(init_state).reshape([1, 2]), inference_path], dim=0)
            # plt.imshow(mask_orig)
            # plt.plot(init[1]*down_sample_rate, init[0]*down_sample_rate, 'ro')
            # plt.plot(end[1]*down_sample_rate, end[0]*down_sample_rate, 'go')
            # plt.plot(inference_path.T[1], inference_path.T[0], 'bo')
            # plt.show()
            return True, inference_path


def plot_obstacles(size: tuple, obstacles, resolution: tuple = (224, 224)):
    ax = plt.gca()
    ax.axis(xmin=0, xmax=size[0], ymin=size[1], ymax=0)
    for obstacle in obstacles:
        [coord_x, coord_y, radius] = obstacle
        ax.add_patch(plt.Circle(xy=(coord_x, coord_y), radius=radius, fc='black', ec='black'))
    plt.savefig(r'./map.jpg', dpi=90)
    plt.clf()
    img = Image.open(r'./map.jpg').convert('1').convert('RGB')
    img = T.ToTensor()(img)
    img = img[:, 53:383, 73:517]
    img = T.Resize(resolution)(img)
    return T.ToPILImage()(img)


def collision_check_circle_edge(s, e, obs, clearance):
    if s[0] < 0 or s[1]>224:
        return True
    if e[0] < 0 or e[1]>224:
        return True
    s = torch.tensor([s[1], s[0]])
    e = torch.tensor([e[1], e[0]])
    dir = e - s
    dir = torch.tensor([dir[1], -dir[0]])/np.linalg.norm(dir)
    for ox, oy, size in obs:
        #  1 pixel or less than 1 pixel when the obstacles that the sizes satisfied the condition transfer into images
        # if size/224*50 < 0.2:
        #     continue
        o = torch.tensor([ox, oy])
        if distance.euclidean(e, o) < size + clearance/2 or distance.euclidean(e, o) < size + clearance/2:
            print("collision(vertex)", e, s, o, size)
            # img = plot_obstacles((224, 224), obs, (224, 224))
            # plt.imshow(img)
            # plt.plot(s[0], s[1], 'ro')
            # plt.plot(e[0], e[1], 'ro')
            # plt.plot(o[0], o[1], 'bo')
            # plt.show()
            return True
        dis = np.dot(dir, o - s)
        if dis>0:
            dir = -dir
        dis=abs(dis)
        projection = o + dis * dir
        dir1 = projection - s
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = projection - e
        dir2 = dir2 / np.linalg.norm(dir2)
        if dis < size + clearance/2 and np.dot(dir1, dir2) < 0:  # clearance requirement
            print("collision(edge)", dir, dis, dir1, dir2, o, size)
            # img = plot_obstacles((224, 224), obs, (224, 224))
            # plt.imshow(img)
            # plt.plot(s[0], s[1], 'ro')
            # plt.plot(e[0], e[1], 'ro')
            # plt.plot(projection[0], projection[1], 'go')
            # plt.plot(o[0], o[1], 'bo')
            # plt.show()
            return True
    return False


def plot_solution(size, obstacles, solution, index, result_root):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axis(xmin=0, xmax=size[0], ymin=0, ymax=size[1])
    for obstacle in obstacles:
        [coord_x, coord_y, radius] = obstacle
        ax.add_patch(plt.Circle(xy=(coord_x, coord_y), radius=radius, fc='black', ec='black'))
    plt.plot(solution.T[1], solution.T[0], 'b--')
    ax.plot(solution[0][1], solution[0][0], 'rs')
    ax.plot(solution[-1][1], solution[-1][0], 'rs')
    # plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度

    plt.savefig(r'{}/map-{}.jpg'.format(result_root, index), dpi=1000, bbox_inches='tight')
    # img = Image.open(r'./result/map-{}_{}.jpg'.format([x.item() for x in solution[0]], [x.item() for x in solution[-1]]))
    # img = T.ToTensor()(img)#T.CenterCrop((224, 224))(T.ToTensor()(img))
    # img = img[:, 53:383, 73:517]
    # T.ToPILImage()(img).save(r'./result/map-cropped-{}_{}.jpg'.format([x.item() for x in solution[0]], [x.item() for x in solution[-1]]))
    # plt.show()
    plt.clf()
    plt.close()


def extract_path_image(mask_root, origin_data_root, result_root, unsolved_problems_txt, clearance):
    unsolved_problems = []
    with open(unsolved_problems_txt, 'r', encoding='utf-8') as f:
        content = f.readlines()
    for line in content:
        unsolved_problems.append(json.loads(line))
    solved_problems_txt = result_root + "/solved_problems.txt"
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    supported = [".png", ".PNG"]
    masks = [os.path.join(mask_root, i) for i in os.listdir(mask_root) if os.path.splitext(i)[-1] in supported]
    masks.sort()
    failure_cases = []
    for m in masks:
        index = m.split('/')[-1].split('.')[0]
        problem = None
        for p in unsolved_problems:
            if p["Index"] == int(index):
                problem = p
                break
        if problem is None:
            print("No matched problem(Index:{})".format(index))
            continue
        print("mask index", index, m)
        folder_index = int(int(index) / NUM_PER_FOLDER)
        img_index = int(int(index) % NUM_PER_FOLDER)
        folder_path = os.path.join(origin_data_root, str(folder_index))
        images, labels, spaces_obs = read_folder(folder_path, is_read_path=True)
        if len(images) != NUM_PER_FOLDER or len(labels) != NUM_PER_FOLDER:
            continue
        init = labels[img_index][3][0]
        end = labels[img_index][3][10]
        m = Image.open(m)
        down_sample_rate = 2
        rst, path = extract_path(m, init_state=init, end_state=end, down_sample_rate=down_sample_rate)
        if not rst:
            print("Extract failed:", index)
            failure_cases.append(index)
            continue
        for i in range(len(path)-1):
            if collision_check_circle_edge(path[i], path[i+1], problem["Obstacles"], clearance):
                failure_cases.append(index)
                print("Collision:", index)
                break
        if index in failure_cases:
            continue
        print("Planning succeed:", index)
        length = sum([distance.euclidean(path[i], path[i+1]) for i in range(len(path)-1)])
        problem["Length"] = length
        problem["Waypoint"] = [list(p) for p in list(path.numpy())]
        with open(solved_problems_txt, "a") as f:
            f.write(json.dumps(problem) + "\n")
        print('cost:', length)
        plot_solution(size=(224, 224), obstacles=problem["Obstacles"], solution=torch.cat([p.unsqueeze(dim=1) for p in path], dim=1).T, index=index, result_root=result_root)
    print(len(failure_cases), failure_cases)


def common_result(folder_clearance_1, folder_clearance_3, result_root):
    if not os.path.exists(folder_clearance_1):
        os.mkdir(folder_clearance_1)
    if not os.path.exists(folder_clearance_3):
        os.mkdir(folder_clearance_3)
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    p_list1 = [int(i.split('.')[0].split('-')[-1]) for i in os.listdir(folder_clearance_1)
              if os.path.splitext(i)[-1] in supported]
    p_list2 = [int(i.split('.')[0].split('-')[-1]) for i in os.listdir(folder_clearance_3)
              if os.path.splitext(i)[-1] in supported]
    for p1 in p_list1:
        if p1 in p_list2:
            shutil.copy(folder_clearance_1 + '/map-{}.jpg'.format(p1),
                        '{}/result_{}_c1.jpg'.format(result_root, p1))
            shutil.copy(folder_clearance_3 + '/map-{}.jpg'.format(p1),
                        '{}/result_{}_c3.jpg'.format(result_root, p1))


if __name__ == '__main__':
    # data_root = '/home/long/source/PlanningNet/data_generation/data/exp_diff_algo/'
    data_root = '/home/long/dataset/exp_diff_data_generation/'
    # data_root = '/home/long/dataset/exp_diff_clearance/'
    # data_root = '/home/long/dataset/exp_diff_algo/'
    # data_root = '/home/long/dataset/exp_diff_data_generation'
    # data_root = '/home/long/dataset/exp_diff_data_generation/Data_InformedRRTstar_40*250/'
    # data_root = '/home/long/dataset/exp_diff_data_generation/Data_RRTstar_40*250/'
    # result_root = '/home/long/dataset/'
    result_root = data_root + '/result_c1'
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    # st = datetime.datetime.now()
    # process_map(data_root, mode='All', is_record_init_end=False)
    # se = datetime.datetime.now()
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), se-st)
    # move_data(data_root)
    # process_map(data_root, mode='Gen_path', is_record_init_end=False)
    # move_data(data_root)
    # index = 934510#int(abs(torch.randn(size=[1]))*1000000)
    # print(index)
    # img_path = '/home/long/dataset/planning224_3_seg/result_3classes_SETR_pmode/{}.png'.format(index)
    # mask_path = '/home/long/dataset/planning224_3_seg/Segmentation2Class/{}.png'.format(index)
    # image = Image.open(img_path)
    # mask = Image.open(mask_path)
    # mask = T.ToTensor()(mask)*255
    # print(mask)
    # plt.imshow(T.ToPILImage()(mask))
    # plt.show()
    # mask = torchvision.transforms.ToTensor()(mask).squeeze()
    # plot_3D_heatmap(mask)
    # generate_txt(data_root)
    # index = 65#int(np.random.random(1)*300000)
    # im = Image.open(data_root + '/map/{}.jpg'.format(index))
    # space1 = Image.open(result_root + '/result_c1(c1)/{}.png'.format(index))
    # space3 = Image.open(result_root + '/result_c3(c1)/{}.png'.format(index))
    # space = Image.open(data_root + '/mask_space/{}.png'.format(index))
    # path = Image.open(data_root + '/mask_path/{}.png'.format(index))
    # shutil.copy(data_root + '/map/{}.jpg'.format(index), '/home/long/map{}.jpg'.format(index))
    # shutil.copy(data_root + '/result_space/{}.png'.format(index), '/home/long/space{}.png'.format(index))
    # shutil.copy(data_root + '/mask_path/{}.png'.format(index), '/home/long/path{}.png'.format(index))
    # plt.subplot(1, 3, 1)
    # plt.imshow(im)
    # # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去 x 轴刻度
    # plt.yticks([])  # 去 y 轴刻度
    # plt.subplot(1, 3, 2)
    # plt.imshow(space1)
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去 x 轴刻度
    # plt.yticks([])  # 去 y 轴刻度
    # plt.subplot(1, 3, 3)
    # plt.imshow(space3)
    # plt.imshow(path, cmap='gray')
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去 x 轴刻度
    # plt.yticks([])  # 去 y 轴刻度
    # plt.savefig(r'./data_{}_result.jpg'.format(index), dpi=1000)
    # plt.show()
    # plt.clf()
    # img_path = '/home/long/dataset/planning224_3_seg/result'
    # image_des_path = '/home/long/dataset/planning224_3_seg/result_3classes_SETR_pmode'
    # images = [os.path.join(img_path, i) for i in os.listdir(img_path)]
    # for im in images:
    #     index = im.split('/')[-1].split('.')[0]
    #     dst = os.path.join(image_des_path, '{}.png'.format(int(index)))
    #     shutil.copy(im, dst)
    # im = os.listdir(data_root+'/map')
    # print(len(im))
    # im = os.listdir(data_root + '/mask_space')
    # print(len(im))
    # im = os.listdir(data_root + '/mask_path')
    # print(len(im))
    # mask_root = data_root + '/result_path'
    # extract_path_image(mask_root=mask_root,
    #                    origin_data_root=data_root+'/original_data',
    #                    result_root=result_root,
    #                    unsolved_problems_txt=data_root+'/unsolved_problems.txt',
    #                    clearance=1/50*224)
    root1 = data_root + '/result_c1'
    root2 = data_root + '/result_c3'
    result_path = data_root + "/result_common"
    common_result(folder_clearance_1=root1, folder_clearance_3=root2, result_root=result_path)

