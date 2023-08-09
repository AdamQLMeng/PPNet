import json
import os

import cv2
import imgviz
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from scipy.spatial import distance

from Path import plot_obstacles, Path
from process_map import add_init_end_single


def coord_rotation(x, radians):
    rotation = np.reshape([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]], [2, 2])
    return np.dot(rotation, x)


def load_data(data_path, planner='RRTstar', subset='train'):
    assert os.path.exists(data_path), "path '{}' does not exist.".format(data_path)
    assert (planner in {'BITstar', 'ABITstar', 'InformedRRTstar', 'RRTstar'})
    assert (subset in {'train', 'val', 'test'})

    envs = []
    paths = []
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    for i, line in enumerate(content):
        problem = json.loads(line)
        solutions = problem["Solution"]
        for s in solutions:
            if s["Planner"] == planner and s["Waypoint"] and s["Time"] < 59:
                paths.append(s["Waypoint"])
                envs.append([[o[:2], o[2]] for o in problem["Obstacles"]])

    if subset == 'train':
        envs = envs
        paths = paths
    if subset == 'val':
        envs = envs[5000:]
        paths = paths[5000:]
    if subset == 'test':
        envs = envs[5000:]
        paths = paths[5000:]

    print('{} set :'.format(subset), len(envs), 'its')
    assert (len(envs) == len(paths))

    return envs, paths


def generated_by_planners(data_path):
    clearance = 1/50*224
    step_img = 1/224
    points_per_seg = 100
    planners = ['BITstar', 'ABITstar', 'InformedRRTstar', 'RRTstar']
    colormap = imgviz.label_colormap()

    for planner in planners:
        generated_data_path = './data_{}'.format(planner)
        if not os.path.exists(generated_data_path):
            os.mkdir(generated_data_path)

        envs, paths = load_data(data_path, planner)

        for i, (e, p) in enumerate(zip(envs, paths)):
            # path planning problem image
            print('No.', i, planner)
            generated_map_path = generated_data_path + '/map'
            if not os.path.exists(generated_map_path):
                os.mkdir(generated_map_path)
            image = plot_obstacles((224, 224), e, resolution=(224, 224))
            image = add_init_end_single(image, [p[0][1], p[0][0]], [p[-1][1], p[-1][0]])
            torchvision.utils.save_image(image, './data_{}/map/{}.jpg'.format(planner, i))
            # plt.subplot(1, 3, 1)
            # plt.imshow(T.ToPILImage()(image))
            # plt.plot(np.array(p).T[0], np.array(p).T[1])
            # plt.plot(np.array(p).T[0], np.array(p).T[1], 'ro')
            # plt.show()
            # plt.clf()

            # path space mask
            mask = torch.zeros((224, 224))
            generated_mask_space_path = generated_data_path + '/mask_space'
            if not os.path.exists(generated_mask_space_path):
                os.mkdir(generated_mask_space_path)

            p = np.array(p)
            dir0 = p[0] - p[-1]
            dir0 = dir0 / np.linalg.norm(dir0)
            dir1 = p[-1] - p[0]
            dir1 = dir1 / np.linalg.norm(dir1)
            print("space initial end")
            for l in range(360):
                d0 = coord_rotation(dir0, l/180*np.pi)
                d1 = coord_rotation(dir1, l / 180 * np.pi)
                for k in range(round(clearance / 2 / step_img)):
                    point = p[0] + k * step_img * d0
                    x = round(point[0])
                    y = round(point[1])
                    if 0 < x < 223 and 0 < y < 223:
                        mask[x][y] = 1
                    point = p[-1] + k * step_img * d1
                    x = round(point[0])
                    y = round(point[1])
                    if 0 < x < 223 and 0 < y < 223:
                        mask[x][y] = 1
            print("space seg")
            waypoints = []
            for l in range(len(p)-1):
                dir = p[l+1] - p[l]
                dir = dir / np.linalg.norm(dir)
                step = distance.euclidean(p[l], p[l+1]) / points_per_seg
                waypoints_seg = []
                for j in range(points_per_seg):
                    waypoints_seg.append(p[l] + j * step * dir)
                dir = np.array([dir[1], -dir[0]])
                for j in range(points_per_seg):
                    for k in range(round(clearance/2/step_img)):
                        point = waypoints_seg[j] + k * step_img * dir
                        x = round(point[0])
                        y = round(point[1])
                        if 0 < x < 223 and 0 < y < 223:
                            mask[x][y] = 1
                        point = waypoints_seg[j] - k * step_img * dir
                        x = round(point[0])
                        y = round(point[1])
                        if 0 < x < 223 and 0 < y < 223:
                            mask[x][y] = 1
                waypoints += waypoints_seg
            mask = np.array(mask.cpu())
            _, space4seg = cv2.threshold(src=mask,  # 要二值化的图片
                                         thresh=0.5,  # 全局阈值
                                         maxval=1,  # 大于全局阈值后设定的值
                                         type=cv2.THRESH_BINARY)  # 设定的二值化类型，THRESH_BINARY：表示小于阈值置0，大于阈值置填充色
            mask = T.ToPILImage()(mask.T).convert(mode='P')
            mask.putpalette(colormap.flatten())
            mask.save('{}/{}.png'.format(generated_mask_space_path, i))
            # plt.subplot(1, 3, 2)
            # plt.imshow(mask)

            # path mask
            generated_mask_path_path = generated_data_path + '/mask_path'
            if not os.path.exists(generated_mask_path_path):
                os.mkdir(generated_mask_path_path)
            mask = np.zeros((224, 224), dtype=np.float32)
            # mask = np.array(mask)
            for point in waypoints:
                x = round(point[0])
                y = round(point[1])
                if 0 < x < 223 and 0 < y < 223:
                    mask[x][y] = 255
            mask = T.ToPILImage()(mask.T).convert(mode='L')
            mask.save('{}/{}.png'.format(generated_mask_path_path, i))
            # plt.subplot(1, 3, 3)
            # plt.imshow(mask)
            # plt.show()


if __name__ == '__main__':
    data_path = "./solved_problems_comparison.txt"
    generated_by_planners(data_path)
