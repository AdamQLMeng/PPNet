import json
import os
import argparse
import datetime

import cv2
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import distance

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils

from GMM import GMM
from Path import plot_obstacles
from PathGenerate import PathGroup
from process_map import add_init_end_single

ORDER = 4
PATHSEGNUM = 10
DIM = 2
total_record = 10000
cnt = 0

class MapGenerate:
    def __init__(self, path_num=5, resolution=224, map_size=50, obstacles_size=5, obstacles_num=50, clearance=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Resolution = resolution
        self.MapSize = map_size
        self.ObstaclesNum = obstacles_num
        self.ObstacleSize = obstacles_size
        self.Clearance = clearance
        self.MapData = []
        self.PathGroup = PathGroup(path_num=path_num, resolution=resolution, map_size=map_size)
        self.PathGroup.generate(path_seg_num=PATHSEGNUM, poly_order=ORDER, dim=2, clearance=clearance)
        self.MapLabel = []

    def generate(self, map_num=100, folder_path='./', round_index=0):
        print('generating {} maps by using {} target paths'.format(map_num, len(self.PathGroup.TargetPaths)))
        for i in range(int(np.round(map_num/len(self.PathGroup.TargetPaths)**2))):
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            if not os.path.exists(os.path.join(folder_path, 'GMM')):
                os.mkdir(os.path.join(folder_path, 'GMM'))
                os.mkdir(os.path.join(folder_path, 'data'))
            for j in range(np.size(self.PathGroup.TargetPaths)):
                label = []
                TargetPath = self.PathGroup.TargetPaths[j]
                for pathseg in self.PathGroup.TargetPaths[j].PathSeg:
                    label.append([pathseg.Poly, pathseg.EndPoint])
                # length = sum([distance.euclidean(TargetPath.PathPoint[i], TargetPath.PathPoint[i+1]) for i in range(len(TargetPath.PathPoint)-1)])
                k = 0
                torchvision.utils.save_image(TargetPath.Space, r'{}/data/{}.jpg'.format(folder_path, j))

                repeat_times = 0
                while k < np.size(self.PathGroup.Paths):
                    repeat_times += 1
                    if repeat_times > 1000000:
                        print('Error:Repeated over 1000000 times! path:', folder_path)
                        break
                    angle = np.random.random([1]) * 360 - 180
                    translation = np.array(np.random.random([2]) * self.Resolution - self.Resolution / 2, dtype=int)
                    translation = [translation[0], translation[1]]
                    rst, convexhull = TargetPath.boundary_check(-angle, [translation[1], translation[0]])
                    if rst:
                        index = i*len(self.PathGroup.TargetPaths)**2 + j * np.size(self.PathGroup.TargetPaths) + k
                        path_obstacles = []
                        segpoint = TargetPath.SegPointImage \
                                   - np.tile(np.array([self.Resolution/2, self.Resolution/2]), [np.shape(TargetPath.SegPointImage)[0], 1])
                        segpoint = TargetPath.coord_rotation(segpoint.T, -angle/180*np.pi).T \
                                   + np.tile(np.array([self.Resolution/2, self.Resolution/2]), [np.shape(TargetPath.SegPointImage)[0], 1])
                        segpoint = segpoint + np.tile([translation[1], translation[0]], [np.shape(TargetPath.SegPointImage)[0], 1])

                        pathpoint = TargetPath.PathPoint \
                                   - np.tile(np.array([self.Resolution/2, self.Resolution/2]), [np.shape(TargetPath.PathPoint)[0], 1])
                        pathpoint = TargetPath.coord_rotation(pathpoint.T, -angle/180*np.pi).T \
                                   + np.tile(np.array([self.Resolution/2, self.Resolution/2]), [np.shape(pathpoint)[0], 1])
                        pathpoint = pathpoint + np.tile([translation[1], translation[0]], [np.shape(pathpoint)[0], 1])

                        # prepare necessary obstacles' settings
                        for obs in TargetPath.obstacles:
                            coord = np.array([obs[1], obs[0]])
                            coord = coord - np.array([self.Resolution / 2, self.Resolution / 2])
                            coord = TargetPath.coord_rotation(coord, -angle / 180 * np.pi).T \
                                        + np.array([self.Resolution / 2, self.Resolution / 2])
                            coord = coord + np.array([translation[1], translation[0]])
                            path_obstacles.append([list(coord)[1], list(coord)[0], float(np.array(obs[-1]))])
                        self.MapLabel.append([label, angle, translation, segpoint, pathpoint])

                        init = segpoint[0]
                        end = segpoint[10]
                        # prepare necessary obstacles' settings
                        # path_obs = TargetPath.PathObs
                        # rotation = T.RandomRotation(degrees=(-angle, -angle), fill=[255, 255, 255])
                        # path_obs = rotation(path_obs.to(self.device))
                        # path_obs = T.functional.affine(path_obs,
                        #                                  translate=translation, angle=0, scale=1,
                        #                                  shear=0, fill=[255, 255, 255])
                        # prepare space of path constrained by clearance
                        path_space = TargetPath.Space
                        rotation = T.RandomRotation(degrees=(-angle, -angle))
                        path_space = rotation(path_space.to(self.device))
                        path_space = T.functional.affine(path_space, translate=translation, angle=0, scale=1,
                                                          shear=0)
                        # generate map that target path is optimal
                        map_img = self.generate_map_randomly(path_point=pathpoint, init=init, end=end, length=TargetPath.Length, path_obstacles=path_obstacles, index=index+round_index*100)

                        # map_img = torch.mul(map_img, path_obs)
                        map_img = map_img + path_space

                        map_img = add_init_end_single(map_img, init, end)
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(T.ToPILImage()(map_img))
                        # plt.plot(pathpoint.T[1], pathpoint.T[0])
                        # plt.plot(np.array(path_obstacles).T[0], np.array(path_obstacles).T[1], 'ro')
                        # plt.subplot(1, 2, 2)
                        # # plt.imshow(T.ToPILImage()(TargetPath.PathObs))
                        # plt.plot(TargetPath.PathPoint.T[1], TargetPath.PathPoint.T[0])
                        # plt.show()
                        torchvision.utils.save_image(map_img,
                                                     r'{}/{}.jpg'.format(folder_path, index))
                        k = k + 1

    def generate_map_randomly(self, path_point, init, end, length, path_obstacles, index):
        unsolved_problems_txt = "./unsolved_problems.txt"
        obs = np.concatenate([(np.random.random(self.ObstaclesNum) * self.MapSize).reshape([-1, 1]),
                              (np.random.random(self.ObstaclesNum) * self.MapSize).reshape([-1, 1]),
                              (np.random.random(self.ObstaclesNum) * self.ObstacleSize).reshape([-1, 1])], axis=1)
        obstacles = []
        for item in obs:
            coord = item[:2]
            coord_img = coord / self.MapSize * self.Resolution
            radius = item[-1].item()
            radius_img = radius / self.MapSize * self.Resolution
            dis = []
            for i, p in enumerate(path_point):
                if i % 2:
                    dis.append(distance.euclidean(p, coord_img))

            if min(dis) > radius_img + self.Clearance / self.MapSize * self.Resolution:
                obstacles.append([coord_img[1], coord_img[0], radius_img])
        global cnt
        if cnt < total_record:
            problem = {"Index": index, "Init": list(init), "End": list(end), "Length": length, "Obstacles": obstacles+path_obstacles}
            with open(unsolved_problems_txt, "a") as f:
                f.write(json.dumps(problem) + "\n")
            cnt += 1

        return plot_obstacles((self.Resolution, self.Resolution), obstacles+path_obstacles, resolution=(self.Resolution, self.Resolution)).to(self.device)


if __name__ == '__main__':
    root = r'./data/exp_diff_clearance'
    obstacles_num = 20
    clearance = 3
    path_num = 10
    map_num = 10 * path_num
    round_num = 100
    start_index = 0
    if not os.path.exists(root):
        os.mkdir(root)
    for i in range(round_num-start_index):
        index = i + start_index
        folder_path = os.path.join(root, '{}'.format(index))
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        print('round:', index)
        map_class = MapGenerate(path_num=path_num, resolution=224, map_size=50, obstacles_num=obstacles_num, clearance=clearance)
        # torch.save(map, './data/map')
        # map = torch.load('./data/map')
        map_class.generate(map_num=map_num, folder_path=folder_path, round_index=index)
        # torch.save(map.PathGroup.TargetPaths.ConvexSpaceFree, '{}/TargetPaths'.format(folder_path+'/data'))
        torch.save(map_class.MapLabel, '{}/MapLabel'.format(folder_path+'/data'))
        map_class = None
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
