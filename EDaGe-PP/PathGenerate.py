# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
import torchvision.transforms as tvtrans
from Path import Path

ORDER = 4
PATHSEGNUM = 10
DIM = 2


class PathGroup:
    def __init__(self, path_num=5, resolution=224, map_size=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Paths = []
        self.TargetPaths = []
        self.PathNum = path_num
        self.Resolution = resolution
        self.MapSize = map_size
        self.MapOffset = self.Resolution / 2
        self.SpacesObs = []
        self.SpacesFree = []
        self.Rotation = []
        self.Translation = []

    def generate(self, path_seg_num=3, poly_order=4, dim=2, clearance=1):
        i = 0
        while i < self.PathNum:
            is_straight = False if np.random.random(1) > 0.01 else True
            path = Path(seg_num=path_seg_num, poly_order=poly_order, dim=dim, clearance=clearance, is_straight=is_straight)
            path.generate(show_now=False)
            path.draw_boundary(show_now=False)
            rst = path.path_obstacles(resolution=self.Resolution, map_size=self.MapSize, map_offset=self.MapOffset)
            print('path generate result:', rst, "Length:", path.Length)
            if rst:
                i = i + 1
                self.Paths.append(path)
                self.TargetPaths.append(path)
        if not np.size(self.TargetPaths):
            return False
        else:
            print('target path list length:', len(self.TargetPaths))
            return True

    def plot(self, path, show_now=True):
        color_list = ['blue', 'green', 'red', 'black', 'yellow', 'pink',
                      'orange', 'purple', 'gray', 'gold', 'navy', 'tan', 'yellowgreen']
        for i in range(np.size(path.PathSeg)):
            x = np.arange(0, 1000) / 1000 * path.PathSeg[i].EndPoint
            y = np.polyval(path.PathSeg[i].Poly, x)
            [x, y] = path.point_transform([x, y], i)
            # [x, y] = path.coordi_rotation([x, y], rotation/180*np.pi) + np.tile(translation, [np.shape([x, y])[1], 1]).T
            plt.plot(x, y, color=color_list[i])
        if show_now:
            plt.show()


if __name__ == '__main__':
    # paths = PathGroup(pathnum=1)
    # print('Generate', paths.generate(pathsegnum=PATHSEGNUM, polyorder=ORDER, dim=2, robotsize=1))
    # torch.save(paths, './data/paths')
    paths = torch.load('./data/paths')
    paths.superimpose()
    # torch.save(paths, './data/paths_imposed')
    # paths = torch.load('./data/paths_imposed')
    # vertices = paths.graph_generate()
    # print(vertices)
    # vertices = np.reshape(vertices, [-1, 2])
    # print(vertices)
    # torchvision.utils.save_image(paths.SpacesObs[0], 'pic.jpg')
    # torchvision.utils.save_image(paths.TargetPaths[0].ConvexSpaceObs, 'ConvexSpaceObs.jpg')
    # torchvision.utils.save_image(paths.Paths[0].Space, 'impose.jpg')
    # plt.plot(vertices.T[0], vertices.T[1], 'ro')
    # plt.show()
    # paths.plot(paths.TargetPaths[0])
