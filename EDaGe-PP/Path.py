import argparse

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import torch
import torchvision
import torchvision.transforms as T
from scipy.spatial import ConvexHull, distance
from PIL import Image
import datetime

from PathSeg import PathSeg

DIM = 2
BOUDARY_WIDTH_4SEG = 3
CLASS_INDEX_4SEG = 1


class BoundaryOneSide:
    def __init__(self):
        self.point = []
        self.direction = []


class Boundary:
    def __init__(self):
        self.upboundary = BoundaryOneSide()
        self.downboundary = BoundaryOneSide()
        self.initboundary = []
        self.endboundary = []


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
    return img


class Path:
    def __init__(self, seg_num=3, poly_order=3, dim=2, clearance=1, is_straight=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PathSeg = []
        self.SegPoint = [[0, 0]]
        self.PathPoint = []
        self.SegPointImage = []
        self.obstacles = []
        self.SegNum = seg_num
        self.PolyOrder = poly_order
        self.Dim = dim
        self.Boundary = Boundary()
        self.BoundaryPoint = []
        self.Clearance = clearance
        self.EndPoint = np.array([0, 0])
        self.Translation = np.array([0, 0])
        self.Rotation = 0
        self.Space = torch.zeros([1])
        self.PathObs = torch.zeros([1])
        self.Resolution = 0
        self.MapSize = 0
        self.MapOffset = 0
        self.ConvexHull = []
        self.is_straight = is_straight
        self.Length = 0

    def generate(self, show_now=True, polys=None):
        for i in range(self.SegNum):
            self.PathSeg.append(PathSeg(self.PolyOrder, self.Dim, is_straight=self.is_straight))
            if polys is None:
                self.PathSeg[i].random()
            else:
                self.PathSeg[i].random(polys[i][0:self.PolyOrder+1], polys[i][self.PolyOrder+1])
        self.transform()
        for i in range(self.SegNum):
            point = np.array([self.PathSeg[i].EndPoint, np.polyval(self.PathSeg[i].Poly, self.PathSeg[i].EndPoint)])
            point = self.point_transform(point, i)
            point = np.reshape(point, [2])
            self.SegPoint.append(point)
        self.SegPoint = np.reshape(self.SegPoint, [-1, 2])
        self.PathPoint = self.plot(show=show_now, show_now=show_now)
        length = [distance.euclidean(self.PathPoint[i], self.PathPoint[i+1]) for i in range(len(self.PathPoint)-1)]
        self.Length = sum(length)
        self.EndPoint = self.SegPoint[np.shape(self.SegPoint)[0]-1]
        if show_now:
            plt.plot(0, 0, 'ro')
            plt.plot(self.SegPoint.T[0], self.SegPoint.T[1], 'go')

    def boundary_check(self, angle, translation):
        convexhull = self.coord_rotation(
            self.ConvexHull.T - np.tile([self.MapOffset, self.MapOffset],
                                                   [np.shape(self.ConvexHull)[0], 1]).T,
            angle / 180 * np.pi)
        convexhull = convexhull.T \
                     + np.tile(translation, [np.shape(convexhull.T)[0], 1]) \
                     + np.tile([self.MapOffset, self.MapOffset], [np.shape(self.ConvexHull)[0], 1])
        for hullpoint in convexhull:
            if hullpoint[0] < 0 or hullpoint[0] >= self.Resolution or hullpoint[1] < 0 or hullpoint[1] >= self.Resolution:
                return False, convexhull
        return True, convexhull

    def path_space(self, resolution=224, map_size=50, map_offset=112):
        self.Resolution = resolution
        self.MapSize = map_size
        self.MapOffset = map_offset
        space = torch.zeros([self.Resolution*2, self.Resolution*2], device='cpu')
        step_len = 1 / self.Resolution * map_size
        dis = 0.8 * self.Clearance / step_len
        for i in range(np.shape(self.Boundary.initboundary)[0]):
            dir = - step_len * self.Boundary.initboundary[i] / np.linalg.norm(self.Boundary.initboundary[i])
            space = self.free_space_bydirection(space, self.Boundary.initboundary[i], dir, dis, mapoffset=self.Resolution)
        for i in range(np.shape(self.Boundary.endboundary)[0]):
            dir = step_len * (np.reshape(self.EndPoint, [2]) - self.Boundary.endboundary[i]) / np.linalg.norm(
                np.reshape(self.EndPoint, [2]) - self.Boundary.endboundary[i])
            space = self.free_space_bydirection(space, self.Boundary.endboundary[i], dir, dis, mapoffset=self.Resolution)
        for i in range(np.size(self.PathSeg)):
            for j in range(np.shape(self.Boundary.upboundary.point)[1]):
                dir = step_len * self.Boundary.upboundary.direction[i][j]
                space = self.free_space_bydirection(space, self.Boundary.upboundary.point[i][j], dir, dis, mapoffset=self.Resolution)
        for i in range(np.size(self.PathSeg)):
            for j in range(np.shape(self.Boundary.downboundary.point)[1]):
                dir = step_len * self.Boundary.downboundary.direction[i][j]
                space = self.free_space_bydirection(space, self.Boundary.downboundary.point[i][j], dir, dis, mapoffset=self.Resolution)
        self.ConvexHull, _ = self.convexhull()
        space_img = Image.fromarray(space.cpu().numpy().astype('uint8')).convert('RGB')
        space_raw = T.ToTensor()(space_img).to(self.device)
        rst, self.Space = self.space_normalization(space_raw, point_trans=True, check_free=False)
        if not rst:
            return False, self.Space
        else:
            return True, self.Space

    def path_obstacles(self, resolution=224, map_size=50, map_offset=112):
        rst, space = self.path_space(resolution, map_size, map_offset)
        if not rst:
            return False
        if self.is_straight:
            self.PathObs = torch.ones([3, resolution, resolution])
        else:
            isle = self.search_isle()
            self.obstacles = self.set_obstacles(isle)
            # self.PathObs = plot_obstacles(size=(resolution, resolution), obstacles=self.obstacles, resolution=(resolution, resolution))
        # print(self.PathObs.shape)
        return True

    def space_normalization(self, space, point_trans=False, check_free=False):
        angle = -135
        self.Rotation = math.atan(self.EndPoint[1] / self.EndPoint[0]) / np.pi * 180 + angle
        rotation = T.RandomRotation(degrees=(-self.Rotation, -self.Rotation))
        space_temp = rotation(space)
        self.ConvexHull = self.ConvexHull - np.tile([self.Resolution, self.Resolution], [np.shape(self.ConvexHull)[0], 1])
        self.ConvexHull = self.coord_rotation(self.ConvexHull.T, - self.Rotation / 180 * np.pi).T
        self.ConvexHull = self.ConvexHull + np.tile([self.Resolution, self.Resolution], [np.shape(self.ConvexHull)[0], 1])

        if not check_free:
            hull_center = torch.mean(torch.tensor(self.ConvexHull), dim=0)
            translation = torch.tensor([self.Resolution / 2, self.Resolution / 2]) - hull_center
        else:
            translation = torch.tensor([self.Resolution / 2, self.Resolution / 2]) - self.EndPoint/2
        translation = [translation[1], translation[0]]
        self.Translation = translation

        if check_free or self.boundary_check(0, translation):
            space_temp = T.functional.affine(space_temp, translate=self.Translation, angle=0, scale=1, shear=0)
            self.ConvexHull = self.ConvexHull + np.tile([translation[1], translation[0]], [np.shape(self.ConvexHull)[0], 1])
            self.ConvexHull = torch.tensor(self.ConvexHull)
            space_temp = space_temp[:, 0:self.Resolution, 0:self.Resolution]
            if point_trans and not check_free:
                self.SegPointImage = self.coord_rotation(self.SegPoint.T, - self.Rotation / 180 * np.pi).T
                self.SegPointImage = self.coord_euclidean2image(self.SegPointImage, mapoffset=np.shape(space)[1] / 2)
                self.SegPointImage = self.SegPointImage + np.tile([translation[1], translation[0]], [np.shape(self.SegPointImage)[0], 1])
                self.PathPoint = self.coord_rotation(self.PathPoint.T, - self.Rotation / 180 * np.pi).T
                self.PathPoint = self.coord_euclidean2image(self.PathPoint, mapoffset=np.shape(space)[1] / 2)
                self.PathPoint = self.PathPoint + np.tile([translation[1], translation[0]], [np.shape(self.PathPoint)[0], 1])
                self.BoundaryPoint = self.coord_rotation(self.BoundaryPoint.T, - self.Rotation / 180 * np.pi).T
                self.BoundaryPoint = self.coord_euclidean2image(self.BoundaryPoint, mapoffset=np.shape(space)[1] / 2)
                self.BoundaryPoint = self.BoundaryPoint + np.tile([translation[1], translation[0]], [np.shape(self.BoundaryPoint)[0], 1])
            return True, space_temp
        else:
            print('boundary_check False')
            space_temp = space_temp[:, 0:self.Resolution, 0:self.Resolution]
            return False, space_temp

    def point_classification_oneside(self, point, boundaryoneside):
        for i in range(np.size(self.PathSeg)):
            for j in range(np.shape(boundaryoneside.point)[1]-1):
                if distance.euclidean(point, boundaryoneside.point[i][j]) <= 0.6 * self.Clearance \
                        and distance.euclidean(point, boundaryoneside.point[i][j + 1]) <= 0.6 * self.Clearance:
                    gradient = (boundaryoneside.point[i][j + 1][1] - boundaryoneside.point[i][j][1]) / (boundaryoneside.point[i][j + 1][0] - boundaryoneside.point[i][j][0])
                    dir_norm = np.array([1, -1 / gradient])
                    dir_point = np.array(point - boundaryoneside.point[i][j])
                    if np.dot(dir_norm, dir_point) < 0:
                        dir_point = (-1) * dir_norm
                    else:
                        dir_point = dir_norm
                    if np.dot(dir_point, boundaryoneside.direction[i][j]) > 0 \
                            and np.dot(dir_point, boundaryoneside.direction[i][j + 1]) > 0:
                        return True
        return False

    def point_classification(self, point):
        point = np.reshape(point, [2])
        if distance.euclidean(point, [0, 0]) <= 0.5 * self.Clearance \
                or distance.euclidean(point, np.reshape(self.EndPoint, [2])) <= 0.5 * self.Clearance:
            return True
        else:
            if self.point_classification_oneside(point, self.Boundary.upboundary) \
                    or self.point_classification_oneside(point, self.Boundary.downboundary):
                return True
            else:
                return False

    def point_transform(self, point, segindex):
        # rotation
        if segindex != 0:
            point = self.coord_rotation(point, self.PathSeg[segindex].Rotation)
        # translation
        if np.size(point) > 2:
            point = point + np.tile(np.reshape(self.PathSeg[segindex].Translation, [2]), [int(np.size(point)/2), 1]).T
        else:
            point = np.reshape(point, [2]) + np.reshape(self.PathSeg[segindex].Translation, [2])
        return point

    def point_calcu(self, x):
        x_temp = x
        index = 0
        for i in range(np.size(self.PathSeg)):
            x_end = self.PathSeg[i].EndPoint
            if x - x_end > 0:
                x = x - x_end
                if i == (np.size(self.PathSeg) - 1):
                    print('The given input is out of the range! Input value:', x_temp)
                    return False
            else:
                index = i
                break
        y = np.polyval(self.PathSeg[index].Poly, x)
        [x, y] = self.point_transform([x, y], index)
        plt.plot(x, y, 'ro')
        return [x, y], index

    def plot(self, show=True, show_now=True):
        point_out = []
        color_list = ['blue', 'green', 'red', 'black', 'yellow', 'pink', 'orange', 'purple', 'navy', 'brown', 'gold', 'lightblue']
        for i in range(np.size(self.PathSeg)):
            x = np.arange(0, 100) / 100 * self.PathSeg[i].EndPoint
            y = np.polyval(self.PathSeg[i].Poly, x)
            point = self.point_transform([x, y], i)
            point_out.append(point.T)
            if show:
                plt.plot(point[0], point[1], color=color_list[i], label='Curve {}'.format(i))
        if show and show_now:
            plt.xticks([])  # 去 x 轴刻度
            plt.yticks([])  # 去 y 轴刻度
            plt.legend()
            plt.savefig(r'./path_concatenate.jpg', dpi=1000)
            plt.show()
        return np.reshape(point_out, [-1, 2])

    @staticmethod
    def coord_rotation(x, radians):
        rotation = np.reshape([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]], [2, 2])
        return np.dot(rotation, x)

    def angle(self, index, index_prev):
        angle = math.atan(self.PathSeg[index].GradSt)
        angle_prev = math.atan(self.PathSeg[index_prev].GradEnd)
        angle = angle_prev - angle
        return angle

    def angle_abs(self, index):
        if index == 0:
            return False
        else:
            angle = 0
            for i in range(index):
                angle = angle + self.angle(i + 1, i)
        return angle

    def translation_seg(self, index):
        t = np.zeros([DIM])
        for i in range(index):
            if i != 0:
                angle = self.angle_abs(i)
                t = t + self.coord_rotation(np.reshape(self.PathSeg[i].Translation, [2]), angle)
            else:
                t = t + np.reshape(self.PathSeg[i].Translation, [2])
        return t

    def translation(self):
        trans_temp = []
        for i in range(np.size(self.PathSeg)):
            trans_temp.append(self.translation_seg(i))
        for i in range(np.size(self.PathSeg)):
            self.PathSeg[i].Translation = np.reshape(trans_temp[i], [2])
        return True

    def rotation(self):
        for i in range(np.size(self.PathSeg)):
            self.PathSeg[i].Rotation = self.angle_abs(i)
        return True

    def transform(self):
        self.translation()
        self.rotation()

    def draw_boundary(self, show_now=True):
        for i in range(np.size(self.PathSeg)):
            x = np.arange(0, 50) / 50 * self.PathSeg[i].EndPoint
            y = np.polyval(self.PathSeg[i].Poly, x)
            pathder = np.polyder(self.PathSeg[i].Poly)
            y_der = np.polyval(pathder, x)
            point = self.point_transform([x, y], i).T
            norm = np.transpose(
                self.coord_rotation(
                    [y_der, - np.ones([np.size(y_der)])], self.PathSeg[i].Rotation))
            for j in range(np.size(x)):
                norm[j] = norm[j] / np.linalg.norm(norm[j])
            self.Boundary.upboundary.point.append(point - 0.5 * self.Clearance * norm)
            self.Boundary.upboundary.direction.append(norm)
            self.Boundary.downboundary.point.append(point + 0.5 * self.Clearance * norm)
            self.Boundary.downboundary.direction.append(-1 * norm)
        for i in range(50):
            self.Boundary.initboundary.append(
                self.coord_rotation(self.Boundary.upboundary.point[0][0], np.pi/50*(i+1)))
        for i in range(50):
            self.Boundary.endboundary.append(
                self.coord_rotation(
                    np.reshape(self.Boundary.upboundary.point[np.size(self.PathSeg)-1][np.shape(self.Boundary.upboundary.point)[1]-1], [2])
                    - np.reshape(self.EndPoint, [2]),
                    -np.pi/50*(i+1))
                + np.reshape(self.EndPoint, [2]))

        # 颠倒init和下边界点顺序，使所有边界点能以某一顺序首位连接
        up = np.reshape(np.array(self.Boundary.upboundary.point), [-1, 2])
        down = np.reshape(np.array(self.Boundary.downboundary.point), [-1, 2])
        init = [self.Boundary.initboundary[len(self.Boundary.initboundary) - i - 1] for i in
                                      range(len(self.Boundary.initboundary))]
        down = [down[len(down) - i - 1] for i in range(len(down))]

        # 所有边界点能以某一顺序首位连接
        self.BoundaryPoint = np.array(init)
        self.BoundaryPoint = np.concatenate((self.BoundaryPoint, np.array(up)), axis=0)
        self.BoundaryPoint = np.concatenate((self.BoundaryPoint, np.array(self.Boundary.endboundary)), axis=0)
        self.BoundaryPoint = np.concatenate((self.BoundaryPoint, np.array(down)), axis=0)

        if show_now:
            plt.plot(np.transpose(self.Boundary.initboundary)[0],
                     np.transpose(self.Boundary.initboundary)[1], 'r')
            plt.plot(np.transpose(self.Boundary.endboundary)[0],
                     np.transpose(self.Boundary.endboundary)[1], 'r')
            plt.plot(np.transpose(self.Boundary.upboundary.point)[0],
                     np.transpose(self.Boundary.upboundary.point)[1], 'r')
            plt.plot(np.transpose(self.Boundary.downboundary.point)[0],
                     np.transpose(self.Boundary.downboundary.point)[1], 'r')

    def coord_image2euclidean(self, x, mapoffset):
        x = np.reshape(x, [-1, 2])
        step_len = self.MapSize / self.Resolution
        coord, coord_x, coord_y = [], 0, 0
        for point in x:
            coord_x = (point[0] - mapoffset) * step_len
            coord_y = (point[1] - mapoffset) * step_len
            coord.append([coord_x, coord_y])
        return np.array(coord)

    def coord_euclidean2image(self, x, mapoffset):
        x = np.reshape(x, [-1, 2])
        step_len = self.MapSize / self.Resolution
        coord, coord_x, coord_y = [], 0, 0
        for point in x:
            coord_x = int(np.round(point[0] / step_len + mapoffset))
            coord_y = int(np.round(point[1] / step_len + mapoffset))
            coord.append([coord_x, coord_y])
        return np.array(coord)

    def convexhull(self):
        # 以边界点作为求凸包的点
        point_list = self.coord_euclidean2image(self.PathPoint, mapoffset=self.Resolution)
        # 求凸包点集合和凸包中心点
        hull = ConvexHull(point_list)
        hull_point = torch.tensor(point_list[hull.vertices]).float()
        hull_center = torch.mean(hull_point, dim=0)
        return hull_point, hull_center

    def free_space_bydirection(self, space, x_init, dir, step_num, mapoffset, value=255):
        for i in range(int(np.round(step_num))):
            index = self.coord_euclidean2image(x_init + i * dir, mapoffset).T
            if 0 < index[0] < np.shape(space)[0] and 0 < index[1] < np.shape(space)[1]:
                space[index[0].item(), index[1].item()] = value
            else:
                return space
        return space

    def free_space_byline(self, space, poly, dir, width_coef=0.3):
        x_cross = - poly[0] / poly[1]
        x_boundary = int(np.round((self.Resolution - poly[0]) / poly[1]))
        step_len = 1 / self.Resolution * self.MapSize
        half_robot_size_step = int(np.round(self.Clearance/step_len*width_coef))  # 用作缓冲，去除计算误差导致的多余障碍

        if dir[0] > 0 and dir[1] > 0:
            x_range = x_boundary
            if x_range < 0:
                x_range = 0
            for i in range(self.Resolution - x_range):
                if int(np.round(np.polyval(poly, x_range + i))) < half_robot_size_step:
                    y_range = 0
                else:
                    y_range = int(np.round(np.polyval(poly, x_range + i))) - half_robot_size_step
                for j in range(self.Resolution - y_range):
                    space[x_range + i, y_range + j] = 255

        if dir[0] < 0 and dir[1] > 0:
            x_range = x_boundary
            if x_range >= self.Resolution:
                x_range = self.Resolution
            for i in range(x_range):
                if int(np.round(np.polyval(poly, i))) < half_robot_size_step:
                    y_range = 0
                else:
                    y_range = int(np.round(np.polyval(poly, i))) - half_robot_size_step
                for j in range(self.Resolution - y_range):
                    space[i, y_range + j] = 255

        if dir[0] < 0 and dir[1] < 0:
            if int(np.round(x_cross)) >= self.Resolution:
                x_range = self.Resolution
            else:
                x_range = int(np.round(x_cross))
            for i in range(x_range):
                if int(np.round(np.polyval(poly, i))) >= self.Resolution - half_robot_size_step:
                    y_range = self.Resolution
                else:
                    y_range = int(np.round(np.polyval(poly, i))) + half_robot_size_step
                for j in range(y_range):
                    space[i, j] = 255

        if dir[0] > 0 and dir[1] < 0:
            if int(np.round(x_cross)) < 0:
                x_range = 0
            else:
                x_range = int(np.round(x_cross))
            for i in range(self.Resolution - x_range):
                if int(np.round(np.polyval(poly, x_range + i))) >= self.Resolution - half_robot_size_step:
                    y_range = self.Resolution
                else:
                    y_range = int(np.round(np.polyval(poly, x_range + i))) + half_robot_size_step
                for j in range(y_range):
                    space[x_range + i, j] = 255
        return space

    def set_obstacles(self, boundarys):
        obstacles = []
        size_clearance = self.Clearance / self.MapSize * self.Resolution * 1.1
        for isle in boundarys:
            center = (isle[0] + isle[-1]) / 2
            dir_tangent = (isle[0] - isle[-1]) / np.linalg.norm(isle[0] - isle[-1])
            dir_normal = np.array([dir_tangent[1], -dir_tangent[0]])
            dir_normal = dir_normal if np.dot(isle[int(len(isle) / 2)] - center, dir_normal) < 0 else -dir_normal
            dis = []
            for p in isle:
                dis.append(abs(np.dot(p - isle[0], dir_normal)))
            size_max = max(dis) * 2
            peak = isle[dis.index(max(dis))]
            obs_size = []
            size_pre = 0
            while sum(obs_size) < size_max:
                radius = torch.rand(1) * size_max / 2
                random_normal = torch.rand(1) if len(obs_size) else 1
                motion = random_normal * (radius + size_pre + (size_clearance if len(obs_size) == 0 else 0))
                motion = max(motion, radius - sum(obs_size))
                coord = (peak if not len(obs_size) else coord) \
                        + motion * dir_normal \
                        + ((torch.rand(1)-0.5)/0.5 * radius/2 * dir_tangent if len(obs_size) else 0)
                dis = []
                for i, p in enumerate(self.PathPoint):
                    if i % 2:
                        dis.append(distance.euclidean(p, coord))
                if min(dis) < radius + self.Clearance / self.MapSize * self.Resolution:
                    radius = min(dis) - self.Clearance / self.MapSize * self.Resolution
                if radius > 0:
                    obs_size.append(motion)
                    size_pre = radius
                    obstacles.append([coord[1], coord[0], radius])
            #     plt.plot(coord[1], coord[0], 'go')
            # plt.plot(self.PathPoint.T[1], self.PathPoint.T[0])
            # plt.plot(peak[1], peak[0], 'ro')
            # plt.show()
        return obstacles

    def search_isle(self, width_coef=0.2):
        isle = []
        step_len = 1 / self.Resolution * self.MapSize
        self.PathPoint = torch.tensor(self.PathPoint)

        for i in range(np.shape(self.ConvexHull)[0]):
            if i == np.shape(self.ConvexHull)[0] - 1:
                index = 0
            else:
                index = i + 1
            if np.linalg.norm(self.ConvexHull[index] - self.ConvexHull[i]) > 5/step_len:
                for j in range(len(self.PathPoint)):
                    dis_init = np.linalg.norm(self.PathPoint[j] - self.ConvexHull[i])
                    dis_end = np.linalg.norm(self.PathPoint[j] - self.ConvexHull[index])
                    if not j:
                        dis_init_min = dis_init
                        isle_init_index = j
                        dis_end_min = dis_end
                        isle_end_index = j
                    else:
                        if dis_init < dis_init_min:
                            dis_init_min = dis_init
                            isle_init_index = j
                        if dis_end < dis_end_min:
                            dis_end_min = dis_end
                            isle_end_index = j
                boundary = self.PathPoint[min(isle_init_index, isle_end_index):max(isle_init_index, isle_end_index)]
                dir = (boundary[0] - boundary[-1]) / np.linalg.norm(boundary[0] - boundary[-1])
                dir = np.array([dir[1], -dir[0]])
                for p in boundary:
                    dis = abs(np.dot(p - boundary[0], dir))
                    if dis > int(np.round(self.Clearance/step_len*width_coef)):
                        break
                if (p != boundary[-1]).any():
                    isle.append(boundary)
        return isle


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    path = Path(seg_num=10, clearance=1, is_straight=False)
    path.generate(show_now=True)
    path.draw_boundary(show_now=False)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    # torch.save(path, './data/path')
    # path = torch.load('./data/path')
    path.path_obstacles()
    plt.subplot(1, 3, 1)
    plt.imshow(torchvision.transforms.ToPILImage()(path.Space))
    plt.subplot(1, 3, 2)
    plt.imshow(torchvision.transforms.ToPILImage()(path.PathObs))
    plt.plot(path.PathPoint.T[1], path.PathPoint.T[0])
    plt.plot(path.ConvexHull.T[1], path.ConvexHull.T[0])
    plt.subplot(1, 3, 3)
    plt.imshow(torchvision.transforms.ToPILImage()(path.PathObs.cuda() + path.Space.cuda()))
    plt.show()
