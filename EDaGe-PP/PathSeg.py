import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

SegLenRange = 7
MinLen = 0


class PathSeg:
    def __init__(self, polyorder=4, dim=2, is_straight=False):
        self.PolyOrder = polyorder
        self.Poly = np.zeros([polyorder + 1, 1])
        self.EndPoint = 0
        self.Length = 0
        self.Translation = np.zeros([dim, 1])
        self.Rotation = 0
        self.GradSt = 0
        self.GradEnd = 0
        self.is_straight = True if is_straight or np.random.random(1) < 0.2 else False

    def random(self, poly=None, endpoint=None):
        x = np.arange(0, 1000) / 100
        y = np.random.random(1000) * 10 - 5
        self.Poly = np.polyfit(x, y, self.PolyOrder) if poly is None else np.array(poly)
        # print(self.Poly)
        # self.Poly = (np.random.random(self.PolyOrder+1)-0.5)/0.5
        # print(self.Poly)
        self.Poly[np.size(self.Poly) - 1] = 0
        if self.is_straight:
            for i in range(np.size(self.Poly) - 2):
                self.Poly[i] = 0
        self.EndPoint = np.random.random(1)*(SegLenRange - MinLen) + MinLen if poly is None else endpoint.item()
        self.translation()
        self.gradient()
        self.length()
        return self.Poly, self.EndPoint

    def translation(self):
        y_end = np.polyval(self.Poly, self.EndPoint)
        self.Translation = np.array([self.EndPoint, y_end])
        return self.Translation

    def gradient(self):
        p_d = np.polyder(self.Poly)
        self.GradSt = np.polyval(p_d, 0)
        self.GradEnd = np.polyval(p_d, self.EndPoint)
        return self.GradSt, self.GradEnd

    def length(self):
        x = np.arange(0, 100) / 100 * (self.EndPoint - 0)
        y = np.polyval(self.Poly, x)
        for i in range(np.size(x) - 1):
            self.Length = self.Length + distance.euclidean([x[i+1], y[i+1]], [x[i], y[i]])
        self.Length = self.Length + distance.euclidean(
            np.reshape(
                [self.EndPoint, np.polyval(self.Poly, self.EndPoint)], [2]),
            [x[np.size(x) - 1], y[np.size(x) - 1]])
        return self.Length

    def plot(self):
        x = np.arange(0, 100) / 100 * (self.EndPoint - 0)
        y = np.polyval(self.Poly, x)
        plt.plot(x, y)
        plt.show()

    def integral(self):
        integral = np.zeros([np.size(self.Poly)+1, 1])
        for i in range(np.size(self.Poly)-1):
            integral[i] = self.Poly[i] / (np.size(self.Poly) - i)
        return integral


if __name__ == '__main__':
    pathseg = PathSeg(polyorder=7)
    pathseg.random()
