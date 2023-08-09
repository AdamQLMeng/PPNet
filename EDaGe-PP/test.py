import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import math
# 声明对象
t = np.arange(0, np.pi*2, np.pi/360)
points = []
for tt in t:
    xx = 2 * math.sin(tt)
    yy = 1 * math.cos(tt)
    plt.plot(xx, yy, 'go')
    points.append([xx, yy])
points = np.reshape(points, [-1, 2])
print(points.shape)
plt.plot(points.T[1], points.T[0], 'bo')
plt.show()