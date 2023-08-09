from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
points = np.random.rand(30, 2)
hull = ConvexHull(points)

plt.plot(points[:, 0], points[:, 1], 'o')  # 绘制原始点
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')  # 绘制边框

print("凸包顶点：")
print(points[hull.vertices])
# 绘制凸包的顶点（逆时针排列）
plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')

plt.show()
