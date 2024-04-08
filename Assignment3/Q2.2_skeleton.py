# Import useful libraries. Feel free to use sklearn.
from sklearn.datasets import make_blobs
import random
import math
from collections import defaultdict
from statistics import mean
import copy
import matplotlib.pyplot as plt
import numpy as np

# Construct a 2D toy dataset for clustering.
X, _ = make_blobs(n_samples=1000,
                  centers=[[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]],
                  cluster_std=[0.2, 0.3, 0.3, 0.3, 0.3],
                  random_state=26)

# Conduct clustering on X using k-Means, and determine the best k with the elbow method.
def k_means(k, X):
    curr_cluster = None
    new_cluster = []
    for _ in range(k):
        # curr_cluster.append([random.uniform(-3,3), random.uniform(-3,3)])
        new_cluster.append([random.uniform(-3,3), random.uniform(-3,3)])
    iterations = 0
    var = 0
    while not curr_cluster or new_cluster != curr_cluster:
        var = 0
        iterations += 1
        curr_cluster = copy.deepcopy(new_cluster)
        countX = defaultdict(list)
        countY = defaultdict(list)
        for x,y in X:
            mindist = float('inf')
            minindex = 0
            for i,c in enumerate(new_cluster):
                dist = math.sqrt((x-c[0])**2+(y-c[1])**2)
                if dist < mindist:
                    mindist = dist
                    minindex = i
            countX[minindex].append(x)
            countY[minindex].append(y)
        for key in countX:
            new_cluster[key][0] = mean(countX[key])
            for x in countX[key]:
                var += abs(new_cluster[key][0]-x)
        for key in countY:
            new_cluster[key][1] = mean(countY[key])
            for y in countY[key]:
                var += abs(new_cluster[key][1]-y)
    return [var, countX,countY]
y = []
x = []
prev = 0
for i in range(1,11):
    dist = 0
    for _ in range(10):
        dist += k_means(i,X)[0]/1000
    y.append(i)
    # x.append(100/dist)
    x.append(prev-dist/10)
xaxis = np.array(x)
yaxis = np.array(y)
plt.plot(yaxis,xaxis)
plt.show()

# visualize clusters
# temp = k_means(5,X)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# colour = ['r','b','g','y','orange']
# for i in range(5):
#     ax.scatter(temp[1][i],temp[2][i],c=colour[i])
# plt.show()