import random
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
# Exercize 2 part II
def ex2():
    # all points in all dimensions
    points = defaultdict()
    for d in range(11):
        # all points in current dimension
        curr_dim = []
        for iter in range(100):
            # current point coordinates
            curr = []
            for i in range(2**d):
                curr.append(random.uniform(0,1))
            curr_dim.append(curr)
        points[d] = curr_dim
    # for each dimension
    euclid_means = []
    manhattan_means = []
    euclid_stdevs = []
    manhattan_stdevs = []
    for d in points:
        curr_points = points[d]
        euclidean_sum = []
        manhattan_sum = []
        # first point in dimension
        for i in range(len(curr_points)):
            x = curr_points[i]
            # second point in dimension
            for j in range(1+i, len(curr_points)):
                curr_euclidean = 0
                curr_manhattan = 0
                y = curr_points[j]
                # for magnitude of each dimension
                for index in range(len(x)):
                    curr_x = x[index]
                    curr_y = y[index]
                    curr_euclidean += (curr_x-curr_y)**2
                    curr_manhattan += abs(curr_x-curr_y)
                euclidean_sum.append(curr_euclidean)
                manhattan_sum.append(curr_manhattan)
        euclid_means.append(statistics.mean(euclidean_sum))
        manhattan_means.append(statistics.mean(manhattan_sum))
        euclid_stdevs.append(statistics.stdev(euclidean_sum))
        manhattan_stdevs.append(statistics.stdev(manhattan_sum))
    axis = [0,1,2,3,4,5,6,7,8,9,10]
    manhattan = plt.subplot(1,2,1)
    euclid = plt.subplot(1,2,2)
    euclid.plot(axis, euclid_means, label = "Euclidean Mean")
    euclid.plot(axis,euclid_stdevs,label = "Euclidean Standard Deviation")
    euclid.set_title("Euclidean")
    manhattan.plot(axis, manhattan_means, label = "Manhattan Mean")
    manhattan.plot(axis, manhattan_stdevs, label = "Manhattan Standard Deviation")
    manhattan.set_title("Manhattan")
    euclid.legend()
    manhattan.legend()
    plt.show()
    print(axis)
    print("Euclidean Means")
    print(euclid_means)
    print("Eucidean Stardard Deviations")
    print(euclid_stdevs)
    print("Manhattan Means")
    print(manhattan_means)
    print("Manhattan Standard Deviations")
    print(manhattan_stdevs)

ex2()
