from itertools import islice
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import queue
# Fit the data into the DBSCAN model
class CustomDBSCAN:
    def __init__(self):
        self.core = -1
        self.border = -2
    def fit(self, data, Eps, MinPt):
        # initialize all points as outliers
        point_label = [0] * len(data)
        point_count = []

        # initilize list for core/border points
        core = []
        border = []

        # Find the neighbours of each individual point
        for i in range(len(data)):
            point_count.append(self.neighbour_points(data, i, Eps))

        # Find all the core points, border points and outliers
        for i in range(len(point_count)):
            if (len(point_count[i]) >= MinPt):
                point_label[i] = self.core
                core.append(i)
            else:
                border.append(i)

        for i in border:
            for j in point_count[i]:
                if j in core:
                    point_label[i] = self.border
                    break

        # Assign points to a cluster

        cluster = 1

        # Here we use a queue to find all the neighbourhood points of a core point and find the
        # indirectly reachable points.
        # We are essentially performing Breadth First search of all points which are within
        # epsilon distance from each other
        for i in range(len(point_label)):
            q = queue.Queue()
            if (point_label[i] == self.core):
                point_label[i] = cluster
                for x in point_count[i]:
                    if(point_label[x] == self.core):
                        q.put(x)
                        point_label[x] = cluster
                    elif(point_label[x] == self.border):
                        point_label[x] = cluster
                while not q.empty():
                    neighbors = point_count[q.get()]
                    for y in neighbors:
                        if (point_label[y] == self.core):
                            point_label[y] = cluster
                            q.put(y)
                        if (point_label[y] == self.border):
                            point_label[y] = cluster
                cluster += 1  # Move on to the next cluster

        return point_label, cluster
    def neighbour_points(self, data, pointId, epsilon):
        points = []
        for i in range(len(data)):
            # Euclidian distance
            if np.linalg.norm([a_i - b_i for a_i, b_i in zip(data[i], data[pointId])]) <= epsilon:
                points.append(i)
        return points
        # Visualize the clusters
    def visualize(self, data, cluster, numberOfClusters):
        N = len(data)
        # Define colors, ideally better to have around 7-10 colors defined
        colors = ['g','k', 'c', 'm', 'y', 'r']

        for i in range(numberOfClusters):
            if (i == 0):
                # Plot all outliers point as blue
                color = 'b'
            else:
                color = colors[i % len(colors)]

            x, y = [], []
            for j in range(N):
                if cluster[j] == i:
                    x.append(data[j, 0])
                    y.append(data[j, 1])
            plt.scatter(x, y, c=color, alpha=1, marker='.')
            plt.show()



            # Reading from the data file
df = pd.read_csv("boxes.csv")

dataset = df.astype(float).values.tolist()

            # normalize dataset
X = StandardScaler().fit_transform(dataset)

custom_DBSCAN = CustomDBSCAN()
point_labels, clusters = custom_DBSCAN.fit(X, 0.25, 4)

print(point_labels, clusters)

custom_DBSCAN.visualize(X, point_labels, clusters)

