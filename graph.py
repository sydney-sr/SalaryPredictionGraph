# === imports ===
import pandas as pd                    # for data manipulation
import heapq                           # for priority queue in dijkstra's algorithm
import time                            # for measuring time if needed
import networkx as nx
import matplotlib.pyplot as plt        # for plotting
import ipywidgets as widgets           # for interactive widgets in Jupyter
from IPython.display import display    # to display widgets
import numpy as np                     # for numerical operations
from io import BytesIO                 # for in-memory byte streams
from PIL import Image as PILImage      # for image handling

# === data load & processing ===
df = pd.read_csv('content/salary_prediction_updated_dataset.csv')  # load salary dataset

# round years of experience and bin into integers
df['ExpBin'] = df['YearsExperience'].round().astype(int)

# bin salary into 10,000 increments
df['SalaryBin'] = (df['Salary'] // 10000) * 10000

# compute average salary by job role, experience, and education level
avg_salary_job = df.groupby('Job Role')['Salary'].mean().to_dict()
avg_salary_exp = df.groupby('ExpBin')['Salary'].mean().to_dict()
avg_salary_edu = df.groupby('Education Level')['Salary'].mean().to_dict()
salary_bins = df['SalaryBin'].unique()  # get unique salary bins

# === graph class ===
class Graph:
    def __init__(self):
        # use a dictionary to store adjacency list for each node
        self.adj = {}

    def add_node(self, node):
        # ensure node is in the graph with an empty edge list
        self.adj.setdefault(node, [])

    def add_edge(self, u, v, weight):
        # add a weighted edge between nodes u and v (undirected)
        self.adj.setdefault(u, []).append((v, weight))
        self.adj.setdefault(v, []).append((u, weight))

    def neighbors(self, u):
        # return list of neighbors and edge weights for node u
        return self.adj.get(u, [])

    def dijkstra(self, source):
        # compute shortest paths from source using dijkstra's algorithm
        dist = {node: float('inf') for node in self.adj}
        dist[source] = 0
        prev = {}
        pq = [(0, source)]  # priority queue initialized with source
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.neighbors(u):
                if (d + w) < dist[v]:
                    dist[v] = d + w
                    prev[v] = u
                    heapq.heappush(pq, (d + w, v))
        return dist, prev

    def bellman_ford(self, source):
        # compute shortest paths from source using bellman-ford algorithm
        dist = {node: float('inf') for node in self.adj}
        dist[source] = 0
        prev = {}
        # relax all edges |V| - 1 times
        for _ in range(len(self.adj) - 1):
            for u in self.adj:
                for v, w in self.adj[u]:
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        prev[v] = u
        return dist, prev

    def has_edge(self, u, v):
        # check if an edge exists from node u to node v
        return any(neighbor == v for neighbor, _ in self.adj.get(u, []))
