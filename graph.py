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



# === Smarter Weighting ===
def smart_edge_weight(u, v, user_exp=None, user_edu=None, user_job=None):
    edu_levels = {
        "High School": 0,
        "Associate Degree": 1,
        "Bachelor's Degree": 2,
        "Master's Degree": 3,
        "PhD": 4
    }

    job_min_edu = {
        "Business Analyst": 2,
        "Data Scientist": 3,
        "Marketing Specialist": 2,
        "Product Manager": 3,
        "Software Engineer": 2
    }

    job_tiers = {
        "Marketing Specialist": 0,
        "Business Analyst": 1,
        "Software Engineer": 2,
        "Product Manager": 3,
        "Data Scientist": 4
    }

    base_salary_u = avg_salary_job.get(u) or avg_salary_exp.get(u) or avg_salary_edu.get(u) or u
    base_salary_v = avg_salary_job.get(v) or avg_salary_exp.get(v) or avg_salary_edu.get(v) or v
    weight = abs(base_salary_u - base_salary_v)

    if user_exp is not None and isinstance(v, int):
        experience_diff = abs(v - user_exp)
        if experience_diff == 0:
            weight *= 0.1
        elif experience_diff <= 2:
            weight *= 0.5
        elif experience_diff <= 5:
            weight *= 1
        elif experience_diff <= 8:
            weight *= 3
        else:
            weight *= 1000

    if user_edu is not None and isinstance(v, str):
        if v == user_edu:
            weight *= 0.1
        else:
            weight *= 5

    if user_edu is not None and isinstance(v, (int, float)) and v in salary_bins:
        edu_salary = avg_salary_edu.get(user_edu, 0)
        if (edu_salary * 0.8) <= v <= (edu_salary * 1.2):
            weight *= 0.5
        else:
            weight *= 5
        edu_level = edu_levels.get(user_edu, 0)
        weight *= 1 / (1 + edu_level * 0.15)

    if user_job is not None and isinstance(v, (int, float)) and v in salary_bins:
        job_salary = avg_salary_job.get(user_job, 0)
        if (job_salary * 0.8) <= v <= (job_salary * 1.2):
            weight *= 0.5
        else:
            weight *= 5
        job_tier = job_tiers.get(user_job, 0)
        weight *= 1 / (1 + job_tier * 0.1)

    if user_job and user_edu and isinstance(v, str):
        user_level = edu_levels.get(user_edu, 0)
        min_required = job_min_edu.get(user_job, 0)
        if user_level < min_required:
            weight *= 10
        elif user_level > min_required + 1:
            weight *= 3

    if user_job and user_edu and isinstance(v, (int, float)) and v in salary_bins:
        user_level = edu_levels.get(user_edu, 0)
        min_required = job_min_edu.get(user_job, 0)
        if user_level < min_required:
            weight *= 20
        elif user_level > min_required + 1:
            weight *= 3

    return weight

# === Build Graph ===
def build_graph_with_profile(user_exp=None, user_edu=None, user_job=None):
    graph = Graph()
    for job in avg_salary_job: graph.add_node(job)
    for exp in avg_salary_exp: graph.add_node(exp)
    for edu in avg_salary_edu: graph.add_node(edu)
    for sal in salary_bins: graph.add_node(sal)

    for job in avg_salary_job:
        for edu in avg_salary_edu:
            w = smart_edge_weight(job, edu, user_edu=user_edu, user_job=user_job)
            graph.add_edge(job, edu, w)

    for edu in avg_salary_edu:
        for exp in avg_salary_exp:
            w = smart_edge_weight(edu, exp, user_exp=user_exp, user_edu=user_edu, user_job=user_job)
            graph.add_edge(edu, exp, w)

    for exp in avg_salary_exp:
        for sal in salary_bins:
            w = smart_edge_weight(exp, sal, user_exp=user_exp, user_edu=user_edu, user_job=user_job)
            graph.add_edge(exp, sal, w)

    return graph

# === Predict Salary ===
def get_shortest_salary_path(graph, source, method='dijkstra', user_exp=None, user_edu=None, user_job=None):
    if method == 'dijkstra':
        dist, prev = graph.dijkstra(source)
    else:
        dist, prev = graph.bellman_ford(source)

    def reconstruct_path(end):
        path = []
        while end in prev:
            path.append(end)
            end = prev[end]
        path.append(source)
        return list(reversed(path))

    sorted_nodes = sorted(dist.items(), key=lambda x: x[1])
    for node, _ in sorted_nodes:
        path = reconstruct_path(node)
        salary_node = next((n for n in reversed(path) if n in salary_bins), None)
        if salary_node:
            return path, salary_node
    return None, None
