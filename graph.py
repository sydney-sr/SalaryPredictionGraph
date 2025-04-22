# sydney sr, divya verma, anna kate wheeler

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

# === Visualization ===
category_colors = {
    'Job Role': 'lightblue',
    'ExpBin': 'lightgreen',
    'Education Level': 'lightcoral',
    'SalaryBin': 'lightyellow'}

# plots just the path from variables given to salary prediction
def visualize_only_path(path, graph, title):
    G_path = nx.Graph()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        weight = next((w for nbr, w in graph.adj[u] if nbr == v), 1)
        G_path.add_edge(u, v, weight=weight)

    # adjust visualization
    pos = nx.spring_layout(G_path)
    node_colors = []
    for node in G_path.nodes:
        if node in avg_salary_job: node_colors.append(category_colors['Job Role'])
        elif node in avg_salary_exp: node_colors.append(category_colors['ExpBin'])
        elif node in avg_salary_edu: node_colors.append(category_colors['Education Level'])
        elif node in salary_bins: node_colors.append(category_colors['SalaryBin'])

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G_path, pos, with_labels=True, node_color=node_colors, node_size=3000, font_size=10, ax=ax)
    nx.draw_networkx_edges(G_path, pos, edge_color='red', width=2, ax=ax)
    ax.set_title(title)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# === UI ===
#welcome screen
welcome_text = widgets.HTML(value="<h1>Welcome to the Salary Predictor!</h1>")
start_button = widgets.Button(description="Start")

#input screen
name_input = widgets.Text(description="Name:")
major_input = widgets.Dropdown(options=list(avg_salary_job.keys()), description="Job Role:")
edu_input = widgets.Dropdown(options=list(avg_salary_edu.keys()), description="Education:")
exp_input = widgets.BoundedIntText(value=1, min=0, max=40, description="Experience:")
submit_button = widgets.Button(description="Submit")

#output screen
output_text = widgets.HTML()
output_image1 = widgets.Image(format='png')
output_image2 = widgets.Image(format='png')

#input screen widgets
def show_input_page(_):
    welcome_text.close()
    start_button.close()
    display(widgets.VBox([name_input, major_input, edu_input, exp_input, submit_button]))

#output screen widgets
def show_output_page(_):
    global G
    user = name_input.value
    job = major_input.value
    edu = edu_input.value
    exp = exp_input.value
    # makes graph with user given variables
    G = build_graph_with_profile(user_exp=exp, user_edu=edu, user_job=job)

    # Dijkstra
    start_time_dijkstra = time.time()
    path_d, salary_d = get_shortest_salary_path(G, job, method='dijkstra', user_exp=exp, user_edu=edu, user_job=job)
    dijkstra_duration = time.time() - start_time_dijkstra

    # Bellman-Ford
    start_time_bf = time.time()
    path_bf, salary_bf = get_shortest_salary_path(G, job, method='bellman_ford', user_exp=exp, user_edu=edu, user_job=job)
    bf_duration = time.time() - start_time_bf

    # handles exceptions
    if salary_d is None and salary_bf is None:
        output_text.value = "<h2 style='font-size:18px;'>No valid path found for your selection.</h2>"
        display(widgets.VBox([output_text]))
        return

    # print graphs and time for algorithm - dijkstra
    if path_d:
        output_image1.value = visualize_only_path(path_d, G, "Dijkstra Path")
        dijkstra_widget = widgets.VBox([
            widgets.Image(value=output_image1.value, format='png', width=600, height=600),
            widgets.HTML(value=f"<div style='text-align:center; font-size:18px;'>⏱️ {dijkstra_duration:.6f} seconds</div>")
        ])
    else:
        dijkstra_widget = widgets.HTML(value="<div style='text-align:center;'>No path</div>")

    # print graphs and time per algorithm - bellman-ford
    if path_bf:
        output_image2.value = visualize_only_path(path_bf, G, "Bellman-Ford Path")
        bf_widget = widgets.VBox([
            widgets.Image(value=output_image2.value, format='png', width=600, height=600),
            widgets.HTML(value=f"<div style='text-align:center; font-size:18px;'>⏱️ {bf_duration:.6f} seconds</div>")
        ])
    else:
        bf_widget = widgets.HTML(value="<div style='text-align:center;'>No path</div>")

    # format output
    output_text.value = f"""
        <h2 style='font-size:24px;'>Hello, {user}!</h2>
        <h3 style='font-size:20px;'>Predicted Salaries</h3>
        <ul style='font-size:18px;'>
            {f"<li><b>Dijkstra:</b> ${salary_d}</li>" if salary_d else ""}
            {f"<li><b>Bellman-Ford:</b> ${salary_bf}</li>" if salary_bf else ""}
        </ul>
    """
    graphs = widgets.HBox([dijkstra_widget, bf_widget])
    display(widgets.VBox([output_text, graphs]))
    print_full_graph(G)

start_button.on_click(show_input_page)
submit_button.on_click(show_output_page)

display(widgets.VBox([welcome_text, start_button]))

# convert custom Graph object to NetworkX graph / display
def print_full_graph(G):
    nx_graph = nx.Graph()
    # create networkx graph
    for node in G.adj:
        nx_graph.add_node(node)
    for u in G.adj:
        for v, weight in G.adj[u]:
            nx_graph.add_edge(u, v, weight=weight)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(nx_graph, k=.2, seed=42)  # Adjust "k" for more spacing (increase k if clustering persists)

    # assign visualization colors
    node_colors = []
    for node in nx_graph.nodes:
        if node in avg_salary_job:
            node_colors.append('lightblue')  # Job Roles
        elif node in avg_salary_exp:
            node_colors.append('lightgreen')  # Experience
        elif node in avg_salary_edu:
            node_colors.append('lightcoral')  # Education
        elif node in salary_bins:
            node_colors.append('lightyellow')  # Salary Bins

    # draw nodes and edges
    nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, font_weight='bold', node_size=3000, font_size=8)
    nx.draw_networkx_edges(nx_graph, pos, width=1.5, alpha=0.5, edge_color='gray')
    plt.title('Entire Graph - Salary Prediction Model', fontsize=16)
    plt.tight_layout()
    plt.show()