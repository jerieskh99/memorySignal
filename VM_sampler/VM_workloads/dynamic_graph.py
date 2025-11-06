import numpy as np
import networkx as nx
import random
import time

# Define the initial number of nodes and edges
num_nodes = 10000
num_edges = 50000

# Create a large random graph using NetworkX
graph = nx.gnm_random_graph(num_nodes, num_edges)

iteration = 0

# Infinite loop to simulate dynamic graph changes
while True:
    # Randomly add or remove nodes and edges
    if random.random() < 0.5:
        # Add a new node with random edges
        new_node = max(graph.nodes) + 1
        num_new_edges = np.random.randint(1, 10)
        for _ in range(num_new_edges):
            target_node = np.random.choice(list(graph.nodes))
            graph.add_edge(new_node, target_node)
        print(f"Iteration {iteration}: Added node {new_node} with {num_new_edges} edges")
    else:
        # Remove a random node
        if len(graph.nodes) > 1:
            remove_node = np.random.choice(list(graph.nodes))
            graph.remove_node(remove_node)
            print(f"Iteration {iteration}: Removed node {remove_node}")

    # Simulate computation on the graph (e.g., pathfinding)
    node_a, node_b = np.random.choice(list(graph.nodes), 2, replace=False)
    try:
        path = nx.shortest_path(graph, source=node_a, target=node_b)
        print(f"Iteration {iteration}: Computed shortest path between {node_a} and {node_b} (length {len(path)})")
    except nx.NetworkXNoPath:
        print(f"Iteration {iteration}: No path between {node_a} and {node_b}")

    iteration += 1
    if iteration % 10 == 0:
        # Simulate periodic large-scale changes
        nodes_to_add = np.random.randint(100, 500)
        for _ in range(nodes_to_add):
            new_node = max(graph.nodes) + 1
            graph.add_node(new_node)
            for _ in range(5):
                graph.add_edge(new_node, np.random.choice(list(graph.nodes)))
        print(f"Iteration {iteration}: Bulk added {nodes_to_add} nodes")

    time.sleep(0.01)  # Small delay to simulate processing time
