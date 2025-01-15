# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import CLARA
import networkx as nx
import matplotlib.pyplot as plt

# 1. Generate Dummy Data for AML Analysis
np.random.seed(42)
transactions = pd.DataFrame({
    'Transaction_ID': np.arange(1, 101),
    'Amount': np.random.uniform(1000, 50000, 100),
    'Country_Risk': np.random.uniform(1, 5, 100),
    'Customer_Risk': np.random.uniform(1, 10, 100),
    'Transaction_Type': np.random.choice(['Online', 'Offline', 'Wire'], 100)
})

# Normalize numerical columns for clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
transactions[['Amount', 'Country_Risk', 'Customer_Risk']] = scaler.fit_transform(transactions[['Amount', 'Country_Risk', 'Customer_Risk']])

# 2. Clustering Models: KMeans vs CLARA
X = transactions[['Amount', 'Country_Risk', 'Customer_Risk']].values

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# CLARA Clustering
clara = CLARA(n_clusters=3, random_state=42)
clara_labels = clara.fit_predict(X)
clara_silhouette = silhouette_score(X, clara_labels)

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize PCA-enhanced KMeans
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.title("PCA-enhanced KMeans Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

# 3. Graph Theory for Criminal Network Analysis
# Create dummy graph data for a criminal network
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [
    ('A', 'B', 10), ('A', 'C', 20), ('B', 'C', 30), 
    ('B', 'D', 40), ('C', 'E', 50), ('D', 'F', 60)
]

# Initialize the graph
G = nx.Graph()
G.add_weighted_edges_from(edges)

# Visualize the Graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
plt.title("Criminal Network Graph")
plt.show()

# Analyze Graph Metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

print("Graph Metrics:")
print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)

# Airline Model Analogy
# Find shortest paths (similar to finding shortest routes in airlines)
shortest_paths = dict(nx.shortest_path_length(G, weight='weight'))
print("\nShortest Paths (Weighted):")
for source, paths in shortest_paths.items():
    print(f"From {source}: {paths}")

# Reinforcement Learning Example for AML (Dummy Framework)
# Here we simulate an RL model updating based on feedback

class FeedbackBasedReinforcement:
    def __init__(self):
        self.threshold = 0.5
    
    def predict(self, risk_score):
        return 1 if risk_score > self.threshold else 0

    def update_threshold(self, feedback):
        self.threshold += feedback * 0.1  # Simulate feedback adjustment

# Simulate risk scores and feedback
risk_scores = np.random.uniform(0, 1, 20)
feedback = np.random.choice([-1, 1], 20)  # -1 for False Positive, 1 for True Positive
model = FeedbackBasedReinforcement()

print("\nReinforcement Learning Updates:")
for i, (score, fb) in enumerate(zip(risk_scores, feedback)):
    pred = model.predict(score)
    print(f"Iteration {i+1}: Score={score:.2f}, Prediction={pred}, Feedback={fb}")
    model.update_threshold(fb)
    print(f"Updated Threshold: {model.threshold:.2f}")
