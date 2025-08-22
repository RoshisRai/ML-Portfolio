"""
Synthetic Dataset Hierarchical Clustering Analysis
=================================================
This script demonstrates hierarchical clustering on synthetic 2D data with:
1. Data visualization with point annotations
2. Dendrogram analysis for optimal cluster selection
3. Agglomerative clustering with different linkage methods
4. Cluster visualization and comparison

Dataset: Custom synthetic 2D coordinates (10 points)
Algorithm: Agglomerative Hierarchical Clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def create_and_visualize_data():
    """Create synthetic dataset and visualize original points."""
    print("=== Synthetic Dataset Hierarchical Clustering ===\n")
    
    # Synthetic 2D data points
    X = np.array([
        [5, 3], [10, 15], [15, 12], [24, 10], [30, 30],
        [85, 70], [71, 80], [60, 78], [70, 55], [80, 91]
    ])
    
    labels = range(1, 11)
    
    print(f"Dataset: {X.shape[0]} points in 2D space")
    print(f"Coordinate ranges: X({X[:,0].min()}-{X[:,0].max()}), Y({X[:,1].min()}-{X[:,1].max()})")
    
    # Visualize original data
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=100, c='blue', alpha=0.7, edgecolors='black')
    
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(3, 3), 
                    textcoords='offset points', fontweight='bold')
    
    plt.title('Original Synthetic Dataset', fontsize=14)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return X

def create_dendrogram(X):
    """Create dendrogram to determine optimal number of clusters."""
    print(f"\n=== Dendrogram Analysis ===")
    
    plt.figure(figsize=(12, 6))
    
    # Compare different linkage methods
    methods = ['single', 'ward']
    for i, method in enumerate(methods):
        plt.subplot(1, 2, i+1)
        
        linked = linkage(X, method=method)
        dendrogram(linked, orientation='top', labels=range(1, 11),
                  distance_sort='descending', show_leaf_counts=True)
        
        plt.title(f'Dendrogram - {method.title()} Linkage')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
    
    plt.tight_layout()
    plt.show()

def perform_clustering_analysis(X):
    """Perform clustering with different configurations and find optimal parameters."""
    print(f"\n=== Clustering Analysis ===")
    
    linkage_methods = ['ward', 'complete', 'average']
    best_score = -1
    best_config = None
    
    # Test different configurations
    for method in linkage_methods:
        for k in range(2, 5):
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=k, metric='euclidean', linkage=method
                )
                labels = clusterer.fit_predict(X)
                score = silhouette_score(X, labels)
                
                print(f"{method} linkage, k={k}: Silhouette Score = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_config = (method, k, labels)
                    
            except Exception as e:
                print(f"{method} linkage, k={k}: Error - {str(e)}")
    
    return best_config, best_score

def visualize_final_results(X, best_config, best_score):
    """Visualize the best clustering results."""
    method, k, labels = best_config
    
    print(f"\n=== Final Results ===")
    print(f"🏆 Best Configuration: {method} linkage with {k} clusters")
    print(f"📊 Silhouette Score: {best_score:.3f}")
    
    # Display cluster information
    for cluster_id in range(k):
        cluster_points = np.sum(labels == cluster_id)
        print(f"Cluster {cluster_id}: {cluster_points} points")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Clustering result
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         s=100, alpha=0.8, edgecolors='black')
    
    # Annotate points
    for i, (x, y) in enumerate(X):
        plt.annotate(f'P{i+1}', xy=(x, y), xytext=(3, 3),
                    textcoords='offset points', fontweight='bold')
    
    plt.title(f'Hierarchical Clustering Results\n{method.title()} Linkage, k={k}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cluster distribution
    plt.subplot(1, 2, 2)
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    bars = plt.bar(unique_labels, counts, color=colors, alpha=0.8, edgecolor='black')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', fontweight='bold')
    
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Points')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Execute the complete hierarchical clustering analysis."""
    # Create and visualize data
    X = create_and_visualize_data()
    
    # Create dendrogram for cluster selection
    create_dendrogram(X)
    
    # Find optimal clustering configuration
    best_config, best_score = perform_clustering_analysis(X)
    
    if best_config:
        # Visualize final results
        visualize_final_results(X, best_config, best_score)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"💡 Insight: The dataset naturally separates into clusters based on spatial proximity")
    else:
        print("❌ No valid clustering configuration found")

if __name__ == "__main__":
    main()