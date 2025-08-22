"""
Customer Segmentation Using Hierarchical Clustering Analysis
===========================================================
This script performs hierarchical clustering analysis on customer mall data to:
1. Segment customers based on annual income and spending behavior
2. Create dendrogram visualization to determine optimal cluster number
3. Apply Agglomerative clustering with Ward linkage
4. Visualize customer segments for business insights

Dataset: Customer Mall dataset with income and spending score features
Algorithm: Agglomerative Hierarchical Clustering with Ward linkage
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(filepath='CustomerMall.csv'):
    """
    Load customer mall dataset and perform initial exploration.
    
    Args:
        filepath (str): Path to the customer mall CSV file
        
    Returns:
        tuple: (DataFrame, feature_matrix) - Original data and selected features
    """
    print("=== Customer Segmentation - Hierarchical Clustering Analysis ===\n")
    
    # Load dataset
    dataset = pd.read_csv(filepath)
    print(f"Dataset Overview:")
    print(f"Shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(dataset.head())
    
    # Extract features: Annual Income and Spending Score
    # Assuming columns 3 and 4 are Annual Income and Spending Score
    X = dataset.iloc[:, [3, 4]].values
    feature_names = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    print(f"\nSelected Features: {feature_names}")
    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Feature Statistics:")
    print(f"Annual Income - Min: {X[:, 0].min():.1f}, Max: {X[:, 0].max():.1f}, Mean: {X[:, 0].mean():.1f}")
    print(f"Spending Score - Min: {X[:, 1].min():.1f}, Max: {X[:, 1].max():.1f}, Mean: {X[:, 1].mean():.1f}")
    
    return dataset, X

def create_dendrogram(X, method='ward'):
    """
    Create and display dendrogram for hierarchical clustering.
    
    Args:
        X (numpy.ndarray): Feature matrix
        method (str): Linkage method for clustering
    """
    print(f"\n=== Dendrogram Analysis ===")
    print(f"Creating dendrogram with '{method}' linkage method...")
    
    plt.figure(figsize=(12, 8))
    dendrogram = sch.dendrogram(
        sch.linkage(X, method=method),
        truncate_mode='lastp',
        p=30,  # Show last 30 clusters
        show_leaf_counts=True,
        leaf_font_size=10
    )
    
    plt.title(f'Dendrogram - Hierarchical Clustering ({method.title()} Linkage)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Customers (Sample Index or Cluster Size)', fontsize=12)
    plt.ylabel('Euclidean Distance', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("📊 Tip: Look for the largest vertical distances to determine optimal clusters")

def perform_hierarchical_clustering(X, n_clusters=5, linkage='ward'):
    """
    Perform Agglomerative Hierarchical Clustering.
    
    Args:
        X (numpy.ndarray): Feature matrix
        n_clusters (int): Number of clusters to form
        linkage (str): Linkage criterion
        
    Returns:
        tuple: (clustering_model, cluster_labels)
    """
    print(f"\n=== Hierarchical Clustering Results ===")
    print(f"Applying Agglomerative Clustering with {n_clusters} clusters...")
    
    # Initialize and fit the clustering model
    hc = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='euclidean',  # Changed from 'affinity' to 'metric'
        linkage=linkage
    )
    
    cluster_labels = hc.fit_predict(X)
    
    # Display clustering information
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster Distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {label}: {count} customers ({percentage:.1f}%)")
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"\nSilhouette Score: {silhouette_avg:.3f}")
    
    return hc, cluster_labels

def visualize_customer_segments(X, cluster_labels, n_clusters=5):
    """
    Create comprehensive visualization of customer segments.
    
    Args:
        X (numpy.ndarray): Feature matrix
        cluster_labels (numpy.ndarray): Cluster assignments
        n_clusters (int): Number of clusters
    """
    # Define colors for clusters
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
    cluster_names = [f'Cluster {i+1}' for i in range(n_clusters)]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Scatter plot with clusters
    plt.subplot(1, 3, 1)
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        plt.scatter(
            X[cluster_mask, 0], X[cluster_mask, 1], 
            s=80, c=colors[i], alpha=0.7, 
            label=cluster_names[i], edgecolors='black', linewidth=0.5
        )
    
    plt.title('Customer Segments\n(Hierarchical Clustering)', fontweight='bold', fontsize=12)
    plt.xlabel('Annual Income (k$)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cluster centers analysis
    plt.subplot(1, 3, 2)
    cluster_centers = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        center_x = X[cluster_mask, 0].mean()
        center_y = X[cluster_mask, 1].mean()
        cluster_centers.append([center_x, center_y])
        
        plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], 
                   s=60, c=colors[i], alpha=0.6, label=cluster_names[i])
        plt.scatter(center_x, center_y, s=200, c='black', marker='x', linewidths=3)
    
    plt.title('Customer Segments with Centroids', fontweight='bold', fontsize=12)
    plt.xlabel('Annual Income (k$)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cluster size distribution
    plt.subplot(1, 3, 3)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar(cluster_names, counts, color=colors[:n_clusters], alpha=0.7, edgecolor='black')
    plt.title('Cluster Size Distribution', fontweight='bold', fontsize=12)
    plt.xlabel('Customer Segments', fontsize=11)
    plt.ylabel('Number of Customers', fontsize=11)
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 1, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return cluster_centers

def analyze_customer_segments(X, cluster_labels, cluster_centers, n_clusters=5):
    """
    Provide business insights for each customer segment.
    
    Args:
        X (numpy.ndarray): Feature matrix
        cluster_labels (numpy.ndarray): Cluster assignments
        cluster_centers (list): Cluster center coordinates
        n_clusters (int): Number of clusters
    """
    print(f"\n=== Customer Segment Analysis ===")
    
    segment_profiles = {
        0: "Budget Conscious",
        1: "High Value",
        2: "Average Customers", 
        3: "Potential Targets",
        4: "Premium Customers"
    }
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_data = X[cluster_mask]
        
        avg_income = cluster_data[:, 0].mean()
        avg_spending = cluster_data[:, 1].mean()
        cluster_size = len(cluster_data)
        
        print(f"\n📊 Cluster {i} - {segment_profiles.get(i, f'Segment {i}')}:")
        print(f"   Size: {cluster_size} customers")
        print(f"   Average Annual Income: ${avg_income:.1f}k")
        print(f"   Average Spending Score: {avg_spending:.1f}/100")
        
        # Business insights based on income and spending patterns
        if avg_income > 60 and avg_spending > 60:
            insight = "🌟 High-value customers with strong purchasing power"
        elif avg_income < 40 and avg_spending < 40:
            insight = "💰 Budget-conscious segment, focus on value propositions"
        elif avg_income > 60 and avg_spending < 40:
            insight = "🎯 High income but low spending - untapped potential"
        elif avg_income < 40 and avg_spending > 60:
            insight = "⚠️  High spending despite lower income - monitor credit risk"
        else:
            insight = "📈 Balanced segment with moderate income and spending"
        
        print(f"   Business Insight: {insight}")

def compare_cluster_methods(X, max_clusters=8):
    """
    Compare different numbers of clusters using silhouette analysis.
    
    Args:
        X (numpy.ndarray): Feature matrix
        max_clusters (int): Maximum number of clusters to test
    """
    print(f"\n=== Optimal Cluster Analysis ===")
    
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        hc = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')  # Changed 'affinity' to 'metric'
        labels = hc.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.3f}")
    
    # Find optimal k
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_range[best_k_idx]
    best_score = silhouette_scores[best_k_idx]
    
    print(f"\n🏆 Optimal number of clusters: {best_k} (Silhouette Score: {best_score:.3f})")
    
    # Plot silhouette analysis
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.8, 
               label=f'Optimal k = {best_k}')
    plt.title('Silhouette Analysis - Hierarchical Clustering', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return best_k

def main():
    """
    Main function to execute the complete hierarchical clustering analysis.
    """
    # Load and explore data
    dataset, X = load_and_explore_data('CustomerMall.csv')
    
    # Optional: Feature scaling (uncomment if needed)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X = X_scaled
    # print("✅ Features have been standardized")
    
    # Create dendrogram for optimal cluster selection
    create_dendrogram(X, method='ward')
    
    # Find optimal number of clusters
    optimal_k = compare_cluster_methods(X)
    
    # Perform hierarchical clustering with optimal clusters
    hc_model, cluster_labels = perform_hierarchical_clustering(X, n_clusters=optimal_k)
    
    # Visualize customer segments
    cluster_centers = visualize_customer_segments(X, cluster_labels, optimal_k)
    
    # Provide business insights
    analyze_customer_segments(X, cluster_labels, cluster_centers, optimal_k)
    
    # Final summary
    print(f"\n=== Analysis Summary ===")
    print(f"Dataset: Customer Mall segmentation data")
    print(f"Features: Annual Income & Spending Score")
    print(f"Algorithm: Agglomerative Hierarchical Clustering (Ward linkage)")
    print(f"Optimal clusters: {optimal_k}")
    print(f"Silhouette score: {silhouette_score(X, cluster_labels):.3f}")
    
    print(f"\n🎯 Business Recommendations:")
    print(f"• Use segment profiles for targeted marketing campaigns")
    print(f"• Focus retention efforts on high-value customer segments")
    print(f"• Develop specific product offerings for each segment")
    print(f"• Monitor customer migration between segments over time")
    
    print(f"\n✅ Hierarchical clustering analysis completed successfully!")

if __name__ == "__main__":
    main()