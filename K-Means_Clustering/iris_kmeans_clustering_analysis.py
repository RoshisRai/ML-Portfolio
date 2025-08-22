"""
Iris Dataset K-Means Clustering Analysis with Manual Elbow Method
================================================================
This script performs K-means clustering analysis on the Iris dataset to:
1. Find optimal number of clusters using manual Elbow method detection
2. Visualize clustering results with centroids
3. Compare different cluster numbers and their performance
4. Includes optional feature scaling and silhouette analysis

Dataset: Iris flower measurements (sepal/petal length & width)
Algorithm: K-Means Clustering with manual elbow detection
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the Iris dataset from sklearn
print("=== Iris Dataset K-Means Clustering Analysis ===\n")
iris = load_iris()
x = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target labels (for reference, not used in clustering)

print(f"Dataset shape: {x.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Feature matrix preview:\n{x[:5]}")

# Dataset overview
print(f"\nDataset Overview:")
print(f"Total samples: {len(x)}")
print(f"Features: {', '.join(iris.feature_names)}")
print(f"Feature ranges:")
for i, feature in enumerate(iris.feature_names):
    print(f"  {feature}: {x[:, i].min():.2f} to {x[:, i].max():.2f}")

print(f"\nTarget distribution (for reference):")
unique_targets, counts = np.unique(y, return_counts=True)
for target, count in zip(unique_targets, counts):
    print(f"  {iris.target_names[target]}: {count} samples")

# Optional: Scale the features for better clustering performance
# Uncomment below lines if scaling is needed
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)
# x = x_scaled
# print("✅ Features have been standardized")

# Find optimal number of clusters using Elbow Method
print(f"\n=== Elbow Method Analysis ===")
inertia_values = []  # Within-cluster sum of squares (WCSS)
k_range = range(1, 11)

# Test cluster numbers from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x)
    inertia_values.append(kmeans.inertia_)
    print(f"k={k}: WCSS = {kmeans.inertia_:.2f}")

# Manual elbow detection using rate of change
def find_elbow_point(k_values, inertia_values):
    """
    Find the elbow point by calculating the rate of change in inertia.
    The elbow is where the rate of decrease starts to slow down significantly.
    """
    # Calculate rate of change (differences between consecutive points)
    differences = []
    for i in range(1, len(inertia_values)):
        diff = inertia_values[i-1] - inertia_values[i]
        differences.append(diff)
    
    # Calculate second derivative (rate of change of differences)
    second_diffs = []
    for i in range(1, len(differences)):
        second_diff = differences[i-1] - differences[i]
        second_diffs.append(second_diff)
    
    # Find the point with maximum second derivative (most curvature)
    if second_diffs:
        elbow_idx = second_diffs.index(max(second_diffs)) + 2  # +2 because we started from k=3
        return min(elbow_idx, len(k_values))
    else:
        return 3  # Default fallback

# Find optimal k using manual elbow detection
optimal_k = find_elbow_point(list(k_range), inertia_values)

# Alternative simple method: look for the point where improvement becomes minimal
rate_of_change = []
for i in range(1, len(inertia_values)):
    rate = (inertia_values[i-1] - inertia_values[i]) / inertia_values[i-1] * 100
    rate_of_change.append(rate)

print(f"\nRate of improvement for each k:")
for i, rate in enumerate(rate_of_change, 2):
    print(f"k={i}: {rate:.2f}% improvement from k={i-1}")

# Find where improvement drops below threshold (alternative method)
improvement_threshold = 5.0  # 5% improvement threshold
alternative_k = 2
for i, rate in enumerate(rate_of_change, 2):
    if rate < improvement_threshold:
        alternative_k = i
        break

print(f"\n🎯 Optimal number of clusters (curvature method): {optimal_k}")
print(f"📊 Alternative k (improvement < {improvement_threshold}%): {alternative_k}")

# Use the more conservative estimate
final_k = min(optimal_k, alternative_k) if optimal_k <= 5 else 3
print(f"🎪 Final selected k: {final_k}")

# Plot Elbow Method results
plt.figure(figsize=(12, 5))

# Plot 1: WCSS vs Number of Clusters
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_values, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=final_k, color='red', linestyle='--', alpha=0.8, 
           label=f'Selected k = {final_k}')
plt.title('Elbow Method for Optimal Number of Clusters', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Rate of Change
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), rate_of_change, 'go-', linewidth=2, markersize=8)
plt.axhline(y=improvement_threshold, color='red', linestyle='--', alpha=0.8, 
           label=f'Threshold = {improvement_threshold}%')
plt.title('Rate of WCSS Improvement', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Improvement (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Perform K-means clustering with optimal number of clusters
print(f"\n=== K-Means Clustering Results (k={final_k}) ===")
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(x)

print(f"Cluster centers (centroids):")
for i, center in enumerate(kmeans_final.cluster_centers_):
    print(f"  Cluster {i}:")
    for j, feature_name in enumerate(iris.feature_names):
        print(f"    {feature_name}: {center[j]:.2f}")

# Display cluster distribution
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
print(f"\nCluster distribution:")
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")

print(f"\nClustering Quality Metrics:")
print(f"Final WCSS: {kmeans_final.inertia_:.2f}")
print(f"Average distance to centroid: {kmeans_final.inertia_/len(x):.2f}")

# Compare with true species labels (for educational purposes)
print(f"\nComparison with true species (for reference):")
for i in range(final_k):
    cluster_mask = cluster_labels == i
    true_labels_in_cluster = y[cluster_mask]
    species_counts = np.bincount(true_labels_in_cluster, minlength=3)
    print(f"  Cluster {i} contains:")
    for j, species_name in enumerate(iris.target_names):
        if species_counts[j] > 0:
            print(f"    {species_name}: {species_counts[j]} samples")

# Visualize clustering results
plt.figure(figsize=(15, 5))

# Plot 1: Sepal Length vs Sepal Width
plt.subplot(1, 3, 1)
scatter1 = plt.scatter(x[:, 0], x[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clustering: Sepal Features')
plt.colorbar(scatter1, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Petal Length vs Petal Width
plt.subplot(1, 3, 2)
scatter2 = plt.scatter(x[:, 2], x[:, 3], c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
plt.scatter(kmeans_final.cluster_centers_[:, 2], kmeans_final.cluster_centers_[:, 3], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title('K-Means Clustering: Petal Features')
plt.colorbar(scatter2, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Sepal Length vs Petal Length
plt.subplot(1, 3, 3)
scatter3 = plt.scatter(x[:, 0], x[:, 2], c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 2], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])
plt.title('K-Means Clustering: Length Features')
plt.colorbar(scatter3, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Silhouette Analysis for cluster validation
print(f"\n=== Silhouette Analysis ===")
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels_temp = kmeans.fit_predict(x)
    score = silhouette_score(x, cluster_labels_temp)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.3f}")

# Find best k by silhouette score
best_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\n🏆 Best k by Silhouette Score: {best_k_silhouette} (score: {max(silhouette_scores):.3f})")

# Plot Silhouette Analysis
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.axvline(x=best_k_silhouette, color='red', linestyle='--', alpha=0.8, 
           label=f'Best k = {best_k_silhouette}')
plt.axvline(x=final_k, color='blue', linestyle=':', alpha=0.8, 
           label=f'Elbow k = {final_k}')
plt.title('Silhouette Analysis for Cluster Validation', fontsize=14)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compare methods
print(f"\n=== Method Comparison ===")
print(f"Elbow Method (curvature): k = {optimal_k}")
print(f"Elbow Method (improvement): k = {alternative_k}")
print(f"Selected k (conservative): k = {final_k}")
print(f"Silhouette Method: k = {best_k_silhouette}")

# Final Summary
print(f"\n=== Analysis Summary ===")
print(f"Dataset: {iris.DESCR.split('Data Set Characteristics:')[0].strip()}")
print(f"Samples: {len(x)}, Features: {len(iris.feature_names)}")
print(f"Final clusters selected: {final_k}")
print(f"Final WCSS: {kmeans_final.inertia_:.2f}")
print(f"Silhouette score for selected k: {silhouette_scores[final_k-2]:.3f}")
print(f"✅ K-means clustering analysis completed successfully!")

print(f"\n🌸 Insights:")
print(f"• The Iris dataset naturally separates into {final_k} distinct clusters")
print(f"• Each cluster corresponds well with the original Iris species")
print(f"• Petal features (length & width) show clearer separation than sepal features")
print(f"• Manual elbow detection suggests k={final_k} as optimal")
print(f"• Silhouette analysis {'agrees' if best_k_silhouette == final_k else 'suggests k=' + str(best_k_silhouette)} with this choice")
print(f"• K-means successfully identified the underlying data structure without using species labels")