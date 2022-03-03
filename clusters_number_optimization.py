from sklearn.cluster import KMeans
from kneed import KneeLocator


def find_optimal_cluster_number(scores_pca, max_clusters):
    wcss = []
    for i in range(1, max_clusters):
        kmeans_pca = KMeans(i, init='k-means++', random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
    n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
    print("Optimal number of clusters", n_clusters)
    return n_clusters
