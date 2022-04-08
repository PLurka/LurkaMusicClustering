from statistics import mean
from joblib import Parallel, delayed
from joblib._memmapping_reducer import has_shareable_memory


def calc_dunn_index(points, clusters, linkage, diameter):
    linkage_algorithms = [single_linkage_distance, complete_linkage_distance, average_linkage_distance,
                          centroid_linkage_distance, centroid_average_linkage_distance]
    diameter_algorithms = [complete_diameter_distance, average_diameter_distance, centroid_diameter_distance]
    selected_linkage = linkage_algorithms[linkage]
    selected_diameter = diameter_algorithms[diameter]
    linkage_array = []
    diameter_array = []

    def calculate_distances(c, k):
        if (c == k).all() or len(get_points_in_cluster(points, find_cluster_index(clusters, c))) == 0 \
                or len(get_points_in_cluster(points, find_cluster_index(clusters, k))) == 0:
            return
        linkage_array.append(selected_linkage(get_points_in_cluster(points, find_cluster_index(clusters, c)),
                                              get_points_in_cluster(points, find_cluster_index(clusters, k)), c, k))
        diameter_array.append(selected_diameter(get_points_in_cluster(points, find_cluster_index(clusters, c)), c))

    Parallel(n_jobs=4, require='sharedmem')(
        delayed(has_shareable_memory)(calculate_distances(c, k)) for c in clusters for k in clusters)

    return min(linkage_array) / max(diameter_array)


def find_cluster_index(clusters, cluster):
    j = 0
    for c in clusters:
        if (c == cluster).all():
            return j
        j += 1


# the smallest distance between two observations from two different clusters
def single_linkage_distance(cluster_1_points, cluster_2_points, centroid_1, centroid_2):
    return min(get_all_distances(cluster_1_points, cluster_2_points))


# the largest distance between two observations from two different clusters
def complete_linkage_distance(cluster_1_points, cluster_2_points, centroid_1, centroid_2):
    return max(get_all_distances(cluster_1_points, cluster_2_points))


# the average distance between two observations from two different clusters
def average_linkage_distance(cluster_1_points, cluster_2_points, centroid_1, centroid_2):
    return mean(get_all_distances(cluster_1_points, cluster_2_points))


# the distance between the centroids of two clusters
def centroid_linkage_distance(cluster_1_points, cluster_2_points, centroid_1, centroid_2):
    return find_distance(centroid_1, centroid_2)


# the average distance between a centroid of a cluster and all observations from a different cluster
def centroid_average_linkage_distance(cluster_1_points, cluster_2_points, centroid_1, centroid_2):
    return mean(get_cluster_vs_all_distances(cluster_1_points, centroid_2))


# the distance between two furthest observations within the same cluster:
def complete_diameter_distance(cluster_points, centroid):
    if len(cluster_points) <= 1:
        return 0
    return max(get_all_distances(cluster_points, cluster_points))


# the average distance between all observations within the same cluster:
def average_diameter_distance(cluster_points, centroid):
    if len(cluster_points) == 1:
        return 0
    return mean(get_all_distances(cluster_points, cluster_points))


# the average distance between all observations within the same cluster:
def centroid_diameter_distance(cluster_points, centroid):
    return 2 * mean(get_cluster_vs_all_distances(cluster_points, centroid))


# Function to find euclidean distance between two points
def find_distance(point1, point2):
    euc_dis = 0
    for i in range(len(point1) - 1):
        euc_dis = euc_dis + (point1[i] - point2[i]) ** 2

    return euc_dis ** 0.5


def get_all_distances(cluster_1_points, cluster_2_points):
    distance_array = []
    for i in cluster_1_points:
        for j in cluster_2_points:
            if (i == j).all():
                continue
            distance_array.append(find_distance(i, j))
    return distance_array


def get_cluster_vs_all_distances(cluster_1_points, centroid_2):
    distance_array = []
    for i in cluster_1_points:
        distance_array.append(find_distance(i, centroid_2))
    return distance_array


def get_points_in_cluster(points, cluster):
    cluster_points = []
    for point in points:
        for c in point[len(point) - 1]:
            if c == cluster:
                cluster_points.append(point)

    return cluster_points
