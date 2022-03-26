import time
from datetime import datetime

import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans

from bezdek_index import calc_bezdek_index
from dunn_index import calc_dunn_index
from silhouette_index import calc_sil_index


def create_df_seg_pca(df_x, n_comps, scores_pca):
    df_seg_pca_kmeans = pd.concat([df_x.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_seg_pca_kmeans.columns.values[(-1 * n_comps):] = ["Component " + str(i + 1) for i in range(n_comps)]
    return df_seg_pca_kmeans


def find_clusters_k_means(n_clusters, scores_pca, df_x, n_comps, max_iter, epsilon, calculate_indices):
    kmeans_pca = KMeans(n_clusters=n_clusters, init='random', max_iter=max_iter, algorithm='full', tol=epsilon,
                        n_init=1)
    start_time = time.time()
    kmeans_pca.fit(scores_pca)
    end_time = time.time()
    print(str(datetime.now()) + " Done clustering!")
    df_seg_pca = create_df_seg_pca(df_x, n_comps, scores_pca)
    cluster_affiliation = []
    for i in kmeans_pca.labels_:
        cluster_affiliation.append([i])
    df_seg_pca['Cluster'] = cluster_affiliation

    u = [[0 for i in range(len(cluster_affiliation))] for j in range(n_clusters)]
    for i in range(len(cluster_affiliation)):
        u[cluster_affiliation[i][0]][i] = 1

    if calculate_indices:
        coords_assign_pca = df_seg_pca.values[:, 9:len(df_seg_pca.values[0])]
        coords_assign_pca_for_sil = coords_assign_pca.copy()
        df_seg_pca['Dunn'] = calc_dunn_index(coords_assign_pca, kmeans_pca.cluster_centers_, 0, 0)
        print(str(datetime.now()) + " Done Dunn Index!")
        df_seg_pca['Coefficient'], df_seg_pca['Entropy'] = 1, 0
        print(str(datetime.now()) + " Done entropy and coefficient value!")
        df_seg_pca['Silhouette'] = calc_sil_index(coords_assign_pca_for_sil, u)
        print(str(datetime.now()) + " Done Silhouette Index!")

    return df_seg_pca, kmeans_pca.cluster_centers_, u, end_time - start_time


def find_clusters_c_means(scores_pca, centers, m, epsilon, max_iter, df_x, n_comps, min_prob, calculate_indices):
    start_time = time.time()
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        scores_pca.T, centers, m, error=epsilon, maxiter=max_iter)
    end_time = time.time()
    print(str(datetime.now()) + " Done clustering!")
    df_seg_pca = pd.concat([df_x.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_seg_pca.columns.values[(-1 * n_comps):] = ["Component " + str(i + 1) for i in range(n_comps)]
    cluster_affiliation = []
    for i in range(len(scores_pca)):
        centers_array = []
        if min_prob == 0:
            local_max_prob = 0
            affiliated_cluster = 0
            for j in range(centers):
                local_max_prob = max(local_max_prob, u[j][i])
                if local_max_prob == u[j][i]:
                    affiliated_cluster = j
            centers_array.append(affiliated_cluster)
        else:
            for j in range(centers):
                if u[j][i] >= min_prob:
                    centers_array.append(j)
        cluster_affiliation.append(centers_array)

    df_seg_pca['Cluster'] = cluster_affiliation
    if calculate_indices:
        coords_assign_pca = df_seg_pca.values[:, 9:len(df_seg_pca.values[0])]
        coords_assign_pca_for_sil = coords_assign_pca.copy()
        df_seg_pca['Dunn'] = calc_dunn_index(coords_assign_pca, cntr, 0, 0)
        print(str(datetime.now()) + " Done Dunn Index!")
        df_seg_pca['Coefficient'], df_seg_pca['Entropy'] = calc_bezdek_index(u, None)
        print(str(datetime.now()) + " Done entropy and coefficient value!")
        df_seg_pca['Silhouette'] = calc_sil_index(coords_assign_pca_for_sil, u)
        print(str(datetime.now()) + " Done Silhouette Index!")

    return df_seg_pca, cntr, u, end_time - start_time


def print_plot(df_seg_pca_kmeans, component_no_1, component_no_2, plot_ax, selected_cluster):
    x = df_seg_pca_kmeans[component_no_1]
    y = df_seg_pca_kmeans[component_no_2]
    cluster_assignment = df_seg_pca_kmeans['Cluster']

    if selected_cluster != "Wszystkie":
        fitting_indexes = []
        j = 0
        for i in cluster_assignment:
            for k in i:
                if k == int(selected_cluster):
                    fitting_indexes.append(j)
            j += 1
        x = x[fitting_indexes]
        y = y[fitting_indexes]
        cluster_assignment = cluster_assignment[fitting_indexes]

    color_names_array = []
    j = 0
    for i in cluster_assignment:
        if len(i) != 0:
            color = ""
            if len(i) > 1:
                for k in i:
                    if color == "":
                        color += str(k)
                    else:
                        color += ", "
                        color += str(k)
                color_names_array.append(color)
            else:
                color_names_array.append(str(i[0]))
        else:
            x = x.drop(j)
            y = y.drop(j)
            cluster_assignment = cluster_assignment.drop(j)
        j += 1

    for i in range(len(x) - len(color_names_array)):
        color_names_array.append(-1)

    j = 0
    color_array = []
    used_clusters = set()
    for i in color_names_array:
        if i not in used_clusters:
            color_array.append(j)
            used_clusters.add(i)
            j += 1
        else:
            color_array.append(color_array[list(color_names_array).index(i)])

    scatter = plot_ax.scatter(x, y, c=color_array, s=10, cmap='gnuplot')
    handles, labels = scatter.legend_elements(alpha=0.6)
    labels_int = [int(''.join(i for i in x if i.isdigit())) for x in labels]
    true_labels = []
    for i in labels_int:
        true_labels.append(color_names_array[color_array.index(i)])
    plot_ax.legend(handles, true_labels, loc="upper right", title="Klastry")

    return plot_ax
