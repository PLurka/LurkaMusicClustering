from sklearn.cluster import KMeans
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


def find_clusters(n_clusters, scores_pca, df_x, n_comps, max_iter, epsilon):
    kmeans_pca = KMeans(n_clusters=n_clusters, init='random', max_iter=max_iter, algorithm='full', tol=epsilon,
                        n_init=1)
    kmeans_pca.fit(scores_pca)
    df_seg_pca_kmeans = pd.concat([df_x.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_seg_pca_kmeans.columns.values[(-1 * n_comps):] = ["Component " + str(i + 1) for i in range(n_comps)]
    df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_
    print(df_seg_pca_kmeans.head())
    return df_seg_pca_kmeans


def print_plot(df_seg_pca_kmeans, component_no_1, component_no_2, plot_ax, selected_cluster):
    x = df_seg_pca_kmeans[component_no_1]
    y = df_seg_pca_kmeans[component_no_2]
    cluster_assignment = df_seg_pca_kmeans['Cluster']
    # predicate, iterable
    if selected_cluster != "Wszystkie":
        fitting_indexes = []
        j = 0
        for i in cluster_assignment:
            if i == int(selected_cluster):
                fitting_indexes.append(j)
            j += 1
        x = x[fitting_indexes]
        y = y[fitting_indexes]
        cluster_assignment = cluster_assignment[fitting_indexes]
    # fig = plt.figure(figsize=(10, 8))
    # axs = fig.subplots()
    # sns.scatterplot(x, y, hue=df_seg_pca_kmeans['Cluster'],
    #                 palette=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'goldenrod', 'tab:cyan'])
    scatter = plot_ax.scatter(x, y, c=cluster_assignment, s=10, cmap='gnuplot')
    handles, labels = scatter.legend_elements(alpha=0.6)
    legend2 = plot_ax.legend(handles, labels, loc="upper right", title="Klastry")

    # plt.title('Clusters by PCA Components', fontsize=20)
    # plt.xlabel(component_no_1, fontsize=18)
    # plt.ylabel(component_no_2, fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.show()
    # fig.savefig("./visualizations/clusters-2d.png")
    return plot_ax
