import time
import tkinter as tk
from cmath import inf
from datetime import datetime
from statistics import variance, mean
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from clustering import find_clusters_c_means
from clustering import find_clusters_k_means, print_plot
from clusters_number_optimization import find_optimal_cluster_number
from dunn_index import find_distance
from principal_component_analysis import get_scores_pca
from spotify import get_spotipy, load_config, get_features_for_playlist

global algorithm, playlist_combobox, clusters, epsilon, iterations, m, min_prob, plot_type, x_var, y_var, \
    new_playlist_name, add_playlist_button, playlists, playlists_id, min_variance, selected_value, i, \
    spoti_playlists, user_config, spotipy_instance, df, scores_pca, n_comps, df_x, track_info, df_seg_pca, \
    plot_canvas, canvas_window, plot_ax, selected_cluster, clusters_number, min_tol, max_iter, m_value, prob_value, \
    dunn_value, coefficient_value, entropy_value, silhouette_value, original_df, bootstrap_iter, bootstrap_ind, min_var


def create_plot(window):
    global df_seg_pca, canvas_window, plot_canvas, plot_ax
    canvas_window = window
    label = ttk.Label(window, text="Wykres")
    label.pack(fill=tk.X, padx=5, pady=5)

    # the figure that will contain the plot
    fig = Figure(figsize=(5, 5), dpi=100)

    # list of squares
    y = [i ** 2 for i in range(-100, 101)]

    # adding the subplot
    plot_ax = fig.add_subplot(111)

    # plotting the graph
    # plot_ax.plot(y)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    plot_canvas = FigureCanvasTkAgg(fig, master=window)
    plot_canvas.draw()

    # placing the canvas on the Tkinter window
    plot_canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(plot_canvas, window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    plot_canvas.get_tk_widget().pack()


def create_textbox(text, container):
    label = ttk.Label(container, text=text)
    label.pack(fill=tk.X, padx=5, pady=5)
    text = tk.StringVar()
    textbox = ttk.Entry(container, textvariable=text)
    return textbox


def add_playlist():
    playlists.append(new_playlist_name.get())
    new_playlist_name.delete(0, len(new_playlist_name.get()))
    playlist_combobox.configure(values=playlists)
    playlist_combobox.current(len(playlists) - 1)


def remove_from_playlist():
    playlists.remove(playlist_combobox.get())
    playlist_combobox.configure(values=playlists)
    playlist_combobox.current([0])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_dataframe():
    global df, spotipy_instance
    if playlist_combobox.get() in df and len(df[playlist_combobox.get()]['playlist']) > 0 and \
            df[playlist_combobox.get()]['playlist'][0] == playlist_combobox.get():
        return
    spotipy_instance = get_spotipy()
    df[playlist_combobox.get()] = pd.DataFrame(
        columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])
    df[playlist_combobox.get()] = get_features_for_playlist(df[playlist_combobox.get()], user_config['username'],
                                                            ('spotify:playlist:' + playlists_id[
                                                                playlists.index(playlist_combobox.get())]),
                                                            spotipy_instance)
    print(df[playlist_combobox.get()])


def perform_clusterization_with_indices():
    set_fields()
    return perform_clusterization(True)


def perform_clusterization(calculate_indices):
    global scores_pca, n_comps, df_x, track_info, df_seg_pca, df, epsilon, iterations, min_variance, \
        clusters_number, m, min_prob, min_tol, max_iter, m_value, prob_value, coefficient_value, entropy_value, \
        silhouette_value
    print(str(datetime.now()) + " Starting clustering...")
    if algorithm.get() == "Algorytm K-Średnich":
        df_seg_pca, centers, u, t = find_clusters_k_means(clusters_number, scores_pca, df_x, n_comps, max_iter, min_tol,
                                                          calculate_indices)
    else:
        df_seg_pca, centers, u, t = find_clusters_c_means(scores_pca, clusters_number, m_value, min_tol, max_iter, df_x,
                                                          n_comps, prob_value, calculate_indices)

    coords_assign_pca = df_seg_pca.values[:, 9:len(df_seg_pca.values[0])]
    values_to_serialize = [df_seg_pca, centers, u, coords_assign_pca, t]
    if calculate_indices:
        dunn_value.config(text=str(df_seg_pca['Dunn'][0]))
        coefficient_value.config(text=str(df_seg_pca['Coefficient'][0]))
        entropy_value.config(text=str(df_seg_pca['Entropy'][0]))
        silhouette_value.config(text=str(df_seg_pca['Silhouette'][0]))

        with open('E:/Studia/Magisterka/Magisterka/Aplikacja/LurkaMusicClustering/pomiary/list-'
                  + playlist_combobox.get().replace("/", "-") + ' alg-' + algorithm.get()
                  + ' k' + str(clusters_number) + ' var' + str(min_var).replace(".", ",") + ' iter' + str(max_iter)
                  + ' e' + str(min_tol).replace(".", ",") + ' m' + str(m_value).replace(".", ",") + ' prob'
                  + str(prob_value).replace(".", ",") + ' rep0' + ' '
                  + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + '.txt', 'w') as f:
            f.write("Dunn Index Value: " + str(df_seg_pca['Dunn'][0]))
            f.write("\nSilhouette Index Value: " + str(df_seg_pca['Silhouette'][0]))
            f.write("\nCoefficient Value: " + str(df_seg_pca['Coefficient'][0]))
            f.write("\nEntropy Value: " + str(df_seg_pca['Entropy'][0]))

    print_playlist_features()
    return values_to_serialize


def find_closest_index(center, centers, already_chosen):
    min_dist = inf
    min_dist_index = -1
    for iterator in range(len(centers)):
        if iterator in already_chosen:
            continue
        curr_dist = find_distance(centers[iterator], center)
        if curr_dist < min_dist:
            min_dist_index = iterator
        min_dist = min(curr_dist, min_dist)
    return min_dist_index


def bootstrap():
    global bootstrap_iter, original_df, dunn_value, clusters_number, min_var, max_iter, min_tol, m_value, prob_value, \
        algorithm

    original_df = df[playlist_combobox.get()].copy()
    set_fields()
    centers_array = []
    values_array = []
    times_array = []

    for iterator in range(int(bootstrap_iter.get())):
        for o in range(len(original_df)):
            df[playlist_combobox.get()] = original_df.sample(n=len(original_df), replace=True)
        print("Bootstrap iteration no: " + str(iterator))
        values = perform_clusterization(False)
        times_array.append(values[4])
        values_array.append(list(values))

        if iterator == 0:
            centers_array.append(list(values[1]))
        else:
            temp_centers = []
            centers = (values[1])
            already_chosen = []
            for centroid in centers_array[0]:
                closest_index = find_closest_index(centroid, centers, already_chosen)
                temp_centers.append(centers[closest_index])
                already_chosen.append(closest_index)
            centers_array.append(temp_centers)
        df[playlist_combobox.get()] = original_df.copy()

    # centra
    cluster_variances = []
    for j in range(len(centers_array[0])):
        # wymiary
        dimension_variances = []
        for k in range(len(centers_array[0][0])):
            # iteracje
            dimension_values = []
            for n in range(len(centers_array)):
                dimension_values.append(centers_array[n][j][k])
            dimension_variance = variance(dimension_values)
            dimension_variances.append(dimension_variance)
        cluster_variances.append(dimension_variances)

    cluster_variances_string = str(cluster_variances).replace("],", "],\n")
    cluster_variances_string += "\n"

    with open('E:/Studia/Magisterka/Magisterka/Aplikacja/LurkaMusicClustering/pomiary/list-'
              + playlist_combobox.get().replace("/", "-") + ' alg-' + algorithm.get()
              + ' k' + str(clusters_number) + ' var' + str(min_var).replace(".", ",") + ' iter' + str(max_iter)
              + ' e' + str(min_tol).replace(".", ",") + ' m' + str(m_value).replace(".", ",") + ' prob'
              + str(prob_value).replace(".", ",") + ' rep' + bootstrap_iter.get() + ' '
              + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + '.txt', 'w') as f:
        f.write(cluster_variances_string)
        f.write("Avg Time: " + str(mean(times_array)))


def set_fields():
    global scores_pca, n_comps, df_x, track_info, max_iter, min_tol, clusters_number, m_value, prob_value, df, epsilon\
        , iterations, min_variance, min_var
    min_var = 0.8
    if is_number(min_variance.get()):
        min_var = float(min_variance.get())
    if playlist_combobox.get() not in df:
        get_dataframe()
    scores_pca, n_comps, df_x, track_info = get_scores_pca(df[playlist_combobox.get()], min_var)
    max_iter = 300
    if is_number(iterations.get()):
        max_iter = int(iterations.get())
    min_tol = 1e-4
    if is_number(epsilon.get()):
        min_tol = float(epsilon.get())
    clusters_number = 7
    if is_number(clusters.get()):
        clusters_number = int(clusters.get())
    m_value = 2
    if is_number(m.get()):
        m_value = float(m.get())
    prob_value = 0
    if is_number(min_prob.get()):
        prob_value = float(min_prob.get())


# bind the selected value changes
def print_playlist_features():
    global df_seg_pca, plot_canvas, plot_ax, selected_cluster
    values = []
    for i in range(0, clusters_number):
        values.append(str(i))
    values.append("Wszystkie")
    selected_cluster['values'] = values

    plot_ax.clear()
    print_plot(df_seg_pca, str.lower(x_var.get()), str.lower(y_var.get()), plot_ax, selected_cluster.get())
    plot_canvas.draw()


def create_playlist_combobox(text, container):
    global playlists, playlist_combobox, new_playlist_name, add_playlist_button, selected_value, user_config, \
        spotipy_instance, clusters_number

    label = ttk.Label(container, text=text)
    label.pack(fill=tk.X, padx=5, pady=5)

    playlist_combobox = create_combobox(label, container, playlists, False)
    playlist_combobox['width'] = 60
    playlist_combobox.grid(column=0, row=0)

    return playlist_combobox


def create_combobox(text, container, values, showlabel):
    global selected_value, i
    if showlabel:
        label = ttk.Label(container, text=text)
        label.pack(fill=tk.X, padx=5, pady=5)
    selected_value.append(tk.StringVar())
    combobox = ttk.Combobox(container, textvariable=selected_value[i])
    combobox['values'] = values
    combobox['state'] = 'readonly'
    combobox.current([0])
    i += 1
    return combobox


def create_frame(container, height, width):
    frame = ttk.Frame(container)
    frame.pack_propagate(0)
    frame.configure(width=width, height=height)
    return frame


def optimize_clusters():
    global clusters, scores_pca
    clusters.delete(0, len(clusters.get()))
    clusters.insert(0, str(find_optimal_cluster_number(scores_pca, 30)))


def create_parameters_frame(container, height, width):
    global algorithm, clusters, epsilon, iterations, m, min_prob, min_variance, playlist_combobox, dunn_value, \
        coefficient_value, entropy_value, silhouette_value, bootstrap_iter, bootstrap_ind
    frame = create_frame(container, height, width)

    algorithm = create_combobox("Wybierz algorytm klasteryzacji:", frame,
                                ["Algorytm K-Średnich", "Algorytm C-Średnich"], True)
    algorithm.pack(fill=tk.X, padx=5, pady=5)

    playlist_combobox = create_playlist_combobox("Wybierz playlistę:", frame)
    playlist_combobox.pack(fill=tk.X, padx=5, pady=5)

    clusters = create_textbox("Liczba skupień", frame)
    clusters.pack(fill=tk.X, padx=5, pady=5)

    min_variance = create_textbox("Minimalna wariancja zachowana po przeprowadzeniu PCA", frame)
    min_variance.pack(fill=tk.X, padx=5, pady=5)

    iterations = create_textbox("Iteracje", frame)
    iterations.pack(fill=tk.X, padx=5, pady=5)

    epsilon = create_textbox("Epsilon", frame)
    epsilon.pack(fill=tk.X, padx=5, pady=5)

    lf_c = ttk.LabelFrame(frame, text='Tylko C-Średnich')
    lf_c.pack(fill=tk.X, padx=5, pady=5)

    m = create_textbox("m", lf_c)
    m.pack(fill=tk.X, padx=5, pady=5)

    min_prob = create_textbox("Minimalne prawdopodobieństwo przynależności", lf_c)
    min_prob.pack(fill=tk.X, padx=5, pady=5)

    lf_indices = ttk.LabelFrame(frame, text='Indeksy oceny')
    lf_indices.pack(fill=tk.X, padx=5, pady=5)

    lf_indices.columnconfigure(0, weight=1)
    lf_indices.columnconfigure(1, weight=1)
    lf_indices.columnconfigure(2, weight=1)
    lf_indices.columnconfigure(3, weight=1)
    lf_indices.rowconfigure(0, weight=1)
    lf_indices.rowconfigure(1, weight=1)

    dunn_label = ttk.Label(lf_indices, text="Indeks Dunna")
    dunn_label.grid(column=0, row=0)

    dunn_value = ttk.Label(lf_indices, text="...")
    dunn_value.grid(column=0, row=1)

    coefficient_label = ttk.Label(lf_indices, text="Wskaźnik podziału")
    coefficient_label.grid(column=1, row=0)

    coefficient_value = ttk.Label(lf_indices, text="...")
    coefficient_value.grid(column=1, row=1)

    entropy_label = ttk.Label(lf_indices, text="Wskaźnik entropii")
    entropy_label.grid(column=2, row=0)

    entropy_value = ttk.Label(lf_indices, text="...")
    entropy_value.grid(column=2, row=1)

    silhouette_label = ttk.Label(lf_indices, text="Indeks sylwetki")
    silhouette_label.grid(column=3, row=0)

    silhouette_value = ttk.Label(lf_indices, text="...")
    silhouette_value.grid(column=3, row=1)

    bootstrap_frame = ttk.LabelFrame(frame, text='Bootstrap')
    bootstrap_frame.pack_propagate(0)
    bootstrap_frame.columnconfigure(0, weight=1)
    bootstrap_frame.columnconfigure(1, weight=1)

    bootstrap_iter = create_textbox("Ilość iteracji", bootstrap_frame)
    bootstrap_iter.grid(column=0, row=0)

    perform_bootstrap_button = ttk.Button(
        bootstrap_frame,
        text="Bootstrap",
        command=bootstrap
    )
    perform_bootstrap_button.grid(column=1, row=0)

    bootstrap_frame.pack(fill=tk.X, padx=5, pady=5)

    button_frame = ttk.Frame(frame)

    button_frame.pack_propagate(0)
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)
    button_frame.columnconfigure(2, weight=1)

    load_playlist_button = ttk.Button(
        button_frame,
        text="Wczytaj Playlistę",
        command=get_dataframe
    )
    load_playlist_button.grid(column=0, row=0)

    perform_clusterization_button = ttk.Button(
        button_frame,
        text="Klasteryzuj",
        command=perform_clusterization_with_indices
    )
    perform_clusterization_button.grid(column=1, row=0)

    optimize_clusters_button = ttk.Button(
        button_frame,
        text="Optymalizuj klastry",
        command=optimize_clusters
    )
    optimize_clusters_button.grid(column=2, row=0)

    button_frame.pack(fill=tk.X, padx=5, pady=5)

    return frame


def refresh_plot(event):
    plot_ax.clear()
    print_plot(df_seg_pca, str.lower(x_var.get()), str.lower(y_var.get()), plot_ax, selected_cluster.get())
    # plot_canvas = FigureCanvasTkAgg(fig, master=canvas_window)
    plot_canvas.draw()


def create_plot_frame(container, height, width):
    global plot_type, x_var, y_var, plot_canvas, selected_cluster
    frame = create_frame(container, height, width)
    plot_type = create_combobox("Wybierz wykres:", frame, ["Rozkładu zmiennych", "Radar graph"], True)
    plot_type.pack(fill=tk.X, padx=5, pady=5)
    x_var = create_combobox("Wybierz zmienną x wykresu:", frame,
                            ["Acousticness", "Danceability", "Energy", "Instrumentalness", "Liveness", "Loudness",
                             "Speechiness", "Valence", "Tempo"], True)
    x_var.bind('<<ComboboxSelected>>', refresh_plot)
    x_var.pack(fill=tk.X, padx=5, pady=5)
    y_var = create_combobox("Wybierz zmienną y wykresu:", frame,
                            ["Acousticness", "Danceability", "Energy", "Instrumentalness", "Liveness", "Loudness",
                             "Speechiness", "Valence", "Tempo"], True)
    y_var.bind('<<ComboboxSelected>>', refresh_plot)
    y_var.pack(fill=tk.X, padx=5, pady=5)
    selected_cluster = create_combobox("Wybierz klaster:", frame, "Wszystkie", True)
    selected_cluster.bind('<<ComboboxSelected>>', refresh_plot)
    selected_cluster.pack(fill=tk.X, padx=5, pady=5)

    create_plot(frame)

    return frame


def display_window():
    global selected_value, i, playlists, playlists_id, spoti_playlists, user_config, spotipy_instance, df
    i = 0
    selected_value = []
    df = dict()

    spotipy_instance = get_spotipy()
    user_config = load_config()
    spoti_playlists = spotipy_instance.user_playlists(user_config['username'])

    playlists = []
    playlists_id = []
    for playlist in spoti_playlists['items']:
        playlists.append(playlist['name'])
        playlists_id.append(playlist['id'])

    root = tk.Tk()

    root.title('Paweł Lurka - C-means and K-means Spotify music clustering')

    window_width = 1280
    window_height = 860
    min_width = 640
    min_height = 480
    max_width = 1920
    max_height = 1080

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 4)
    center_y = int(screen_height / 2 - window_height / 2.5)

    root.minsize(min_width, min_height)
    root.maxsize(max_width, max_height)

    # set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    # set icon
    root.iconbitmap('./cluster_icon.ico')

    # set parameters and plot frames
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    parameters_frame = create_parameters_frame(root, window_height, (window_width / 2))
    parameters_frame.grid(column=0, row=0)

    plot_frame = create_plot_frame(root, window_height, (window_width / 2))
    plot_frame.grid(column=1, row=0)

    set_fields()

    # keep the window displaying
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()
