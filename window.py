import tkinter as tk
import pandas as pd
from tkinter import ttk
from tkinter.messagebox import showinfo
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from k_means_clustering import find_clusters, print_plot
from principal_component_analysis import get_scores_pca
from spotify import get_spotipy, load_config, get_features_for_playlist

global algorithm, playlist_combobox, clusters, epsilon, iterations, m, min_prob, plot_type, x_var, y_var, \
    new_playlist_name, add_playlist_button, playlists, playlists_id, min_variance, selected_value, i,\
    spoti_playlists, user_config, spotipy_instance, df, scores_pca, n_comps, df_x, track_info, df_seg_pca_kmeans, \
    plot_canvas, canvas_window, plot_ax, selected_cluster, clusters_number


def create_plot(window):
    global df_seg_pca_kmeans, canvas_window, plot_canvas, plot_ax
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
    if len(df['playlist']) > 0 and df['playlist'][0] == playlist_combobox.get():
        return
    spotipy_instance = get_spotipy()
    df = pd.DataFrame(
        columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])
    df = get_features_for_playlist(df, user_config['username'],
                                   ('spotify:playlist:' + playlists_id[playlists.index(playlist_combobox.get())]),
                                   spotipy_instance)


def perform_clusterization():
    global scores_pca, n_comps, df_x, track_info, df_seg_pca_kmeans, df, epsilon, iterations, min_variance, clusters_number
    min_var = 0.8
    if is_number(min_variance.get()):
        min_var = float(min_variance.get())
    scores_pca, n_comps, df_x, track_info = get_scores_pca(df, min_var)
    max_iter = 300
    if is_number(iterations.get()):
        max_iter = int(iterations.get())
    min_tol = 1e-4
    if is_number(epsilon.get()):
        min_tol = float(epsilon.get())
    clusters_number = 7
    if is_number(clusters.get()):
        clusters_number = int(clusters.get())
    if algorithm.get() == "Algorytm K-Średnich":
        df_seg_pca_kmeans = find_clusters(clusters_number, scores_pca, df_x, n_comps, max_iter, min_tol)


def create_playlist_combobox(text, container):
    global playlists, playlist_combobox, new_playlist_name, add_playlist_button, selected_value, user_config, \
        spotipy_instance, clusters_number

    label = ttk.Label(container, text=text)
    label.pack(fill=tk.X, padx=5, pady=5)

    playlist_combobox = create_combobox(label, container, playlists, False)
    playlist_combobox['width'] = 60
    playlist_combobox.grid(column=0, row=0)

    # bind the selected value changes
    def print_playlist_features(event):
        global df, scores_pca, n_comps, df_x, track_info, df_seg_pca_kmeans, plot_canvas, canvas_window, plot_ax, \
            selected_cluster, iterations, min_variance, epsilon
        """ handle the month changed event """
        get_dataframe()
        print(df)
        perform_clusterization()

        values = []
        for i in range(0, clusters_number):
            values.append(str(i))
        values.append("Wszystkie")
        selected_cluster['values'] = values

        plot_ax.clear()
        print_plot(df_seg_pca_kmeans, str.lower(x_var.get()), str.lower(y_var.get()), plot_ax, selected_cluster.get())
        plot_canvas.draw()

    playlist_combobox.bind('<<ComboboxSelected>>', print_playlist_features)

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


def create_parameters_frame(container, height, width):
    global algorithm, clusters, epsilon, iterations, m, min_prob, min_variance, playlist_combobox
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

    lf = ttk.LabelFrame(frame, text='Tylko C-Średnich')
    lf.pack(fill=tk.X, padx=5, pady=5)

    m = create_textbox("m", lf)
    m.pack(fill=tk.X, padx=5, pady=5)

    min_prob = create_textbox("Minimalne prawdopodobieństwo przynależności", lf)
    min_prob.pack(fill=tk.X, padx=5, pady=5)

    return frame


def refresh_plot(event):
    plot_ax.clear()
    print_plot(df_seg_pca_kmeans, str.lower(x_var.get()), str.lower(y_var.get()), plot_ax, selected_cluster.get())
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
    df = pd.DataFrame(
        columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])

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
    window_height = 900
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

    # keep the window displaying
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()
