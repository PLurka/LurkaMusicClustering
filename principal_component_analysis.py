from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_scores_pca(df, min_var):
    non_features = ['name', 'artist', 'track_URI', 'playlist']
    track_info = df[non_features]
    df_x = df.drop(columns=non_features)
    df_x.head()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_x)
    pca = PCA()
    pca.fit(X_std)
    evr = pca.explained_variance_ratio_
    print(evr)
    for i, exp_var in enumerate(evr.cumsum()):
        if exp_var >= min_var:
            n_comps = i + 1
            break
        if min_var >= 1:
            n_comps = 9
    print("Number of components:", n_comps)
    pca = PCA(n_components=n_comps)
    pca.fit(X_std)
    scores_pca = pca.transform(X_std)
    return scores_pca, n_comps, df_x, track_info
