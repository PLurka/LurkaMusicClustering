import spotipy
import spotipy.util as util
import yaml


# returns sp
def get_spotipy():
    stream = open('config.yaml')
    user_config = yaml.load(stream, Loader=yaml.FullLoader)
    token = util.prompt_for_user_token(user_config['username'],
                                       scope='playlist-read-private',
                                       client_id=user_config['client_id'],
                                       client_secret=user_config['client_secret'],
                                       redirect_uri=user_config['redirect_uri'])
    print(token)
    return spotipy.Spotify(auth=token)


def load_config():
    stream = open('config.yaml')
    user_config = yaml.load(stream, Loader=yaml.FullLoader)
    return user_config


# A function to extract track names and URIs from a playlist
def get_playlist_info(username, playlist_uri, sp):
    # initialize vars
    offset = 0
    tracks, uris, names, artists = [], [], [], []

    # get playlist id and name from URI
    playlist_id = playlist_uri.split(':')[2]
    playlist_name = sp.user_playlist(username, playlist_id)['name']

    # get all tracks in given playlist (max limit is 100 at a time --> use offset)
    while True:
        results = sp.user_playlist_tracks(username, playlist_id, offset=offset)
        tracks += results['items']
        if results['next'] is not None:
            offset += 100
        else:
            break

    # get track metadata
    for track in tracks:
        names.append(track['track']['name'])
        artists.append(track['track']['artists'][0]['name'])
        uris.append(track['track']['uri'])

    return playlist_name, names, artists, uris


# Extract features from each track in a playlist
def get_features_for_playlist(df, username, uri, sp):
    # get all track metadata from given playlist
    playlist_name, names, artists, uris = get_playlist_info(username, uri, sp)

    # iterate through each track to get audio features and save data into dataframe
    for name, artist, track_uri in zip(names, artists, uris):
        # print(json.dumps(track_uri, indent=4))
        # ^ DEBUG STATEMENT ^

        # access audio features for given track URI via spotipy
        audio_features = sp.audio_features(track_uri)

        # get relevant audio features
        feature_subset = [audio_features[0][col] for col in df.columns if
                          col not in ["name", "artist", "track_URI", "playlist"]]

        # compose a row of the dataframe by flattening the list of audio features
        row = [name, artist, track_uri, *feature_subset, playlist_name]
        df.loc[len(df.index)] = row
    return df