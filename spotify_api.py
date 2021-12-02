import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "45cfe204e32d42e89a050a1eec8bf1f9"
client_secret = "cbfcace375974373b858b231a990c01e"
redirect_uri = "http://localhost:8080"

auth_manager = SpotifyClientCredentials(client_id= client_id, 
                    client_secret = client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_song_attributes(data, feature, print_info = False):
    updated_data = []
    for track_name, artist, lyrics in data:
        q = "track:" + track_name + " artist:" + artist
        track_results = sp.search(q, limit=1, offset=0, type='track,artist', market=None)
        for _, t in enumerate(track_results['tracks']['items']):
            response_artist = t['artists'][0]['name']
            response_track = t['name']
            track_id = t['uri']
            if print_info: 
                print (response_artist)
                print (response_track)
                print (track_id)
                print ("--------------")
            audio_feature = sp.audio_features(tracks=t['uri'])
            updated_data.append([track_name, artist, lyrics, audio_feature[0][feature], t['uri']])
    return updated_data


def get_playlist(playlist_name):
    playlist_id = []
    q = "playlist:" + playlist_name
    playlist_results = sp.search(q, limit=1, offset=0, type='playlist', market=None)
    for _, t in enumerate(playlist_results['playlists']['items']):
        print (t)
        print (t["name"])
        playlist_id.append(t["uri"])
    return playlist_id




playlist_id = get_playlist("RapCaviar")

print ("playlist id: ", playlist_id[0])

playlist_info = sp.playlist(playlist_id[0], fields="tracks", market=None, additional_types=('track', ))

print (playlist_info)


for _, t in enumerate(playlist_info['tracks']['items']):
    print (t)
    print (t["name"])
    # print (t[""])

featured = sp.featured_playlists()

for _, t in enumerate(featured['playlists']['items']):
    print (t["name"])
    print (t["uri"])

data = [["8TEEN", "Khalid", "I still live with my parents"], 
        ["The Way Life Goes", "Lil Uzi Vert", "I like that girl too much"],
        ["Trying", "midwxst", "Back to the basics"]]

desired_feature = "valence"
print_info = True
updated_data = get_song_attributes(data, desired_feature, False)

print (updated_data)


