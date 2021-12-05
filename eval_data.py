
import lyricsgenius
import random
import re
import spotify_api
import pickle


def save_playlist_data(playlist_id, filename):
	data = spotify_api.get_playlist_tracks(playlist_id)

	with open(filename, "wb") as f:
		if data != []:
			pickle.dump(data, f)


