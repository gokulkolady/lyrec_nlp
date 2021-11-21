# generate dataset (train, test, val)
# randomly get artistID, get all songs from artistID

import lyricsgenius
import random
import re

#genius = lyricsgenius.Genius("uZp3-3BY12KCvrTaSmi3Gv9EuTEAp-t4X4QOZ1OJbzWgVZakFrP4GF0Vsj0cz_Lu")
#artist = genius.search_artist("Kendrick Lamar", max_songs=1, sort="title")
#print(artist.songs[0].lyrics.type)

class GeniusAPI:
	token = ""
	genius = None

	def __init__(self, token):
		self.token = token
		self.genius = lyricsgenius.Genius(token)

	def get_songs_from_artist(self, artist_id):
		songs = []
		page_offset = 1

		while page_offset:
			response = self.genius.artist_songs(artist_id, per_page=50, page=page_offset)
			songs.extend(response["songs"])
			page_offset = response["next_page"]
		
		return songs
	
	def process_songs_lyrics(self, songs):
		songsL = [] # list of songs [songname, artist, lyrics]
		
		for s in songs:
			name = s["title"]
			artist = s["artist_names"]
			songId = s["id"]
			lyrics = self.genius.lyrics(songId)
			if lyrics == None:
				continue
			lyrics = lyrics[:-28] # remove 5EmbedShare URLCopyEmbedCopy from end
			lyrics = lyrics.lower()
			lyrics = lyrics.replace("\n"," ")
			lyrics = lyrics.replace("\\'", "'")
			lyrics = lyrics.replace(".", "")
			lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics) #remove [] () and the contents in between
			song = [name, artist, lyrics]
			songsL.append(song)

		return songsL

	
	def get_random_artists(self):
		max_artist_id = 2961456
		artist_ids = []
		
		random_ids = random.sample(range(2961456), 10)

		for i in random_ids:
			try:
				response = self.genius.artist(i)
				artist_ids.append(i)
			except:
				continue
		
		return artist_ids

	def build_dataset(self):
		print("start")
		artist_ids = self.get_random_artists()
		print("found artist ids")
		dataset = []
		for artist_id in artist_ids:
			print(artist_id)
			songs = self.get_songs_from_artist(artist_id)
			processedSongs = self.process_songs_lyrics(songs)
			dataset.extend(processedSongs)
		return dataset


geniusAPI = GeniusAPI("uZp3-3BY12KCvrTaSmi3Gv9EuTEAp-t4X4QOZ1OJbzWgVZakFrP4GF0Vsj0cz_Lu")
print(geniusAPI.build_dataset()[0:5])
