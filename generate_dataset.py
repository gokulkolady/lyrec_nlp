# generate dataset (train, test, val)
# randomly get artistID, get all songs from artistID

import lyricsgenius
import random
import re
import spotify_api
import pickle

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
            while True:
                try:
                    response = self.genius.artist_songs(artist_id, per_page=50, page=page_offset)
                    songs.extend(response["songs"])
                    page_offset = response["next_page"]
                    break
                except:
                    pass
        
        return songs
    
    def process_songs_lyrics(self, songs):
        songsL = [] # list of songs [songname, artist, lyrics]
        
        for s in songs:
            name = s["title"]
            artist = s["artist_names"]
            songId = s["id"]
            while True:
                try:
                    lyrics = self.genius.lyrics(songId)
                    break
                except:
                    pass
            if lyrics == None:
                continue
            lyrics = lyrics[:-28] # remove 5EmbedShare URLCopyEmbedCopy from end
            lyrics = lyrics.lower()
            lyrics = lyrics.replace("\n"," ")
            lyrics = lyrics.replace("\\'", "'")
            lyrics = lyrics.replace("\'", "'")
            lyrics = lyrics.replace(".", "")
            lyrics = lyrics.replace(",", "")
            lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics) #remove [] () and the contents in between
            song = [name, artist, lyrics]
            song_with_valence = spotify_api.get_song_attributes([song], "valence")
            if song_with_valence != []:
                songsL.append(song_with_valence)

        return songsL

    
    def get_random_artists(self):
        max_artist_id = 2961456
        artist_ids = set()
        
        random_ids = random.sample(range(2961456), 50)

        for i in random_ids:
            try:
                response = self.genius.artist(i)
                artist_ids.add(i)
            except:
                continue
        
        return artist_ids

    def build_dataset(self):
        print("start")
        all_artist_ids = set()
        print("found artist ids")
        dataset = []
        multiple = 1
        while len(dataset) < 10000:
            print("len dataset")
            print(len(dataset))
            artist_random = self.get_random_artists()
            artist_ids = artist_random.difference(all_artist_ids)
            all_artist_ids.update(artist_random)

            for artist_id in artist_ids:
                print(artist_id)
                songs = self.get_songs_from_artist(artist_id)
                processedSongs = self.process_songs_lyrics(songs)
                dataset.extend(processedSongs)
            
            if len(dataset)/25 > multiple:
                with open("20000_song_dataset.pkl", "wb") as f:
                    if dataset != []:
                        pickle.dump(dataset, f)
                multiple += 1
            

        return dataset


geniusAPI = GeniusAPI("uZp3-3BY12KCvrTaSmi3Gv9EuTEAp-t4X4QOZ1OJbzWgVZakFrP4GF0Vsj0cz_Lu")
dataset_100 = geniusAPI.build_dataset()

with open("20000_song_dataset.pkl", "wb") as f:
    if dataset_100 != []:
        pickle.dump(dataset_100, f)

print(dataset_100[0:5])
