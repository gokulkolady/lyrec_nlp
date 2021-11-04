import lyricsgenius

genius = lyricsgenius.Genius("uZp3-3BY12KCvrTaSmi3Gv9EuTEAp-t4X4QOZ1OJbzWgVZakFrP4GF0Vsj0cz_Lu")
artist = genius.search_artist("Kendrick Lamar", max_songs=1, sort="title")
print(artist.songs[0].lyrics.type)