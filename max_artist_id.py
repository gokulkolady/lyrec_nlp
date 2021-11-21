# find max artist id

import lyricsgenius
genius = lyricsgenius.Genius("uZp3-3BY12KCvrTaSmi3Gv9EuTEAp-t4X4QOZ1OJbzWgVZakFrP4GF0Vsj0cz_Lu")

artist_id = 2961456
response = genius.artist_songs(artist_id, per_page=50, page=1)
print(response["songs"])
max_err = 0
0/0
while artist_id < 2961595:
	try:
		response = genius.artist_songs(artist_id, per_page=50, page=1)
		print(artist_id)
	except:
		max_err += 1
		print(max_err)
	artist_id += 1


max_artist_id = 2961456
