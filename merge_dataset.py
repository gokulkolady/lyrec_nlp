import pickle
import itertools

all_data = []

with open('data/1000_song_dataset.pkl', 'rb') as f:
    all_data = pickle.load(f)

with open('10000_song_dataset.pkl', 'rb') as f:
    all_data += pickle.load(f)


with open('2000_song_dataset.pkl', 'rb') as f:
    all_data += pickle.load(f)

with open('3000_song_dataset.pkl', 'rb') as f:
    all_data += pickle.load(f)

with open('20000_song_dataset.pkl', 'rb') as f:
    all_data += pickle.load(f)


all_data.sort()
print(len(all_data))
unique_data = list(all_data for all_data,_ in itertools.groupby(all_data))

with open('unique_song_dataset.pkl', 'wb') as f:
	pickle.dump(unique_data, f)

print(len(unique_data))
