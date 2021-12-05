import re
import numpy as np

import numpy as np
import re
import string
import random
import pickle

MIN_OCCUR = 50

class Tokenizer:
  def __init__(self, min_occur=MIN_OCCUR):
    self.word_to_token = {}
    self.token_to_word = {}
    self.word_count = {}

    self.word_to_token['<unk>'] = 0
    self.token_to_word[0] = '<unk>'
    self.vocab_size = 1

    self.min_occur = min_occur

  def fit(self, corpus):
    for review in corpus:
      words = review.split()
      for word in words:
          if word not in self.word_count:
              self.word_count[word] = 0
          self.word_count[word] += 1

    for review in corpus:
      words = review.split()
      for word in words:
        if self.word_count[word] < self.min_occur:
          continue
        if word in self.word_to_token:
          continue
        self.word_to_token[word] = self.vocab_size
        self.token_to_word[self.vocab_size] = word
        self.vocab_size += 1

  def tokenize(self, corpus):
    tokenized_corpus = []
    for review in corpus:
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
      tokenized_review = []
      for word in words:
        if word not in self.word_to_token:
          tokenized_review.append(0)
        else:
          tokenized_review.append(self.word_to_token[word])
      tokenized_corpus.append(tokenized_review)
    return tokenized_corpus

  def de_tokenize(self, tokenized_corpus):
    corpus = []
    for tokenized_review in tokenized_corpus:
      review = []
      for token in tokenized_review:
        review.append(self.token_to_word[token])
      corpus.append(" ".join(review))
    return corpus


class CountVectorizer:
  def __init__(self, min_occur=MIN_OCCUR):
    self.tokenizer = Tokenizer(min_occur)

  def fit(self, corpus):
    self.tokenizer.fit(corpus)

  def transform(self, corpus):
    n = len(corpus)
    X = np.zeros((n, self.tokenizer.vocab_size))
    for i, review in enumerate(corpus):
      words = review.split()
      for word in words:
        if word not in self.tokenizer.word_count or self.tokenizer.word_count[word] < self.tokenizer.min_occur:
          X[i][0] += 1
        else:
          X[i][self.tokenizer.word_to_token[word]] += 1
    return X

def get_ngrams(tokenized_corpus, window_size, pad_idx=2006):
    ngrams = []
    for i, review in enumerate(tokenized_corpus):
        for j, word in enumerate(review):
            min_ind = max(0, j-window_size)
            max_ind = min(len(review), j+window_size+1)
            ctx = np.zeros(2 * window_size, dtype=np.int64) + pad_idx
            for ik, k in enumerate(range(min_ind, j)):
                ctx[ik] = review[k]
            for ik, k in enumerate(range(j+1, max_ind)):
                ctx[window_size+ik] = review[k]
            ngrams.append((ctx, review[j]))
    return ngrams

def transform_tfidf(matrix):
    # `matrix` is a `|V| x |D|` TD matrix of raw counts, where `|V|` is the 
    # vocabulary size and `|D|` is the number of documents in the corpus. This
    # function should return a version of `matrix` with the TF-IDF transform
    # applied. Note: this function should be nondestructive: it should not
    # modify the input; instead, it should return a new object.
    return matrix * np.log(matrix.shape[1]/np.count_nonzero(matrix, axis=1, keepdims=True))



# Load data for tfidf processing from pickle
all_data = []

with open('1000_song_dataset.pkl', 'rb') as f:
    all_data = pickle.load(f)

with open('10000_song_dataset.pkl', 'rb') as f:
    all_data += pickle.load(f)

all_data = [internal[0] for internal in all_data]
lyric_data = [internal[2] for internal in all_data]

# Create Term Document matrix
vectorizer = CountVectorizer()
vectorizer.fit(lyric_data)
td_matrix = vectorizer.transform(lyric_data).T
print(f"TD matrix is {td_matrix.shape[0]} x {td_matrix.shape[1]}")

# Convert to TF-IDF matrix
td_matrix_tfidf = transform_tfidf(td_matrix)
# for c in range(td_matrix.shape[1]):
#   print(td_matrix_tfidf[:,c].tolist())

# Append tfidf vectors to all_data
for c in range(len(all_data)):
  all_data[c].append(td_matrix_tfidf[:,c].tolist())

# Save updated all_data to pickle file
with open('1000_song_tfidf_dataset.pkl', 'wb') as f2:
  pickle.dump(all_data, f2)