import re
import numpy as np

import numpy as np
import re
import string
import random

class Tokenizer:
  def __init__(self, min_occur=10):
    self.word_to_token = {}
    self.token_to_word = {}
    self.word_count = {}

    self.word_to_token['<unk>'] = 0
    self.token_to_word[0] = '<unk>'
    self.vocab_size = 1

    self.min_occur = min_occur

  def fit(self, corpus):
    for review in corpus:
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
      for word in words:
          if word not in self.word_count:
              self.word_count[word] = 0
          self.word_count[word] += 1

    for review in corpus:
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
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
  def __init__(self, min_occur=10):
    self.tokenizer = Tokenizer(min_occur)

  def fit(self, corpus):
    self.tokenizer.fit(corpus)

  def transform(self, corpus):
    n = len(corpus)
    X = np.zeros((n, self.tokenizer.vocab_size))
    for i, review in enumerate(corpus):
      review = review.strip().lower()
      words = re.findall(r"[\w']+|[.,!?;]", review)
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

# Create Term Document matrix
train_reviews = None # Get from Saumya's code
vectorizer = CountVectorizer()
vectorizer.fit(train_reviews)
td_matrix = vectorizer.transform(train_reviews).T
print(f"TD matrix is {td_matrix.shape[0]} x {td_matrix.shape[1]}")

# Convert to TF-IDF matrix
td_matrix_tfidf = transform_tfidf(td_matrix)