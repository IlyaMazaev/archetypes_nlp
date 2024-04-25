import io
import os

import time

start_time = time.time()


def load_corpus(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    documents = []
    for line in fin:
        documents.append(line.split())
    return documents


def save_dictionary(fname, dictionary, args):
    length, dimension = args
    fin = io.open(fname, 'w', encoding='utf-8')
    fin.write('%d %d\n' % (length, dimension))
    for word in dictionary:
        fin.write('%s %s\n' % (word, ' '.join(map(str, dictionary[word]))))


def load_dictionary(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    length, dimension = map(int, fin.readline().split())
    dictionary = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        dictionary[tokens[0]] = map(float, tokens[1:])
    return dictionary


# SVD
# TFIDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def make_matrix_W_list_of_words(corpus_path, min_df, max_df=None, token_pattern=None, use_idf=True):
    '''
    corpus_path - is a path to the corpus, where one line - one text

    min_df - is the minimum times (or fraction of the texts) a word must occur in the corpus

    max_df - is the maximum times (or fraction of the texts) a word must occur in the corpus
    if it is None, there are no upper bound

    token_pattern - alphabet, which will be considered. Usually can be all letters of the language and numbers
    if None all symbols will be OK

    use_idf - is bool value whether to use idf
    '''
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
        if token_pattern:
            vectorizer = TfidfVectorizer(analyzer='word', min_df=min_df, token_pattern=token_pattern, use_idf=use_idf)
        else:
            vectorizer = TfidfVectorizer(analyzer='word', min_df=min_df, use_idf=use_idf)
        data_vectorized = vectorizer.fit_transform(corpus_file)
    return data_vectorized, vectorizer.get_feature_names_out()


W, words_list = make_matrix_W_list_of_words('merged_corpus.txt', 1)

print(W.shape)

from scipy.sparse.linalg import svds
import numpy as np


def apply_svd(W, k, output_folder):
    '''
    W - matrix texts x words
    k - the rank of the SVD, must be less than any dimension of W
    '''
    # Apply the SVD function
    u, sigma, vt = svds(W, k)

    # The function does not garantee, that the order of the singular values is descending
    # So, we need to dreate it by hand
    descending_order_of_inds = np.flip(np.argsort(sigma))
    u = u[:, descending_order_of_inds]
    vt = vt[descending_order_of_inds]
    sigma = sigma[descending_order_of_inds]

    # Checking that sizes are ok
    assert sigma.shape == (k,)
    assert vt.shape == (k, W.shape[1])
    assert u.shape == (W.shape[0], k)

    # os.makedirs("/content_svd/")

    # Now, we'll save all the matrixes in folder (just in case)
    with open(output_folder + '/' + str(k) + '_sigma_vt.npy', 'wb') as f:
        np.save(f, np.dot(np.diag(sigma), vt).T)
    with open(output_folder + '/' + str(k) + '_sigma.npy', 'wb') as f:
        np.save(f, sigma)
    with open(output_folder + '/' + str(k) + '_u.npy', 'wb') as f:
        np.save(f, u)
    with open(output_folder + '/' + str(k) + '_vt.npy', 'wb') as f:
        np.save(f, vt)
    return np.dot(np.diag(sigma), vt).T


vv = apply_svd(W, 8, '/content_svd')
print(vv.shape)


def create_dictionary(words_list, vv, output_file):
    dictionary = {}
    for word, vector in zip(words_list, vv):
        dictionary[word] = vector
    np.save(output_file, dictionary)
    return dictionary


dictionary = create_dictionary(words_list, vv, 'dictionary_svd_8.npy')

print(dictionary)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
