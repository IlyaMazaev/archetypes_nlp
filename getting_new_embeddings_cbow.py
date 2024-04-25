import io

import time

start_time = time.time()


def load_corpus(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    documents = []
    for line in fin:
        documents.append(line.split())
    return documents


"""Next, we need save dictionary in file and load it from file"""


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


"""### 1. Load our corpus"""

documents = load_corpus('merged_corpus.txt')

"""For checking, the corpus contains 12692 documents"""

len(documents)

"""### 2. Train word2vec model

You can tunning this model yourself:
- *vector_size*: dimensionality of the word vectors (default 100)
- *window*: maximum distance between the current and predicted word within a sentence (default 5)
- *min_count*: ignores all words with total frequency lower than this (default 5)
- *workers*: use these many worker threads to train the model (default 3)
- *sg*: training algorithm; 1 for skip-gram, otherwise CBOW  (default 0)
- *hs*: if 1, hierarchical softmax will be used for model training; if 0, and negative is non-zero, negative sampling will be used (default 0)
- *negative*: if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20); if set to 0, no negative sampling is used (default 5)
- *max_vocab_size*: limits the RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones; every 10 million word types need about 1GB of RAM; set to None for no limit (default None)
- *epochs*: number of iterations (epochs) over the corpus (default 5)

Read more: https://radimrehurek.com/gensim/models/word2vec.html
"""

# Commented out IPython magic to ensure Python compatibility.

from gensim.models import Word2Vec

dimension = 8
model = Word2Vec(sentences=documents, vector_size=dimension, min_count=1)

dictionary = {key: model.wv[key] for key in model.wv.key_to_index}

"""For checking, the dictionary contains 192881 different words (if min_count = 1)"""

len(dictionary)

"""Using word2vec models, you can find the closest word. For instance, I test the quality of the model on abstract nouns"""

print(model.wv.most_similar('королева'))

"""### 3. Save dictionary in file"""

save_dictionary('merged_lit_dictionary_cbow_8.txt', dictionary, (len(dictionary), dimension))

"""### 4. Check that everything is saved correctly (optional)"""

loaded_dictionary = load_dictionary('merged_lit_dictionary_cbow_8.txt')
print(len(dictionary) == len(loaded_dictionary))

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")