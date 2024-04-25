import io
from tqdm import tqdm
import time
import datetime
import itertools
import json

from sklearn.metrics import calinski_harabasz_score, silhouette_score
import numpy as np
from scipy.special import gamma
from scipy.spatial import cKDTree
from collections import defaultdict

from clusterval import Clusterval
from sklearn.datasets import load_iris, make_blobs

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Wishart:
    clusters_to_objects: defaultdict
    object_labels: np.ndarray
    clusters: np.ndarray
    kd_tree: cKDTree
    internal_distances: defaultdict  # Changed to store distances as dict for precise mapping #

    def fit(self, data):
        self.kd_tree = cKDTree(data=data)
        self.internal_distances = defaultdict(dict)

    def clustering(self, X, wishart_neighbors, significance_level, workers, batch_weight_in_gb=10):
        distances = np.empty(0).ravel()

        batch_size = batch_weight_in_gb * (1024 ** 3) * 8
        batches_count = X.shape[0] // (batch_size // X.shape[1]) if batch_size < X.size else 1

        batches = np.array_split(X, batches_count)
        for batch in batches:
            batch_dists, _ = self.kd_tree.query(x=batch, k=wishart_neighbors + 1, workers=workers)
            batch_dists = batch_dists[:, -1].ravel()
            distances = np.hstack((distances, batch_dists))

        indexes = np.argsort(distances)
        X = X[indexes]

        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype=int) - 1

        # index in tuple
        # min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)

        batches = np.array_split(X, batches_count)
        idx_batches = np.array_split(indexes, batches_count)
        del X, indexes

        for batch, idx_batch in zip(batches, idx_batches):
            _, neighbors = self.kd_tree.query(x=batch, k=wishart_neighbors + 1, workers=workers)
            neighbors = neighbors[:, 1:]

            for real_index, idx in enumerate(idx_batch):
                neighbors_clusters = np.concatenate(
                    [self.object_labels[neighbors[real_index]], self.object_labels[neighbors[real_index]]])
                unique_clusters = np.unique(neighbors_clusters).astype(int)
                unique_clusters = unique_clusters[unique_clusters != -1]

                if len(unique_clusters) == 0:
                    self._create_new_cluster(idx, distances[idx])
                else:
                    max_cluster = unique_clusters[-1]
                    min_cluster = unique_clusters[0]
                    if max_cluster == min_cluster:
                        if self.clusters[max_cluster][-1] < 0.5:
                            self._add_elem_to_exist_cluster(idx, distances[idx], max_cluster)
                        else:
                            self._add_elem_to_noise(idx)
                    else:
                        my_clusters = self.clusters[unique_clusters]
                        flags = my_clusters[:, -1]
                        if np.min(flags) > 0.5:
                            self._add_elem_to_noise(idx)
                        else:
                            significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                            significan *= wishart_neighbors
                            significan /= size
                            significan /= np.power(np.pi, dim / 2)
                            significan *= gamma(dim / 2 + 1)
                            significan_index = significan >= significance_level

                            significan_clusters = unique_clusters[significan_index]
                            not_significan_clusters = unique_clusters[~significan_index]
                            significan_clusters_count = len(significan_clusters)
                            if significan_clusters_count > 1 or min_cluster == 0:
                                self._add_elem_to_noise(idx)
                                self.clusters[significan_clusters, -1] = 1
                                for not_sig_cluster in not_significan_clusters:
                                    if not_sig_cluster == 0:
                                        continue

                                    for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                        self._add_elem_to_noise(bad_index)
                                    self.clusters_to_objects[not_sig_cluster].clear()
                            else:
                                for cur_cluster in unique_clusters:
                                    if cur_cluster == min_cluster:
                                        continue

                                    for bad_index in self.clusters_to_objects[cur_cluster]:
                                        self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                    self.clusters_to_objects[cur_cluster].clear()

                                self._add_elem_to_exist_cluster(idx, distances[idx], min_cluster)

        return self.clean_data()

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq: index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype=int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis=0)

        self.internal_distances[len(self.clusters) - 1] = {index: dist}  # Initialize with first element's distance #

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)

        self.internal_distances[cluster_label][index] = dist

    def compute_average_internal_distance(self):
        if not self.internal_distances:
            return None

        total_distance = 0
        total_count = 0
        for cluster_id, distances_dict in self.internal_distances.items():
            if distances_dict:  # Check if the dictionary is not empty
                cluster_distances = list(distances_dict.values())
                total_distance += sum(cluster_distances)
                total_count += len(cluster_distances)

        if total_count > 0:
            return total_distance / total_count
        else:
            return None

    def compute_average_internal_distance_for_subset(self, subset_indices):
        # print("given subset_indices:", subset_indices)

        distances = []
        for index in subset_indices:
            cluster = self.object_labels[index]
            if cluster >= 0 and index in self.internal_distances[cluster]:
                distances.append(self.internal_distances[cluster][index])

        if distances:
            return np.mean(distances)
        else:
            return None

    def get_clusters_to_indexes(self):
        return dict(self.clusters_to_objects)

    def sort_clusters_by_subset(self, subset_indices):
        cluster_counts = defaultdict(int)

        # Count how many elements from the subset are in each cluster
        for cluster_index, elements in self.clusters_to_objects.items():
            # Use set intersection for efficient counting
            cluster_counts[cluster_index] = len(set(subset_indices) & set(elements))

        # Sort clusters by the count of subset elements in descending order
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_clusters


english_lowercase_letters = set("abcdefghijklmnopqrstuvwxyz0123456789_½¼àαντηáèοόλεó¾ôμὴςіἁπλѣῆі")


def generate_bigrams_itertools(word_embeddings):
    skipped_counter = 0
    words = list(filter(lambda x: not any(char in english_lowercase_letters for char in x), word_embeddings.keys()))

    for bigram in itertools.product(words, repeat=2):
        word1, word2 = bigram

        if any(char in english_lowercase_letters for char in ''.join(bigram)):
            skipped_counter += 1
            if skipped_counter % 1000000 == 0:
                print("skipped", bigram, skipped_counter)
            continue  # Skip english

        embedding1 = word_embeddings[word1]
        embedding2 = word_embeddings[word2]
        bigram_embedding = [sum(pair) for pair in zip(embedding1, embedding2)]
        yield bigram, bigram_embedding


def generate_bigrams(word_embeddings):
    words = list(word_embeddings.keys())
    skipped_counter = 0
    for i in range(len(words) - 1):
        word1 = words[i]
        embedding1 = word_embeddings[word1]

        for j in range(i + 1, len(words)):
            word2 = words[j]
            embedding2 = word_embeddings[word2]

            bigram = (word1, word2)
            if any(char in english_lowercase_letters for char in ''.join(bigram)):
                skipped_counter += 1
                if skipped_counter % 1000000 == 0:
                    print("skipped", bigram, skipped_counter)
                continue  # Skip english
            bigram_embedding = [sum(pair) for pair in zip(embedding1, embedding2)]

            yield bigram, bigram_embedding


def extract_bigrams_from_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # Split the text into words
        words = ' '.join((text.split('\n')[::500])).split()
        # Remove punctuation and convert to lowercase
        words = [word.strip(",.!?") for word in words]
        words = [word.lower() for word in words if word]
        words = list(filter(lambda word: not any(char in english_lowercase_letters for char in word), words))
        # Generate bigrams
        bigrams = [(words[i_word], words[i_word + 1]) for i_word in range(len(words) - 1)]
        return list(set(bigrams))


if __name__ == '__main__':
    start_time = time.time()
    # CLUSTERING ARCHETYPES

    file_path = "merged_corpus.txt"  # Replace this with the path to your text file
    # file_path = "combined_text_lower_normalized.txt"  # Replace this with the path to your text file
    # file_path = "russian_nofraglit_corpus.txt"  # Replace this with the path to your text file
    bigrams = extract_bigrams_from_text_file(file_path)
    print(bigrams[::100])  # Print the first 10 bigrams as an example


    def load_dictionary_cbow(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        length, dimension = map(int, fin.readline().split())
        dictionary = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            dictionary[tokens[0]] = list(map(float, tokens[1:]))
        return dictionary


    dict_from_file = load_dictionary_cbow("merged_lit_dictionary_cbow_8.txt")
    # dict_from_file = dictionary_loaded = np.load('dictionary_svd_8.npy', allow_pickle=True)[()]

    # print(list(dict_from_file.keys())[::100])

    sliced_embeddings = dict()
    for key, val in list(dict_from_file.items()):  # [26400:]:
        sliced_embeddings[key] = dict_from_file[key]  # [-5:]

    # print(list(sliced_embeddings.keys())[::50])
    print("len(bigrams):", len(bigrams))
    print("len(sliced_embeddings):", len(sliced_embeddings))


    # Function to get embeddings for words or placeholders if not found
    def get_embedding(word):
        if word in sliced_embeddings:
            return sliced_embeddings[word]
        else:
            return np.zeros_like(next(iter(sliced_embeddings.values())))


    # loading unique bigrams for A texts made with unique_bigr_of_A.py
    filename = "unique_bigrams_for_A.json"
    with open(filename, 'r') as json_file:
        unique_bigr_of_A = json.load(json_file)

    # Compute embeddings for each bigram
    bigram_embeddings = []
    subset_indices = []
    for bigram in tqdm(bigrams):
        word1_emb = get_embedding(bigram[0])
        word2_emb = get_embedding(bigram[1])
        bigram_emb = np.concatenate([word1_emb, word2_emb])
        bigram_embeddings.append(bigram_emb)

    lookup_dict = defaultdict(dict)
    for i, orig_bigr in enumerate(bigrams):
        lookup_dict['_'.join(orig_bigr)] = i

    subset_indices = []
    for A_bigr in tqdm(unique_bigr_of_A):
        if lookup_dict['_'.join(A_bigr)]:
            subset_indices.append(lookup_dict['_'.join(A_bigr)])

    print(subset_indices)
    print("len(subset_indices) :", len(subset_indices))

    filename = "subset_indices.json"
    # saving the list to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(subset_indices, json_file)

    # Convert list of embeddings to numpy array
    bigram_embeddings = np.array(bigram_embeddings)

    with open('cluster_output_21_04.txt', 'a', encoding='utf-8') as f:
        current_time = datetime.datetime.now()
        print("Current date and time:", current_time)
        print("Current date and time:", current_time, file=f)

    for h in [50000, 100000]:
        for k in [10, 15, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
                  3500, 4000, 4500, 5000]:
            start_time = time.time()
            # Perform clustering using Wishart algorithm
            wishart_clusterer = Wishart()
            wishart_clusterer.fit(bigram_embeddings)
            cluster_labels = wishart_clusterer.clustering(bigram_embeddings, wishart_neighbors=k, significance_level=h,
                                                          workers=-1)
            with open('cluster_output.txt', 'a', encoding='utf-8') as f:
                print("wishart_neighbors :", k, "significance_level : ", h)
                print("wishart_neighbors :", k, "significance_level : ", h, file=f)
                print("Cluster labels:", cluster_labels)
                print("Cluster labels:", cluster_labels, file=f)
                try:
                    ch_score = calinski_harabasz_score(bigram_embeddings, cluster_labels)
                    print("calinski_harabasz_score:", ch_score)
                    print("calinski_harabasz_score:", ch_score, file=f)
                except ValueError as e:
                    print(f"Caught a ValueError in calinski_harabasz_score: {e}")
                    print(f"Caught a ValueError in calinski_harabasz_score: {e}", file=f)

                try:
                    sil_score = silhouette_score(bigram_embeddings, cluster_labels)
                    print("Silhouette Score:", sil_score)
                    print("Silhouette Score:", sil_score, file=f)
                except ValueError as e:
                    print(f"Caught a ValueError in silhouette_score: {e}")
                    print(f"Caught a ValueError in silhouette_score: {e}", file=f)

                average_internal_distance = wishart_clusterer.compute_average_internal_distance()
                print("Average internal cluster distance:", average_internal_distance)
                print("Average internal cluster distance:", average_internal_distance, file=f)

                average_internal_distance_subset = wishart_clusterer.compute_average_internal_distance_for_subset(
                    subset_indices)
                print("Average internal distance for the subset:", average_internal_distance_subset)
                print("Average internal distance for the subset:", average_internal_distance_subset, file=f)

                sorted_indexes = wishart_clusterer.sort_clusters_by_subset(subset_indices)

                print("sorted_indexes:", sorted_indexes)
                print("sorted_indexes:", sorted_indexes, file=f)
                clusters = wishart_clusterer.get_clusters_to_indexes()

                for best_cluster_count in sorted_indexes:
                    try:
                        print(
                            f"idx: {best_cluster_count[0]} | total_bigrams: {len(clusters[best_cluster_count[0]])} | A_bigrams: {best_cluster_count[1]} | ratio: {best_cluster_count[1] / len(clusters[best_cluster_count[0]])}")
                        print(
                            f"idx: {best_cluster_count[0]} | total_bigrams: {len(clusters[best_cluster_count[0]])} | A_bigrams: {best_cluster_count[1]} | ratio: {best_cluster_count[1] / len(clusters[best_cluster_count[0]])}",
                            file=f)
                    except ZeroDivisionError:
                        print(
                            f"idx: {best_cluster_count[0]} | total_bigrams: {len(clusters[best_cluster_count[0]])} | A_bigrams: {best_cluster_count[1]} | ratio: div_zero")
                        print(
                            f"idx: {best_cluster_count[0]} | total_bigrams: {len(clusters[best_cluster_count[0]])} | A_bigrams: {best_cluster_count[1]} | ratio: div_zero",
                            file=f)
                        break
                    print(f"idx {best_cluster_count} : ",
                          list(map(lambda x: bigrams[x], clusters[best_cluster_count[0]])), file=f)
                    if best_cluster_count[1] == 0:
                        break

                execution_time = time.time() - start_time
                print(f"Clustering execution time: {execution_time} seconds")
                print(f"Clustering execution time: {execution_time} seconds", file=f)
                print()
                print("\n", file=f)

    # start_tsne_time = time.time()
    # # TSNE
    # tsne = TSNE(n_jobs=-1, verbose=1)  # TSNE(n_components=2, perplexity=15, learning_rate=10)
    # tsne_result = tsne.fit_transform(bigram_embeddings)
    # np.savetxt('tsne_data_combined.csv', tsne_result, delimiter=',')
    # end_tsne_time = time.time()
    # execution_time = end_tsne_time - start_tsne_time
    # print(f"TSNE execution time: {execution_time} seconds")
    #
    # # load the t-SNE data from the CSV file whenever needed
    # loaded_tsne_result = np.loadtxt('tsne_data_combined.csv', delimiter=',')
    #
    # plt.figure(figsize=(10, 10))
    # plt.scatter(loaded_tsne_result[:, 0], loaded_tsne_result[:, 1], s=1, alpha=0.5, marker='.', color='blue')
    # for i in tqdm(subset_indices):
    #     # If the index is in subset_indices, plot the point in red
    #     plt.scatter(loaded_tsne_result[i, 0], loaded_tsne_result[i, 1], s=1, alpha=0.5, marker='.', color='red')
    #
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.title('Scatter Plot of t-SNE Visualization')
    # plt.grid(True)
    # plt.show()
