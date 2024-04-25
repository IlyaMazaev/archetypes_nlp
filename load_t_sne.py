import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import json


filename = "subset_indices.json"
with open(filename, 'r') as json_file:
    subset_indices = json.load(json_file)


# load the t-SNE data from the CSV file whenever needed
loaded_tsne_result = np.loadtxt('tsne_data_combined.csv', delimiter=',')

plt.figure(figsize=(10, 6))
plt.scatter(loaded_tsne_result[:, 0], loaded_tsne_result[:, 1], s=1, alpha=0.5, marker='.', color='blue')
for i in tqdm(subset_indices):
    # If the index is in subset_indices, plot the point in red
    plt.scatter(loaded_tsne_result[i, 0], loaded_tsne_result[i, 1], s=1, alpha=0.5, marker='.', color='red')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Scatter Plot of t-SNE Visualization')
plt.grid(True)
plt.show()
