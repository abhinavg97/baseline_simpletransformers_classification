import seaborn as sns
import torch

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def plot_heatmap(data, vmin=-1., vmax=1., save_name='heatmap.pdf'):
    """ Plots a heatmap.

    :param data: DataFrame or ndarray.
    :param vmin:
    :param vmax:
    :param save_name:
    """

    fig, ax = plt.subplots()
    plt.figure(figsize=(11.7, 8.27))

    sns.color_palette("viridis", as_cmap=True)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, annot=False, fmt="f", linewidths=.5, cbar=True, cmap='viridis_r')
    plt.savefig(save_name)


x_gcn = torch.load('X_gcn.pt')
x_glove = torch.load('X_glove.pt')


words = {
    'tent': 808,
    'water': 141,
    'wifi': 2385,
    'medicine': 873,
    'nepal': 6,
    'italy': 8,
    'earthquake': 2,
    'need': 61,
    'available': 1050,
    'availability': 13987,
    'recharge': 6415,
    'rieti': 660,
    'amatrice': 37,
    'monument': 5990,
    'everest': 176,
    'parbat': 16076,
    'iaf': 626,
    'relief': 21
}

glove_tensor = []
gcn_tensor = []

for key in sorted(words):
    glove_tensor += [x_glove[words[key]].tolist()]
    gcn_tensor += [x_gcn[words[key]].tolist()]

gcn_sim = cosine_similarity(gcn_tensor)
gcn_normed = (gcn_sim - gcn_sim.mean(axis=0)) / gcn_sim.std(axis=0)

glove_sim = cosine_similarity(glove_tensor)
glove_normed = (glove_sim - glove_sim.mean(axis=0)) / glove_sim.std(axis=0)


torch.save(torch.Tensor(gcn_sim), 'gcn_cosine_sim.pt')
torch.save(torch.Tensor(glove_sim), 'glove_cosine_sim.pt')

print(glove_normed)


index = sorted(words)
columns = sorted(words)

# data = pd.DataFrame(gcn_normed, index=index, columns=columns)
# plot_heatmap(data=data, save_name='gcn_heatmap.pdf')


data1 = pd.DataFrame(glove_normed, index=index, columns=columns)
plot_heatmap(data=data1, save_name='glove_heatmap.pdf')
