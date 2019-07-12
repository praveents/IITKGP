import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, \
    v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import seaborn as sns # More snazzy plotting library

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

np.random.seed(123)

# read the data
music_data = pd.read_csv('default_plus_chromatic_features_1059_tracks_new.csv', header=None)

# details about the data frame
print(music_data.head())
print(music_data.describe())

print('Shape of the data set: ' + str(music_data.shape))

# get features and target
X = music_data.iloc[:, 0:116]
Y = music_data.iloc[:, 116:117]

# save labels as string
Labels = Y
Data = X
Labels_keys = Labels[116].unique().tolist()
Labels = list(Labels[116])
print('Activity labels: ' + str(Labels_keys))

# check for missing values
Temp = pd.DataFrame(Data.isnull().sum())
Temp.columns = ['Sum']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])))

# check the optimal k value
ks = range(1, 60)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k, max_iter=200)
    model.fit(Data)
    inertias.append(model.inertia_)
plt.figure(figsize=(14, 14))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()


def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels
    and clustering performance parameters.

    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels

    Output:
    performance table
    """
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    y_clust = k_means.predict(data_frame)
    # print(ct)
    print('% 9s' % 'SSE   ARI ')
    print('%i   %.3f'
          % (k_means.inertia_,
             adjusted_rand_score(true_labels, y_clust)))


for i in [4, 8, 12, 16, 22, 33, 44, 55]:
    k_means(n_clust=i, data_frame=Data, true_labels=Labels)

# check for optimal number of features
lpca = PCA(n_components=16, random_state=123)
X_pca = lpca.fit_transform(Data)
print("original shape:   ", Data.shape)
print("transformed shape:", X_pca.shape)
features = range(lpca.n_components_)

plt.figure(figsize=(12, 12))
plt.bar(features[:16], lpca.explained_variance_[:16], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:16])
plt.show()

# Set a 16 KMeans clustering
kmeans = KMeans(n_clusters=16)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(X_pca)

# Define our own color map
LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# Plot the scatter digram
plt.figure(figsize=(14, 14))
plt.scatter(X_pca[:, 0], X_pca[:, 1],  alpha=0.5)
plt.show()


def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(Data)
    print('Shape of original Data df: ' + str(Data.shape))
    print('Shape of the new Data df: ' + str(Data_reduced.shape))
    plt.scatter(Data_reduced[:, 0], Data_reduced[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()
    return Data_reduced


for i in [2, 4, 8, 16, 32]:
    data = pca_transform(n_comp=i)
    k_means(n_clust=2, data_frame=data, true_labels=Labels)

print('Done')
