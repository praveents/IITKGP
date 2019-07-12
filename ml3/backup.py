
wcss = []

#run KMeans
for nrClusters in range(2, 9):
      id_n=nrClusters
      kmeans = KMeans(n_clusters=id_n, random_state=0).fit(dataset1_standardized)
      id_label=kmeans.labels_
      wcss.append(kmeans.inertia_)
      print(id_label)
      #plot result

plt.plot(range(1, 8), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()