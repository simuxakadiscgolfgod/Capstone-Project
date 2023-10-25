# based on: https://github.com/gboeing/2014-summer-travels/blob/master/clustering-scikitlearn.ipynb

import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
%matplotlib inline

# define the number of kilometers in one radian
kms_per_radian = 6371.0088
# load the data set
df = pd.read_csv('/content/accident_info_fixed.csv', encoding='utf-8')

# analyzing only cases with fatalities & removing outliers
df = df[df['zuvusiuSkaicius'] > 0]
df = df[df['ilguma'] < 35]

# represent points consistently
coord_df = df[['platuma', 'ilguma']]
coords = coord_df.to_numpy()

# define epsilon as 1 kilometer, converted to radians for use by haversine
epsilon = 1 / kms_per_radian

start_time = time.time()
db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_

# get the number of clusters
num_clusters = len(set(cluster_labels))

# all done, print the outcome
message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
print(message.format(len(df), num_clusters, 100*(1 - float(num_clusters) / len(df)), time.time()-start_time))
print('Silhouette coefficient: {:0.03f}'.format(metrics.silhouette_score(coords, cluster_labels)))

# turn the clusters in to a pandas series, where each element is a cluster of points
clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])

clusters.drop(clusters.tail(1).index, inplace=True) # drop last n rows / somehow last row results in an empty list and prompts error later

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

centermost_points = clusters.map(get_centermost_point)
# unzip the list of centermost points (lat, lon) tuples into separate lat and lon lists
lats, lons = zip(*centermost_points)

# from these lats/lons create a new df of one representative point for each cluster
rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
rep_points.tail()

# plot the final reduced set of coordinate points vs the original full set
fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rep_points['lon'], rep_points['lat'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(df['ilguma'], df['platuma'], c='k', alpha=0.9, s=3)
ax.set_title('DBSCAN clustering; min_samples=5, max_dist=1km')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(20.8, 26.8)
ax.set_ylim(53.9, 56.5)
ax.legend([df_scatter, rs_scatter], ['Fatalities', 'Clusters'], loc='upper right')
plt.show()
