import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

hidden_df = pd.read_csv("hidden.csv")
label_df = pd.read_csv("label_list.csv")
tags_df = pd.read_csv("tags.csv", names = ['track_id', 'genre', 'unk', 'unk2'])
hidden_df = hidden_df.sort_values(by='id')

hidden_df['user_idx'] = hidden_df['id']
label_df['track_id'] = label_df['first(track_id, false)']
label_df = label_df.drop(['Unnamed: 0', 'first(track_id, false)'], axis=1)
tags_df = tags_df.drop(['unk', 'unk2'], axis=1)
print(hidden_df.columns)
print(label_df.columns)
print(tags_df.columns)

df_subset = pd.DataFrame()
df_subset['user_idx'] = hidden_df['user_idx']
df_subset['features'] = hidden_df['features']

df_subset = pd.merge(df_subset, label_df, on='user_idx', how='left').dropna(axis=0)
df_subset = pd.merge(df_subset, tags_df, on='track_id', how='left').dropna(axis=0)
df_subset['y'] = [s.lower() for s in df_subset['genre']]

df_group = df_subset.groupby('y')
count_df = df_group.count().sort_values(by='user_idx', ascending=False)
count_df['count'] = count_df['user_idx']
count_df = count_df.reset_index()
top = 10
count_df = count_df.iloc[:top, :]
count_df = count_df.drop(['user_idx', 'features', 'track_id', 'genre'], axis=1)
count_df.columns = ['genre', 'count']

df_subset = pd.merge(df_subset, count_df, on='genre', how='left').dropna(axis=0)
df_subset = df_subset.groupby('user_idx').first().reset_index()
df_subset = df_subset.sample(n=1000)
features = np.array(df_subset['features'])
X = np.zeros((features.shape[0], 15))
for i in range(features.shape[0]):
    X[i] = np.array([float(item) for item in features[i][1:-1].split(", ")])

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=5000)
tsne_results = tsne.fit_transform(X)

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="genre",
    data=df_subset,
    legend="full",
    alpha=0.7
)
plt.savefig("TSNE")
plt.show()