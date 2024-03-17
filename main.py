# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# Read csv file into a pandas dataframe
fpath = 'smoking_drinking_dataset_Ver01.csv'
df = pd.read_csv(fpath)

# Create a copy of our data frame to work with
data = df.copy()

# Print summary statistics
print('The dataset has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
print(data.describe())

print(data.head())
data.dtypes.value_counts()
print(data.columns)
data.dtypes
numeric_attributes = data.columns[data.dtypes != "object"]
categorical_attributes = data.columns[data.dtypes == "object"]

print(numeric_attributes)
print(categorical_attributes)
# Print the total number of missing values
print("There are {} missing values in this dataset".format(data.isnull().sum().sum()))

print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))

print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))

x_cols = ['sex', 'age', 'height', 'weight', 'waistline', 'sight_left',
       'sight_right', 'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS',
       'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride', 'hemoglobin',
       'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT',
       'gamma_GTP', 'SMK_stat_type_cd', 'DRK_YN']
df1 = df[x_cols]

x_cats = ['sex', 'weight']
df2 = pd.get_dummies(df1, columns=x_cats)

# Sample data to reduce computational load
df2 = df2.sample(2000)

# Encode 'DRK_YN' column
df2['DRK_YN_encoded'] = df2['DRK_YN'].map({'N': 0, 'Y': 1})

# Drop the original 'DRK_YN' column
df2.drop(['DRK_YN'], axis=1, inplace=True)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(df2)

# Perform KMeans clustering with different numbers of clusters
numClusters = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SSE = []
for k in numClusters:
    k_means = cluster.KMeans(n_clusters=k, n_init=10)
    k_means.fit(X)
    SSE.append(k_means.inertia_)

# Plot SSE vs Number of Clusters
plt.plot(numClusters, SSE)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')

# Perform dimensionality reduction using t-SNE
pca = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
X2 = pca.fit_transform(X)

# Perform clustering with different numbers of clusters
range_n_clusters = [3, 4, 5]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Plot clusters in 2D
    colors = cm.nipy_spectral(df2['DRK_YN_encoded'].astype(float) / 2)
    ax2.scatter(
        X2[:, 0], X2[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    ax2.set_title("The visualization of the clustered data")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()
