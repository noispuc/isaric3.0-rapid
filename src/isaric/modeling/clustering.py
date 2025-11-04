from sklearn.cluster import KMeans

def apply_kmeans(df, n_clusters=3):
    """
    Description:
        Applies K-Means clustering to group data into clusters.

    Args:
        df (pandas.DataFrame): Input dataset.
        n_clusters (int): Number of clusters to form.

    Returns:
        tuple: (KMeans model, cluster labels as numpy array)
    """
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(df)
    return model, labels
