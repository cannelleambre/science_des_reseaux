import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from haversine import haversine, Unit
import math

def read_csv(file_path):
    """Read the CSV file and return a DataFrame."""
    return pd.read_csv(file_path)

def haversine_distance(coord1, coord2):
    """Calculate the Haversine distance between two coordinates."""
    return haversine(coord1, coord2, unit=Unit.KILOMETERS)

def custom_metric(x, y):
    """Custom distance metric combining Haversine and Euclidean distances."""
    # Calculate Haversine distance for geographical coordinates
    coord_dist = haversine_distance((x[0], x[1]), (y[0], y[1]))
    # Calculate Euclidean distance for other features
    feature_dist = np.linalg.norm(x[2:] - y[2:])
    # Combine the distances (you can adjust the weights as needed)
    return coord_dist + feature_dist

def create_ball_tree(data):
    """Create a Ball Tree from the data using a custom distance metric."""
    # Extract the features for clustering
    coordinates = data[['LAT', 'LON']].values
    features = data[['PIR', 'CIR', 'SERVICE']].values

    # Combine coordinates and features
    combined_features = np.hstack((coordinates, features))

    # Create the Ball Tree with the custom metric
    tree = BallTree(combined_features, leaf_size=40, metric=custom_metric)
    return tree, combined_features

def find_clusters(tree, features, max_diameter=90):
    """Find clusters using the Ball Tree with a maximum diameter."""
    # Query the tree to find neighbors within the maximum diameter
    distances, indices = tree.query_radius(features, r=max_diameter / 2, return_distance=True)
    return distances, indices

def write_clusters_to_csv(data, indices, output_file):
    """Write the clusters to a CSV file."""
    # Create a list to store the cluster information
    cluster_data = []

    # Assign cluster IDs
    cluster_id = 0
    cluster_map = {}

    for i, ind in enumerate(indices):
        for j in ind:
            if j not in cluster_map:
                cluster_map[j] = cluster_id
                cluster_id += 1

    # Create a DataFrame to store the cluster information
    for i, ind in enumerate(indices):
        for j in ind:
            cluster_data.append({
                'Point_Index': i,
                'Cluster_ID': cluster_map[j],
                'LON': data.at[i, 'LON'],
                'LAT': data.at[i, 'LAT'],
                'PIR': data.at[i, 'PIR'],
                'CIR': data.at[i, 'CIR'],
                'SERVICE': data.at[i, 'SERVICE']
            })

    # Convert the list to a DataFrame
    cluster_df = pd.DataFrame(cluster_data)

    # Write the DataFrame to a CSV file
    cluster_df.to_csv(output_file, index=False)

def main(file_path, output_file):
    # Read the CSV file
    data = read_csv(file_path)

    # Create the Ball Tree
    tree, features = create_ball_tree(data)

    # Find clusters
    distances, indices = find_clusters(tree, features)

    # Write clusters to a CSV file
    write_clusters_to_csv(data, indices, output_file)

if __name__ == "__main__":
    # Specify the path to your input CSV file and the output CSV file
    input_file_path = 'generated.csv'
    output_file_path = 'clusters.csv'
    main(input_file_path, output_file_path)
