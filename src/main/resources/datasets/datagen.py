import numpy as np
import matplotlib.pyplot as plt


def generate_dbscan_test_data(n_points=500):
    np.random.seed(42)
    # Cluster 1
    c1 = np.random.normal(loc=0.5, scale=0.02, size=(n_points // 2, 2))
    # Cluster 2
    c2 = np.random.normal(loc=2.0, scale=0.02, size=(n_points // 2, 2))

    data = np.vstack([c1, c2])
    return data


points = generate_dbscan_test_data()

# Save to CSV for your Java program
np.savetxt("test.csv", points, delimiter=",", fmt="%f")

plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6)
plt.title("Synthetic Dataset: 2 Clusters (eps=0.1, minPts=3)")
plt.show()
