import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# Load data from idx files
def load_data(images_path, labels_path):
    images = idx2numpy.convert_from_file(images_path)
    labels = idx2numpy.convert_from_file(labels_path)
    return images.reshape(images.shape[0], -1), labels  # Flatten images


# Apply PCA and plot explained variance
def apply_pca_and_plot_variance(data):
    pca = PCA()
    pca.fit(data)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(explained_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.title('PCA Explained Variance')
    plt.show()
    return pca


# Plot PCA projections
def plot_pca_components(data, labels, pca, components=(1, 2, 3)):
    transformed_data = pca.transform(data)
    plt.figure(figsize=(8, 5))
    plt.scatter(transformed_data[:, components[0] - 1], transformed_data[:, components[1] - 1], c=labels, cmap='tab10',
                s=1)
    plt.xlabel(f'PC{components[0]}')
    plt.ylabel(f'PC{components[1]}')
    plt.colorbar()
    plt.title(f'PCA - PC{components[0]} vs PC{components[1]}')
    plt.show()


# Plot PCA for one digit
def plot_pca_one_digit(data, labels, digit, pca):
    # 筛选指定数字的数据
    digit_data = data[labels == digit]
    transformed_data = pca.transform(digit_data)

    # 计算到中心的距离
    distances = np.linalg.norm(transformed_data - np.mean(transformed_data, axis=0), axis=1)

    # 选择距离中心最近和最远的两个点
    indices = np.argsort(distances)

    plt.figure(figsize=(8, 5))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='blue', s=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA of Digit {digit} - PC1 vs PC2')

    # 标记代表性点（集中位置和偏远位置）
    plt.scatter(transformed_data[indices[0], 0], transformed_data[indices[0], 1], c='green', s=50, marker='x')
    plt.scatter(transformed_data[indices[1], 0], transformed_data[indices[1], 1], c='green', s=50, marker='x')
    plt.scatter(transformed_data[indices[-1], 0], transformed_data[indices[-1], 1], c='red', s=50, marker='x')
    plt.scatter(transformed_data[indices[-2], 0], transformed_data[indices[-2], 1], c='red', s=50, marker='x')

    plt.show()

    # 显示代表性数据点的原始图像（假设 digit_data 是展平的 28x28 图像）
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(digit_data[indices[0]].reshape(28, 28), cmap='gray')
    axes[0].set_title('Close Point 1')
    axes[1].imshow(digit_data[indices[1]].reshape(28, 28), cmap='gray')
    axes[1].set_title('Close Point 2')
    axes[2].imshow(digit_data[indices[-1]].reshape(28, 28), cmap='gray')
    axes[2].set_title('Far Point 1')
    axes[3].imshow(digit_data[indices[-2]].reshape(28, 28), cmap='gray')
    axes[3].set_title('Far Point 2')
    for ax in axes:
        ax.axis('off')
    plt.show()


# Apply UMAP and visualize
def apply_umap_and_plot(data, labels, n_neighbors=15, min_dist=0.1, spread=1.0):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread)
    embedding = reducer.fit_transform(data)
    plt.figure(figsize=(8, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=1)
    plt.colorbar()
    plt.title(f'UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}')
    plt.show()


# Main execution
if __name__ == '__main__':
    images_path = 'train-images-idx3-ubyte/train-images.idx3-ubyte'
    labels_path = 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'

    # Load the dataset
    data, labels = load_data(images_path, labels_path)

    # 1. PCA and explained variance
    pca = apply_pca_and_plot_variance(data)

    # Plot first PC1 vs PC2 and PC1 vs PC3
    plot_pca_components(data, labels, pca, components=(1, 2))
    plot_pca_components(data, labels, pca, components=(1, 3))

    # 2. PCA for one digit (e.g., digit 0)
    plot_pca_one_digit(data, labels, digit=1, pca=pca)

    # 3. UMAP on original data and first few PCs
    apply_umap_and_plot(data, labels, n_neighbors=15, min_dist=0.1, spread=1.0)
    # Optionally, try UMAP on PCA-transformed data (e.g., first 20 PCs)
    pca_transformed_data = pca.transform(data)[:, :20]
    #n_neighbors=10,15,20,25
    #min_dist=0.05,0.1,0.5,1
    #spread = 0.5, 1.0, 1.5, 2.0
    # apply_umap_and_plot(pca_transformed_data, labels, n_neighbors=15, min_dist=0.1, spread=1.0)
    for i in [0.5,1.0,1.5,2.0]:
        apply_umap_and_plot(pca_transformed_data, labels, n_neighbors=15, min_dist=0.1, spread=i)
    for i in [0.05,0.1,0.5,1]:
        apply_umap_and_plot(pca_transformed_data, labels, n_neighbors=15, min_dist=i, spread=1.0)
    for i in [10,15,20,25]:
        apply_umap_and_plot(pca_transformed_data, labels, n_neighbors=i, min_dist=0.1, spread=1.0)
