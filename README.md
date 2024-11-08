# MNIST 数据集的 PCA 和 UMAP 分析

## 项目概述
本项目旨在使用主成分分析（PCA）和均匀流形近似与投影（UMAP）对 MNIST 数据集中的手写数字图像进行降维分析。通过这些技术，我们可以探索数据的内在结构，并可视化不同数字的分布情况。

## 环境设置
确保您的环境中安装了以下 Python 库：
- `idx2numpy`：用于将 IDX 文件转换为 NumPy 数组。
- `numpy`：用于数值计算。
- `matplotlib.pyplot`：用于数据可视化。
- `sklearn.decomposition`：用于执行主成分分析（PCA）。
- `umap`：用于执行 Uniform Manifold Approximation and Projection（UMAP）降维。
- `os`：用于设置环境变量。

## 数据文件
### IDX 文件
- `train-images-idx3-ubyte/train-images.idx3-ubyte`：包含训练图像数据的 IDX 文件。
- `train-labels-idx1-ubyte/train-labels.idx1-ubyte`：包含训练标签数据的 IDX 文件。

## 参数设置
### PCA 参数
- `apply_pca_and_plot_variance`：用于计算和绘制 PCA 的累积解释方差。
- `plot_pca_components`：用于绘制 PCA 投影，参数 `components` 指定了要绘制的主成分对，默认为 (1, 2) 和 (1, 3)。
- `plot_pca_one_digit`：用于绘制特定数字（例如 1）的 PCA 投影，并显示代表性数据点的原始图像。

### UMAP 参数
- `apply_umap_and_plot`：用于应用 UMAP 降维并绘制结果。
  - `n_neighbors`：邻域点的数量，影响局部结构的捕捉，默认为 15。
  - `min_dist`：嵌入点之间的最小距离，影响点的聚集程度，默认为 0.1。
  - `spread`：嵌入点的有效尺度，影响点的相对距离，默认为 1.0。

### UMAP 参数调整
在主函数中，对 `spread`、`min_dist` 和 `n_neighbors` 进行了一系列的调整，以探索不同参数对 UMAP 结果的影响：
- `spread` 值：0.5, 1.0, 1.5, 2.0。
- `min_dist` 值：0.05, 0.1, 0.5, 1。
- `n_neighbors` 值：10, 15, 20, 25。

### 环境变量设置
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
