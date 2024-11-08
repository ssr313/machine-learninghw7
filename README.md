# MNIST 数据集的 PCA 和 UMAP 分析

## 概述
本项目旨在使用 PCA 和 UMAP 对 MNIST 数据集中的手写数字图像进行降维分析。通过这些技术，我们可以探索数据的内在结构，并可视化不同数字的分布情况。

## 补充文件
### IDX文件
- `train-images-idx3-ubyte/train-images.idx3-ubyte`：包含训练图像数据的IDX文件。
- `train-labels-idx1-ubyte/train-labels.idx1-ubyte`：包含训练标签数据的IDX文件。

## 编程环境
### Python库
- `idx2numpy`：用于将IDX文件转换为NumPy数组。
- `numpy`：用于数值计算。
- `matplotlib.pyplot`：用于数据可视化。
- `sklearn.decomposition`：用于执行主成分分析（PCA）。
- `umap`：用于执行Uniform Manifold Approximation and Projection（UMAP）降维。
- `os`：用于设置环境变量。

## 实验中使用的参数

### PCA相关参数
- `apply_pca_and_plot_variance`：用于计算和绘制PCA的累积解释方差。
- `plot_pca_components`：用于绘制PCA投影，参数`components`指定了要绘制的主成分对，默认为(1, 2)和(1, 3)。
- `plot_pca_one_digit`：用于绘制特定数字（例如1）的PCA投影，并显示代表性数据点的原始图像。

### UMAP相关参数
- `apply_umap_and_plot`：用于应用UMAP降维并绘制结果。
  - `n_neighbors`：邻域点的数量，影响局部结构的捕捉，默认为15。
  - `min_dist`：嵌入点之间的最小距离，影响点的聚集程度，默认为0.1。
  - `spread`：嵌入点的有效尺度，影响点的相对距离，默认为1.0。

### UMAP参数调整
在主函数中，对`spread`、`min_dist`和`n_neighbors`进行了一系列的调整，以探索不同参数对UMAP结果的影响。
- `spread`值：0.5
