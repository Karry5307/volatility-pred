import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# data 表示过去 k 个交易日的统计特征，label 表示未来五个交易日的 log_returns
class DataSet:
	def __init__(self, data: pd.DataFrame, label: pd.Series):
		self.data = data
		self.label = label

# 将股票数据每隔 k 个交易日划分为一个区间，返回一个包含所有区间的列表，每个区间是一个 DataFrame
# 最后一个区间如果不是 k 个交易日则舍弃。
# 如果最后一个区间不足 5 个交易日，那么倒数第二个区间也舍弃，保证每个区间至少能看得到未来 5 个交易日的数据
# 第一个数据段的前几个值不存在滑动窗口导致 NaN 的情况，此时舍弃该段
def partition(df, k: int, lookahead: int = 5):
	result = []
	for i in range(0, len(df), k):
		if i + k <= len(df) and i + k + lookahead <= len(df):
			data = df.iloc[i : i + k].reset_index(drop=True)
			label = df["log_return"].iloc[i + k : i + k + lookahead].reset_index(drop=True)
			if not data.isna().any().any():
				result.append(DataSet(data, label))
	return result

# 对所有的数据进行一个聚类，看看能不能找到一些有意义的模式
# 聚类结果好像不是很好。
def KMeansCluster(datasets: list, clusterCount: int = 5):
	X = []
	for dataset in datasets:
		features = dataset.data[["rmean", "rstd", "rskew", "rkurt", "rvol", "pkvol", "gkvol"]].values.flatten()
		X.append(features)	
	X = pd.DataFrame(X)
	kmeans = KMeans(n_clusters=clusterCount, random_state=0).fit(X)
	return kmeans.labels_

# 使用 K-Medoids 聚类算法，通过如下方式定义距离函数：
# g(x) = sgn(x) * |x|^p
# d(x, y) = sqrt(sum_i (g(x_i) - g(y_i))^2)
def KMedoidsCluster(datasets: list, clusterCount: int = 5, p: float = 0.2, maxIter: int = 100):
	X = []
	for dataset in datasets:
		features = dataset.data[["log_return", "rmean", "rstd", "rvol", "gkvol"]].values.flatten()
		X.append(features)
	X = np.asarray(X, dtype=float)

	sampleCount = X.shape[0]
	n_clusters = min(clusterCount, sampleCount)

	Xg = np.sign(X) * (np.abs(X) ** p)
	diff = Xg[:, np.newaxis, :] - Xg[np.newaxis, :, :]
	D = np.sqrt(np.sum(diff * diff, axis=2))

	medoids = np.arange(clusterCount)
	for _ in range(maxIter):
		labels = np.argmin(D[:, medoids], axis=1)
		newMedoids = medoids.copy()
		for c in range(n_clusters):
			members = np.where(labels == c)[0]
			if len(members) == 0:
				nonMedoids = np.setdiff1d(np.arange(sampleCount), newMedoids)
				if len(nonMedoids) > 0:
					distToMedoids = np.min(D[nonMedoids][:, newMedoids], axis=1)
					newMedoids[c] = nonMedoids[np.argmax(distToMedoids)]
				continue
			intra = D[np.ix_(members, members)]
			bestId = np.argmin(np.sum(intra, axis=1))
			newMedoids[c] = members[bestId]
		if np.array_equal(newMedoids, medoids):
			break
		medoids = newMedoids

	labels = np.argmin(D[:, medoids], axis=1)
	return labels