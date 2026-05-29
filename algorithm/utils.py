import numpy as np
import pandas as pd

# data 表示过去 k 个交易日的统计特征，label 表示未来五个交易日的 log_returns 的方均根
class DataSet:
	def __init__(self, data: pd.DataFrame, label: float):
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
			futureReturn = df["log_return"].iloc[i + k:i + k + lookahead].reset_index(drop=True)
			label = (futureReturn ** 2).mean() ** 0.5
			if not data.isna().any().any():
				result.append(DataSet(data, label))
	return result

# 将一个 DataSet 列表转换成能够丢给 sklearn 训练的特征矩阵和标签向量
def extract(dataset: list, featureIndex: list):
	X, y = [], []
	for item in dataset:
		features = item.data[featureIndex].values.flatten()
		X.append(features)
		y.append(item.label)
	return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

def calcError(yTrue: np.ndarray, yPred: np.ndarray, metric: str = "rmse"):
	metric = metric.lower()
	if metric == "rmse":
		return float(np.sqrt(np.mean((yTrue - yPred) ** 2)))
	if metric == "mae":
		return float(np.mean(np.abs(yTrue - yPred)))
	raise ValueError("Metric 不支持！")