import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

def plotStats(df):
	x = range(len(df))

	fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
	axes = axes.ravel()

	subplotConfig = [
		("log_return", "black", "Log Return"),
		(("rmean", "rstd"), ("black", "red"), "Mean / Std"),
		(("rskew", "rkurt"), ("blue", "#ffd700"), "Skew / Kurtosis"),
		(("rvol", "pkvol", "gkvol"), ("red", "#ffd700", "green"), "Volatility Measures"),
	]

	for ax, item in zip(axes, subplotConfig):
		columns, colors, title = item
		if isinstance(columns, str):
			ax.plot(x, df[columns], color=colors, linewidth=1.0)
			ax.set_ylim(-0.1, 0.1)
		else:
			for column, color in zip(columns, colors):
				ax.plot(x, df[column], color=color, linewidth=1.0)

		ax.set_title(title)
		ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.4)
		if isinstance(columns, str):
			ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
		ax.tick_params(axis="x", labelbottom=False, bottom=True)

		ticks = list(range(0, len(df), 5))
		if len(df) and ticks[-1] != len(df) - 1:
			ticks.append(len(df) - 1)
		ax.set_xticks(ticks)
		ax.set_xticklabels([])

	plt.tight_layout()
	plt.close(fig)
	return fig

# 用于将同一个 cluster 的图像画在一起
def plotCluster(cluster: list):
	fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
	axes = axes.ravel()

	plotConfig = [
		("log_return", "black", "Log Return"),
		("rstd", "red", "Std"),
		("rvol", "red", "rvol"),
		("gkvol", "green", "gkvol"),
	]

	for ax, (column, color, title) in zip(axes, plotConfig):
		for df in cluster:
			x = range(len(df))
			ax.plot(x, df[column], color=color, linewidth=1.0)

		ax.set_title(title)
		ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.4)
		if column == "log_return":
			ax.set_ylim(-0.1, 0.1)
			ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
		ax.tick_params(axis="x", labelbottom=False, bottom=True)

		ticks = []
		if cluster:
			ticks = list(range(0, max(len(df) for df in cluster), 5))
			max_len = max(len(df) for df in cluster)
			if max_len and ticks and ticks[-1] != max_len - 1:
				ticks.append(max_len - 1)
		ax.set_xticks(ticks)
		ax.set_xticklabels([])

	plt.tight_layout()
	plt.close(fig)
	return fig

# 给出了所有的 data 和 labels，用 TSNE 将它们降维到二维空间并画出来，看看能不能找到一些有意义的模式
# data 是一个 DataFrame 的列表，labels 是一个整数列表，表示每个 DataFrame 属于哪个 cluster
def plotTSNE(data: list, labels: list):
	X = []
	for dataset in data:
		features = dataset[["rmean", "rstd", "rskew", "rkurt", "rvol", "pkvol", "gkvol"]].values.flatten()
		X.append(features)
	X = pd.DataFrame(X)

	tsne = TSNE(n_components=2, random_state=0)
	XEmbedded = tsne.fit_transform(X)

	fig, ax = plt.subplots(figsize=(10, 6))
	scatter = ax.scatter(XEmbedded[:, 0], XEmbedded[:, 1], c=labels, cmap="tab10", alpha=0.7)
	ax.legend(*scatter.legend_elements(), title="Clusters")
	ax.set_title("t-SNE Visualization of Clusters")
	ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)

	plt.tight_layout()
	plt.close(fig)
	return fig