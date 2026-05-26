import matplotlib.pyplot as plt


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
