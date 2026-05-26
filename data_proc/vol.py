import math

def calcLogReturns(df):
	df["log_return"] = (df["close"] / df["close"].shift(1)).apply(
		lambda x: math.log(x) if x > 0 else 0
	)
	return df

# 计算窗口上对数收益率的滑动平均值、标准差、偏度和峰度
def calcRollingMoment(df, moment: str = "mean", window: int = 5):
	raw = df["log_return"].rolling(window=window, min_periods=window).agg(moment)
	for i in range(1, window):
		raw.iloc[i - 1] = df["log_return"].iloc[:i].agg(moment)

	df[f"r{moment}"] = raw
	return df

# 计算滑动窗口波动率，为窗口内对数收益率的方均根
# 对于缺失的几个 NaN，滑动窗口大小为实际大小
def calcRollingVolatility(df, window: int = 5):
	assert window > 1, "窗口大小必须大于 1"
	assert "log_return" in df.columns, "数据框必须包含 log_return 字段"

	raw = df["log_return"].rolling(window=window, min_periods=window).apply(
		lambda x: (x ** 2).mean() ** 0.5
	)

	for i in range(1, window):
		raw.iloc[i - 1] = (df["log_return"].iloc[:i] ** 2).mean() ** 0.5

	df["rvol"] = raw
	return df

# 计算 Parkinson 波动率
def calcParkinsonVolatility(df):
	df["pkvol"] = (df["high"] / df["low"]).apply(
		lambda x: math.log(x) / math.sqrt(4 * math.log(2)) if x > 0 else 0
	)
	return df

# 计算 Garman-Klass 波动率
def calcGarmanKlassVolatility(df):
	lnHL = (df["high"] / df["low"]).apply(lambda x: math.log(x) if x > 0 else 0)
	lnCO = (df["close"] / df["open"]).apply(lambda x: math.log(x) if x > 0 else 0)
	df["gkvol"] = (0.5 * lnHL ** 2 - (2 * math.log(2) - 1) * lnCO ** 2) ** 0.5
	return df

def calcStats(df, index, window: int = 5):
	for idx in index:
		if idx.startswith("r") and (not idx.endswith("vol")):
			df = calcRollingMoment(df, idx[1:], window)
		elif idx == "rvol":
			df = calcRollingVolatility(df, window)
		elif idx == "pkvol":
			df = calcParkinsonVolatility(df)
		elif idx == "gkvol":
			df = calcGarmanKlassVolatility(df)
	return df