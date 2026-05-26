import math
import time
import os
import pandas as pd

import tushare as ts

def readToken(filePath: str):
	with open(filePath, "r") as f:
		token = f.read().strip()
	return token

ts.set_token(readToken("data/token.txt"))
pro = ts.pro_api()

cacheDir = "cache"

def getStockPool():
	with open("data/stock_pool.txt", "r") as f:
		stockPool = [line.strip() for line in f]
	return stockPool

# 获得指定时间段的股票数据并清洗，只保留交易日期和 OHLCV 数据
# 如果发生网络问题，采用二进制指数退避算法重试
# 由于接口返回的数据是从新到旧，需要反向转置成从旧到新的顺序，方便进行滑动窗口计算
def getStockData(code: str, startDate: str, endDate: str, maxRetries: int = 10):
	cachedFile = f"{cacheDir}/csv/{"".join(code.split("."))}_{startDate}_{endDate}.csv"
	if os.path.exists(cachedFile):
		return pd.read_csv(cachedFile)
	retryCount, dfRaw = 0, None
	while retryCount < maxRetries:
		try:
			dfRaw = pro.daily(ts_code=code, start_date=startDate, end_date=endDate)
			break
		except Exception as _:
			print(f"发生网络问题，正在重试...（{retryCount + 1}/{maxRetries}）")
			retryCount += 1
			time.sleep(0.1 * 2 ** retryCount)
	if dfRaw is None:
		raise Exception("无法获取数据，请检查网络连接或稍后再试。")
	df = dfRaw[["trade_date", "open", "high", "low", "close", "vol"]]
	df = df.iloc[::-1].reset_index(drop=True)
	df.to_csv(cachedFile, index=False)
	return df

# 获取一组股票的指定时间段数据
def getMultipleStockData(codes: list, startDate: str, endDate: str):
	data = {}
	for code in codes:
		data[code] = getStockData(code, startDate, endDate)
	return data