import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from .utils import extract

# 在 train 上训练各类集成学习算法，然后在 test 上返回预测结果
def randomForest(train, test, estimatorCount: int = 400):
	featureIndex = ["log_return", "rmean", "rstd", "rvol", "pkvol", "gkvol"]
	XTrain, yTrain = extract(train, featureIndex)
	XTest, _ = extract(test, featureIndex)

	model = RandomForestRegressor(n_estimators=estimatorCount, n_jobs=-1)
	model.fit(XTrain, yTrain)

	return model.predict(XTest)

def GBDT(train, test, estimatorCount: int = 200, lr: float = 0.05):
	featureIndex = ["log_return", "rmean", "rstd", "rvol", "pkvol", "gkvol"]
	XTrain, yTrain = extract(train, featureIndex)
	XTest, _ = extract(test, featureIndex)

	model = GradientBoostingRegressor(n_estimators=estimatorCount, learning_rate=lr, max_depth=3)
	model.fit(XTrain, yTrain)

	return model.predict(XTest)