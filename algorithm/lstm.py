import torch
import datetime as dt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import extract, calcError

class LSTMRegressor(nn.Module):
	# 输入是 B 个长度为 L 含有 F 个特征的序列，输出是 B 个对应的 scalar label，使用 BiRNN
	# (B, L, F) -> (B, 1)
	def __init__(self, inputSize: int = 1, hiddenSize: int = 128, layers: int = 2, dropout: float = 0.2):
		super(LSTMRegressor, self).__init__()
		self.lstm = nn.LSTM(
			inputSize, 
			hiddenSize, 
			layers, 
			batch_first=True, 
			dropout=dropout,
			bidirectional=True
		)
		self.head = nn.Sequential(
			nn.LayerNorm(hiddenSize * 2),
			nn.Linear(hiddenSize * 2, hiddenSize),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hiddenSize, 1)
		)

	def forward(self, x):
		_, (hn, _) = self.lstm(x)
		features = torch.cat([hn[-2], hn[-1]], dim=-1)
		out = self.head(features)
		return out.squeeze(-1)

# verbose 的参数的含义是 none 不输出，batch 最细，epoch 只输出每个 epoch 测试集上的 RMSE
def trainLSTM(train, test, 
			  epochs: int = 100, batchSize: int = 32, lr: float = 1e-4,
			  verbose: str = "epoch"):
	assert verbose in ["none", "batch", "epoch"], "verbose 参数不合法！"

	featureIndex = ["log_return", "rmean", "rstd", "rvol", "pkvol", "gkvol"]
	XTrain, yTrain = extract(train, featureIndex)
	XTest, yTest = extract(test, featureIndex)

	XTrain = torch.tensor(XTrain, dtype=torch.float32).unsqueeze(-1)
	yTrain = torch.tensor(yTrain, dtype=torch.float32)
	XTest = torch.tensor(XTest, dtype=torch.float32).unsqueeze(-1)

	XTrain = XTrain.reshape(XTrain.shape[0], -1, len(featureIndex))
	XTest = XTest.reshape(XTest.shape[0], -1, len(featureIndex))

	# print(XTrain.shape) 应该是 (6867, L, F)
	# print(yTrain.shape) 应该是 (6867)
	# print(XTest.shape) 应该是 (3313, L, F)	

	trainDataset = TensorDataset(XTrain, yTrain)
	trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

	model = LSTMRegressor(len(featureIndex), 128, 2, 0.1)
	criterion = nn.SmoothL1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

	bestRMSE = float("inf")
	bestModel = None

	for epoch in range(epochs):
		currentBatch = 0
		start = dt.datetime.now()
		model.train()
		for batchX, batchY in trainLoader:
			currentBatch += 1
			optimizer.zero_grad()
			outputs = model(batchX)
			loss = criterion(outputs, batchY)
			loss.backward()
			optimizer.step()
			if currentBatch % 20 == 0 and verbose == "batch":
				print(f"Epoch {epoch + 1}/{epochs}, Batch {currentBatch}, Loss: {loss.item():.6f}")
		end = dt.datetime.now()
		trainCost = (end - start).total_seconds()

		# 在测试集上评估当前模型性能
		model.eval()
		with torch.no_grad():
			pred = model(XTest).numpy()
		rmse = calcError(yTest, pred, metric="rmse")
		if rmse < bestRMSE:
			bestRMSE = rmse
			# 直接赋值 bestModel = model 会导致后续训练修改 bestModel 的参数，因此需要通过 state_dict
			bestModel = LSTMRegressor(len(featureIndex), 128, 2, 0.1)
			bestModel.load_state_dict(model.state_dict())

		if verbose in ["epoch", "batch"]:
			print(f"Epoch {epoch + 1}/{epochs}, Test RMSE: {rmse:.6f}, Train Cost: {trainCost:.3f}s")

	torch.save(bestModel.state_dict(), f"cache/models/lstm_rmse{int(bestRMSE * 1e6)}_lr{lr}.pth")

	bestModel.eval()
	with torch.no_grad():
		predictions = bestModel(XTest).numpy()

	return predictions