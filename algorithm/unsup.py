import numpy as np
from .utils import extract

# 使用 K-Medoids 聚类算法，通过如下方式定义距离函数：
# g(x) = (1 + tanh(ln(99x / (1 - x) + eps)))/ 2
# 此外，加入时间维度的考量，令 w_i = alpha + (1 - alpha) * exp(-beta * (n - i))
# # d(x, y) = sqrt(sum_i w_i (g(x_i) - g(y_i))^2)
def KMedoidsCluster(dataset: list, clusterCount: int = 5, 
					alpha: float = 0.05, beta: float = 0.09, maxIter: int = 100):
	
	featureIndex = ["rvol", "pkvol"]
	X, _ = extract(dataset, featureIndex)

	assert X.shape[1] % len(featureIndex) == 0
	sampleCount, n = X.shape[0], X.shape[1] // len(featureIndex)
	clusterCount = min(clusterCount, sampleCount)

	if sampleCount == 0:
		return np.array([], dtype=int)

	w = np.array([alpha + (1 - alpha) * np.exp(-beta * (n - i)) for i in range(n)])
	w = np.repeat(w, len(featureIndex))
	sqrtW = np.sqrt(w)

	# 将特征映射到加权欧氏空间，避免构造全量 n x n 距离矩阵
	XClip = np.clip(X, 1e-8, 1 - 1e-8)
	Xg = (1 + np.tanh(np.log(99 * XClip / (1 - XClip) + 1e-8))) / 2
	Xw = Xg * sqrtW

	def distToPoint(points: np.ndarray, point: np.ndarray):
		return np.linalg.norm(points - point, axis=1)

	def clusterMedoid(members: np.ndarray, batchSize: int = 128):
		memberPoints = Xw[members]
		memberNorm2 = np.sum(memberPoints ** 2, axis=1, keepdims=True)
		bestMember = members[0]
		bestScore = np.inf
		for start in range(0, len(members), batchSize):
			candidateIds = members[start : start + batchSize]
			candidatePoints = Xw[candidateIds]
			candidateNorm2 = np.sum(candidatePoints ** 2, axis=1, keepdims=True).T
			dist2 = memberNorm2 + candidateNorm2 - 2 * (memberPoints @ candidatePoints.T)
			dist2 = np.maximum(dist2, 0.0)
			totalDist = np.sqrt(dist2).sum(axis=0)
			localBest = np.argmin(totalDist)
			if totalDist[localBest] < bestScore:
				bestScore = totalDist[localBest]
				bestMember = candidateIds[localBest]
		return bestMember

	medoids = np.arange(clusterCount)
	for _ in range(maxIter):
		medoidPoints = Xw[medoids]
		distToMedoids = np.stack([distToPoint(Xw, medoidPoint) for medoidPoint in medoidPoints], axis=1)
		labels = np.argmin(distToMedoids, axis=1)
		newMedoids = medoids.copy()
		for c in range(clusterCount):
			members = np.where(labels == c)[0]
			if len(members) == 0:
				nonMedoids = np.setdiff1d(np.arange(sampleCount), newMedoids)
				if len(nonMedoids) > 0:
					distToCurrent = np.stack([distToPoint(Xw[nonMedoids], Xw[m]) for m in newMedoids], axis=1)
					newMedoids[c] = nonMedoids[np.argmax(np.min(distToCurrent, axis=1))]
				continue
			newMedoids[c] = clusterMedoid(members)
		if np.array_equal(newMedoids, medoids):
			break
		medoids = newMedoids

	medoidPoints = Xw[medoids]
	distToMedoids = np.stack([distToPoint(Xw, medoidPoint) for medoidPoint in medoidPoints], axis=1)
	labels = np.argmin(distToMedoids, axis=1)
	return labels