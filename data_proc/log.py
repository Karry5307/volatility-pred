# 一个 lr 对应 102 行 log，第一行代表学习率
# 后面 100 行是每个 epoch 的情况，最后一行是这个 lr 最好的模型在测试集上的表现
class LSTMLrLogEntry:
    def __init__(self, lr: float, history: list[float], bestRMSE: float, error: float):
        self.lr = lr
        self.history = history
        self.bestRMSE = bestRMSE
        self.error = error
    
    def __str__(self):
        return f"学习率：{self.lr}, 最佳 RMSE: {self.bestRMSE}, 误差: {self.error}\n" + \
            "\n".join([f"Epoch {i + 1}: RMSE={rmse:.6f}, Time={time:.3f}s" for i, (rmse, time) in enumerate(self.history)])

def parseTuneLSTMLr():
    result = {}
    with open("cache/logs/lstm.txt", "r") as f:
        lines = f.readlines()
        assert len(lines) % 102 == 0, "日志文件格式不正确！"
        for i in range(0, len(lines), 102):
            lr = float(lines[i].split()[1])
            history = []
            for j in range(i + 1, i + 101):
                epochInfo = lines[j].strip().split(" ")
                rmse = epochInfo[4].rstrip(",")
                time = epochInfo[7].rstrip("s")
                history.append((float(rmse), float(time)))
            finalInfo = lines[i + 101].strip().split(" ")
            bestRMSE = float(finalInfo[4].rstrip("，波动率均值为"))
            error = float(finalInfo[6])
            result[lr] = LSTMLrLogEntry(lr, history, bestRMSE, error)
    return result

# 一个 lr 对应 52 行 log，第一行代表头数
class TransformerHeadLogEntry:
    def __init__(self, heads: int, history: list[float], bestRMSE: float, error: float):
        self.heads = heads
        self.history = history
        self.bestRMSE = bestRMSE
        self.error = error
    
    def __str__(self):
        return f"头数：{self.heads}, 最佳 RMSE: {self.bestRMSE}, 误差: {self.error}\n" + \
            "\n".join([f"Epoch {i + 1}: RMSE={rmse:.6f}, Time={time:.3f}s" for i, (rmse, time) in enumerate(self.history)])

def parseTuneTransformerHead():
    result = {}
    with open("cache/logs/transformer.txt", "r") as f:
        lines = f.readlines()
        assert len(lines) % 52 == 0, "日志文件格式不正确！"
        for i in range(0, len(lines), 52):
            heads = int(lines[i].split()[1])
            history = []
            for j in range(i + 1, i + 51):
                epochInfo = lines[j].strip().split(" ")
                rmse = epochInfo[4].rstrip(",")
                time = epochInfo[7].rstrip("s")
                history.append((float(rmse), float(time)))
            finalInfo = lines[i + 51].strip().split(" ")
            bestRMSE = float(finalInfo[4].rstrip("，波动率均值为"))
            error = float(finalInfo[6])
            result[heads] = TransformerHeadLogEntry(heads, history, bestRMSE, error)
    return result