
# 将股票数据每隔 k 个交易日划分为一个区间，返回一个包含所有区间的列表，每个区间是一个 DataFrame
# 最后一个区间如果不是 k 个交易日则舍弃
def partition(df, k: int):
    result = []
    for i in range(0, len(df), k):
        if i + k <= len(df):
            result.append(df.iloc[i : i + k].reset_index(drop=True))
    return result