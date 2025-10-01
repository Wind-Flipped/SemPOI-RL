import numpy as np
import torch
from sklearn.metrics import f1_score
import math
try:
    import ipdb
except:
    pass

# POI Metrics
def p_rec(tops, labels, k):
    """
    Calculate the recall score for top-k recommendations.
    Returns:
        float: Recall
    This function computes the recall, which measures the proportion of actual items of interest that are
    included in the top-k recommendations. The function iterates over pairs of top-k recommendations and
    corresponding actual items, calculates the recall for each user, and then averages these values.
    """
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label) # 计算它们的交集
        r = len(hit) / (len(set(label)) - 1)
        res += r
    return res

def p_precision(tops, labels, k):
    """
    Calculate the precision score for top-k recommendations.
    Returns:
        float
    This function computes the precision, which measures the proportion of recommended items in the top-k list
    that are actual items of interest. The function iterates over pairs of top-k recommendations and corresponding
    actual items, calculates the precision for each user, and then averages these values.
    """
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        r = len(hit) / k # 1 for '0'
        res += r
    return res

def p_f1(tops, labels, k):
    """
    Calculate the F1 score for top-k recommendations.
    Returns:
        float
    This function computes the F1 score, which is the harmonic mean of precision and recall. The function iterates
    over pairs of top-k recommendations and corresponding actual items, calculates the F1 score for each user,
    and then averages these values. It handles cases where the denominator in the F1 score calculation is zero.
    """
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        p = len(hit) / k # 1 for '0'
        r = len(hit) / (len(set(label)) - 1) # 1 for '0'
        try:
            res += (2 * p * r / (p + r))
        except:
            res += 0
    return res

def p_ndcg(tops, labels, k):
    """
    Calculate the normalized discounted cumulative gain (NDCG) for top-k recommendations.
    Returns:
        float
    This function computes the NDCG, a measure of ranking quality. For each user, it calculates the DCG (Discounted
    Cumulative Gain) and IDCG (Ideal DCG) and then normalizes the DCG by IDCG to get the NDCG score. It averages
    these scores across all users. The function handles both relevant (rel = 1) and non-relevant (rel = 0) items.
    """
    res = 0.
    for top, label in zip(tops, labels):
        dcg = 0.
        idcg = 0.
        for i, p in enumerate(top[:k], start=1):
            rel = 1 if np.isin(p, label) else 0
            dcg += (2 ** rel - 1) / (np.log2(i + 1))
            idcg += 1 / (np.log2(i + 1))
        ndcg = dcg / idcg
        res += ndcg
    return res

# Region Metrics
def r_map(tops, labels, weight=None):
    """
    Calculate the mean average precision (MAP) for region-based recommendations.
    Returns:
        float
    This function computes MAP, a measure that considers the order of recommendations. It iterates over pairs
    of recommended and actual regions, calculates the precision at each relevant item found, averages these
    precision values, and then averages across all instances. If weights are provided, they are applied to
    each instance's precision.
    """
    map_ = []
    for instance_idx, (top, label) in enumerate(zip(tops, labels)):
        m = 0.
        relative_num = 0.
        for i, k in enumerate(top,start=1):
            if k == label:
                m += (relative_num + 1) / i
                relative_num += 1
        if relative_num > 0:
            m /= relative_num
        if weight: m *= weight[instance_idx]
        map_.append(m)
    return np.mean(map_)

def r_precision(tops, labels, weight=None):
    """
    Calculate the precision for region-based recommendations.
    Returns:
        float
    This function computes precision for region-based recommendations. It iterates over each pair of recommended
    and actual regions, calculates the proportion of correctly predicted regions, and then averages these values
    across all instances. If weights are provided, they are applied to each instance's precision score.
    """
    res = []
    tops = tops.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    for instance_idx, (ps, l) in enumerate(zip(tops, labels)):
        # ipdb.set_trace()
        showup = np.sum(ps == l)
        prec = showup / len(ps)
        if weight: prec *= weight[instance_idx]
        res.append(prec)
    return np.mean(res)

def r_acc(predict, label):
    """
    Calculate the accuracy for region-based recommendations.
    Returns:
        float
    This function computes accuracy, which is the ratio of correctly predicted regions to the total number
    of predictions. It compares the predicted regions with the actual regions of interest and calculates the
    proportion of correct predictions.
    """
    return torch.sum(predict == label) / label.size(0)

def r_f1(predict, label, avg):
    """
    Calculate the F1 score for region-based recommendations.
    Returns:
        float
    The F1 score is a measure of a test's accuracy and considers both the precision and recall. It is the
    harmonic mean of precision and recall. This function computes the F1 score for the given predictions and
    labels, applying the specified averaging method.
    """
    return f1_score(label.cpu(), predict.cpu(), average=avg)

def weight_func(x):
    """
    Calculate a weighting factor based on the input value.
    Returns:
        float
    This function calculates a weighting factor using a cosine function. It is often used to transform a value
    (like similarity or relevance scores) into a weighting factor that can be used in further calculations or
    algorithms. The transformation is designed to decrease the weight as the input value increases.
    """
    return np.cos(np.pi / 2 * x * 10)

# ============================= Trip metrics =========================== #
def f1_score(target, predict, noloop=False):
    """
    Compute F1 Score for recommended trajectories
    :param target: the actual trajectory
    :param predict: the predict trajectory
    :param noloop:

    :return: f1
    """
    assert (isinstance(noloop, bool))
    assert (len(target) > 0)
    assert (len(predict) > 0)

    if noloop:
        intersize = len(set(target) & set(predict))
    else:
        match_tags = np.zeros(len(target), dtype=np.bool_)
        for poi in predict:
            for j in range(len(target)):
                if not match_tags[j] and poi == target[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(target)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1

    f1 = 2 * precision * recall * 1.0 / denominator

    return f1


def pairs_f1_score(target, predict):
    """
    Compute Pairs_F1 Score for recommended trajectories
    :param target:
    :param predict:
    :return: pairs_f1
    """
    # Check if number of elements > 0
    assert target.numel() > 0
    n = target.numel()
    nr = predict.numel()
    if n == 1 or nr == 1:
        return 1.0 if target.item() == predict.item() else 0.0
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    order_dict = dict()
    for i, poi in enumerate(target):
        order_dict[poi.item()] = i

    nc = 0
    for i in range(nr):
        poi1 = predict[i].item()
        for j in range(i + 1, nr):
            poi2 = predict[j].item()
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        pairs_f1 = 0
    else:
        pairs_f1 = 2. * precision * recall / (precision + recall)

    return pairs_f1

def count_repetition_percentage(input_data):
    # if list
    if isinstance(input_data, list):
        unique_items = set(input_data)
    # if tensor
    elif hasattr(input_data, 'numpy'):
        unique_items = set(input_data.cpu().numpy().tolist())
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total_items = len(input_data)
    repetition_items_count = total_items - len(unique_items)
    repetition_ratio = repetition_items_count / total_items

    return repetition_ratio


def count_adjacent_repetition_rate(input_data):
    if isinstance(input_data, list):
        predictions = input_data
    elif hasattr(input_data, 'numpy'):
        predictions = input_data.cpu().numpy().flatten().tolist()
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total = len(predictions)
    if total < 2:
        return 0.0

    repeated = sum(1 for i in range(1, total) if predictions[i] == predictions[i - 1])
    repetition_ratio = repeated / (total - 1)

    return repetition_ratio

# ============================= New Metrics =========================== #
def hit_rate(predict: torch.Tensor, target: torch.Tensor) -> float:
    """Hit Rate: 逐位置命中率 (与分类准确率相同)。两序列长度相同。
    Args:
        predict: 预测序列 (1D tensor)
        target:  真值序列 (1D tensor)
    Returns: float
    """
    assert predict.shape == target.shape
    if target.numel() == 0:
        return 0.0
    return (predict == target).float().mean().item()


def recall_rate(predict: torch.Tensor, target: torch.Tensor, unique: bool = True) -> float:
    """Recall: 预测命中的真实 POI 覆盖率。
    两种模式:
      unique=False (默认): 按元素计数, min(预测与真实逐元素匹配次数, 真实长度)/真实长度。
      unique=True: 基于集合: |Pred∩True| / |True|。
    Args:
        predict: 预测序列
        target: 真实序列
        unique: 是否基于集合
    """
    assert target.numel() > 0
    if unique:
        p_set = set(predict.tolist())
        t_set = set(target.tolist())
        return len(p_set & t_set) / max(1, len(t_set))
    # 多重匹配 (稳定匹配策略)
    matched = 0
    used = [False]*target.numel()
    t_list = target.tolist()
    for pid in predict.tolist():
        for i, tid in enumerate(t_list):
            if not used[i] and pid == tid:
                used[i] = True
                matched += 1
                break
    return matched / target.numel()


def levenshtein_distance(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> float:
    """最短编辑距离 (Levenshtein). 可归一化到 [0,1] (距离 / max_len)。"""
    p = predict.tolist()
    t = target.tolist()
    n, m = len(p), len(t)
    if n == 0:
        return 0.0 if m == 0 else (1.0 if normalize else m)
    if m == 0:
        return 1.0 if normalize else n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if p[i-1] == t[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # 删除
                           dp[i][j-1] + 1,      # 插入
                           dp[i-1][j-1] + cost) # 替换
    dist = dp[n][m]
    if normalize:
        return dist / max(n, m)
    return float(dist)


def dtw_distance(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> float:
    """Dynamic Time Warping (基于0/1匹配代价)。代价: 相等=0, 不等=1。
    若 normalize=True 返回 distance / (n+m)。"""
    p = predict.tolist()
    t = target.tolist()
    n, m = len(p), len(t)
    if n == 0 or m == 0:
        return 0.0
    inf = 1e9
    dp = [[inf]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if p[i-1] == t[j-1] else 1
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    dist = dp[n][m]
    return dist / (n + m) if normalize else float(dist)


def _haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def average_geo_distance_error(predict: torch.Tensor, target: torch.Tensor, poi_meta: dict) -> float:
    """平均地理距离误差: 逐位置 (经纬度已知) 计算预测与真值之间的哈弗辛距离 (km) 的平均。
    缺失坐标的位置跳过; 若全部缺失返回 np.nan。
    poi_meta[num_id] 应含 'lat','lon'."""
    assert predict.shape == target.shape
    distances = []
    for pid, tid in zip(predict.tolist(), target.tolist()):
        pm = poi_meta.get(pid)
        tm = poi_meta.get(tid)
        if not pm or not tm:
            continue
        if 'lat' not in pm or 'lat' not in tm:
            continue
        distances.append(_haversine(pm['lat'], pm['lon'], tm['lat'], tm['lon']))
    if not distances:
        return float('nan')
    return float(np.mean(distances))


def category_consistency_rate(predict: torch.Tensor, target: torch.Tensor, poi_meta: dict) -> float:
    """类别一致率: 逐位置比较预测与真值 POI 的主类别(main_category 或 categories[0]) 是否相同。
    若任一位置缺类别则跳过; 若全部跳过返回 np.nan。"""
    assert predict.shape == target.shape
    matches = 0
    total = 0
    for pid, tid in zip(predict.tolist(), target.tolist()):
        pm = poi_meta.get(pid)
        tm = poi_meta.get(tid)
        if not pm or not tm:
            continue
        def _extract(meta):
            if meta.get('main_category'):
                return meta['main_category']
            cats = meta.get('categories')
            if isinstance(cats, list) and cats:
                return cats[0]
            return None
        pc = _extract(pm)
        tc = _extract(tm)
        if pc is None or tc is None:
            continue
        total += 1
        if pc == tc:
            matches += 1
    if total == 0:
        return float('nan')
    return matches / total


def region_match_rate(predict: torch.Tensor, target: torch.Tensor, poi_meta: dict) -> float:
    """区域一致率: 逐位置比较预测与真值 POI 的主区域(main_region 或 regions[0]) 是否相同。
    若任一位置缺区域则跳过; 若全部跳过返回 np.nan。
    """
    assert predict.shape == target.shape
    matches = 0
    total = 0
    for pid, tid in zip(predict.tolist(), target.tolist()):
        pm = poi_meta.get(pid)
        tm = poi_meta.get(tid)
        if not pm or not tm:
            continue
        def _extract(meta):
            if meta.get('main_region'):
                return meta['main_region']
            regs = meta.get('regions')
            if isinstance(regs, list) and regs:
                return regs[0]
            return None
        pr = _extract(pm)
        tr = _extract(tm)
        if pr is None or tr is None:
            continue
        total += 1
        if pr == tr:
            matches += 1
    if total == 0:
        return float('nan')
    return matches / total


def relative_path_distance_error(predict: torch.Tensor, target: torch.Tensor, poi_meta: dict, normalize: bool = True) -> float:
    """相对路径距离误差:
    计算预测轨迹与真实轨迹的路径总长 (相邻点哈弗辛距离求和), 返回 |L_pred - L_true| / L_true (若 normalize)。
    若真实路径长度为 0 或无法计算则返回 nan。
    Args:
        predict: 预测序列 1D tensor
        target:  真值序列 1D tensor
        poi_meta: 含 lat/lon 的字典
        normalize: 是否返回相对误差 (True)；False 时返回绝对差 |L_pred - L_true| (km)
    """
    def path_len(seq):
        total = 0.0
        coords = []
        for pid in seq.tolist():
            meta = poi_meta.get(pid)
            if meta and 'lat' in meta and 'lon' in meta:
                coords.append((meta['lat'], meta['lon']))
        if len(coords) < 2:
            return float('nan')
        for (lat1, lon1), (lat2, lon2) in zip(coords[:-1], coords[1:]):
            total += _haversine(lat1, lon1, lat2, lon2)
        return total

    L_t = path_len(target)
    L_p = path_len(predict)
    if math.isnan(L_t) or math.isnan(L_p):
        return float('nan')
    if normalize:
        if L_t == 0:
            return float('nan')
        return abs(L_p - L_t) / L_t
    else:
        return abs(L_p - L_t)


def centroid_geo_distance(predict: torch.Tensor, target: torch.Tensor, poi_meta: dict) -> float:
    """重心(质心)经纬度距离:
    取预测与真实轨迹中所有有效坐标的 (lat, lon) 平均作为质心, 计算两质心间哈弗辛距离 (km)。
    若任一轨迹无有效坐标返回 nan。
    """
    def centroid(seq):
        lats = []
        lons = []
        for pid in seq.tolist():
            meta = poi_meta.get(pid)
            if meta and 'lat' in meta and 'lon' in meta:
                lats.append(meta['lat'])
                lons.append(meta['lon'])
        if not lats:
            return None
        return sum(lats)/len(lats), sum(lons)/len(lons)

    c_p = centroid(predict)
    c_t = centroid(target)
    if c_p is None or c_t is None:
        return float('nan')
    return _haversine(c_p[0], c_p[1], c_t[0], c_t[1])

def diversity_rate(predict: torch.Tensor) -> float:
    """多样性: 预测序列中不同 POI 的比例。"""
    p_list = predict.tolist()
    if not p_list:
        return float('nan')
    return len(set(p_list)) / len(p_list)