import numpy as np
import torch
from sklearn.metrics import f1_score
import math
try:
    import ipdb
except:
    pass
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
    """Hit Rate: position-wise accuracy (equivalent to classification accuracy).
    Args:
        predict: Predicted sequence (1D tensor)
        target:  Ground-truth sequence (1D tensor)
    Returns: float
    """
    assert predict.shape == target.shape
    if target.numel() == 0:
        return 0.0
    return (predict == target).float().mean().item()


def recall_rate(predict: torch.Tensor, target: torch.Tensor, unique: bool = True) -> float:
    """Recall: coverage of ground-truth POIs hit by predictions.
    Modes:
        unique=False: Count element-wise matches, min(matches, |True|) / |True|.
        unique=True: Set-based ratio |Pred ∩ True| / |True|.
    Args:
        predict: Predicted sequence
        target: Ground-truth sequence
        unique: Whether to compute using set semantics
    """
    assert target.numel() > 0
    if unique:
        p_set = set(predict.tolist())
        t_set = set(target.tolist())
        return len(p_set & t_set) / max(1, len(t_set))
    # Allow multiple matches using a stable matching style strategy
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
    """Minimum edit distance (Levenshtein). Optionally normalized to [0, 1] via distance / max_len."""
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
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                           dp[i][j-1] + 1,      # insertion
                           dp[i-1][j-1] + cost) # substitution
    dist = dp[n][m]
    if normalize:
        return dist / max(n, m)
    return float(dist)


def dtw_distance(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> float:
    """Dynamic Time Warping with a 0/1 matching cost (match=0, mismatch=1).
    When normalize=True, returns distance / (n + m)."""
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
    """Average geographic distance error: mean Haversine distance (km) between
    predicted and ground-truth POIs at each position when latitude/longitude are available.
    Skip positions with missing coordinates; return np.nan if all positions are skipped.
    Expects poi_meta[num_id] to include 'lat' and 'lon'."""
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
    """Category consistency: compare the primary category (main_category or categories[0])
    between predicted and ground-truth POIs at each position.
    Skip positions lacking category metadata; return np.nan if every position is skipped."""
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
    """Region consistency: compare the primary region (main_region or regions[0])
    between predicted and ground-truth POIs at each position.
    Skip positions lacking region metadata; return np.nan if every position is skipped."""
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
    """Relative path-length error:
    Compare total path length of predicted versus ground-truth trajectories (sum of pairwise Haversine distances).
    Return |L_pred - L_true| / L_true when normalize=True.
    If the ground-truth path has zero length or cannot be computed, return np.nan.
    Args:
        predict: Predicted sequence as a 1D tensor
        target: Ground-truth sequence as a 1D tensor
        poi_meta: Dictionary containing lat/lon metadata
        normalize: Whether to compute relative error (True) or absolute difference |L_pred - L_true| in km
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
    """Centroid geodesic distance:
    Compute the centroid (mean latitude/longitude) of each trajectory using valid coordinates
    and return the Haversine distance (km) between the two centroids.
    Return np.nan if either trajectory has no valid coordinates."""
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
    """Diversity: proportion of unique POIs within the predicted sequence."""
    p_list = predict.tolist()
    if not p_list:
        return float('nan')
    return len(set(p_list)) / len(p_list)