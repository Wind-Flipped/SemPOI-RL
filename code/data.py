from random import shuffle, choice
import numpy as np
import scipy.sparse as sp
from copy import copy
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import pandas as pd
import collections
from os.path import join
import torch
import json
# import dgl
import os
import pickle
from collections import Counter
try:
    import ipdb
except ImportError:
    pass
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timezone
from utils import *

def convert_timestamp(region, timestamp, city_tz_mapping):

    ts_sec = int(timestamp)
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    local_tz = pytz.timezone(city_tz_mapping.get(region, "UTC"))
    local_dt = dt_utc.astimezone(local_tz)
    return local_dt, local_dt.hour


def compute_trajectory_duration(trajectory):
    times = []
    for point in trajectory:
        ts = point[2]
        ts_sec = int(ts)
        dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        times.append(dt)
    if not times:
        return None, None, None, None, None
    min_time = min(times)
    max_time = max(times)
    duration = max_time - min_time
    return duration, min_time, max_time

class KGDataset(Dataset):
    """
    A custom dataset class for handling knowledge graph (KG) data in machine learning models.
    This class processes and stores knowledge graph data, including entities, relations, and triples.
    """
    def __init__(self, args):
        kg_data = pd.read_csv(args.kg_path, sep='\t', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.args = args

    @property
    def entity_count(self):
        """
        Returns the total count of unique entities in the knowledge graph.
        Returns:
            int
        """
        # start from one
        return self.kg_data['t'].max() + 2

    @property
    def relation_count(self):
        """
        Returns the total count of unique relations in the knowledge graph.
        Returns:
            int
        """
        return self.kg_data['r'].max()+2

    def get_kg_dict(self, poi_num):
        """
        Generates a dictionary with POI-specific KG entity and relation information.
        Returns:
            dict
        """
        entity_num = self.args.entity_num_per_poi # 2
        p2es = dict()
        p2rs = dict()
        for poi in range(poi_num):
            rts = self.kg_dict.get(poi, False)
            if rts:
                tails = list(map(lambda x:x[1], rts))
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) >= entity_num):
                    p2es[poi] = torch.IntTensor(tails).to(self.args.device)[:entity_num]
                    p2rs[poi] = torch.IntTensor(relations).to(self.args.device)[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count]*(entity_num-len(tails)))
                    relations.extend([self.relation_count]*(entity_num-len(relations)))
                    p2es[poi] = torch.IntTensor(tails).to(self.args.device)
                    p2rs[poi] = torch.IntTensor(relations).to(self.args.device)
            else:
                p2es[poi] = torch.IntTensor([self.entity_count]*entity_num).to(self.args.device)
                p2rs[poi] = torch.IntTensor([self.relation_count]*entity_num).to(self.args.device)
        return p2es, p2rs


    def generate_kg_data(self, kg_data):
        """
        Constructs a dictionary representation of the knowledge graph.
        Returns:
            dict
            list
        """
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        """
        Returns the total number of head entities in the knowledge graph.
        Returns:
            int
        """
        return len(self.kg_dict)

    def __getitem__(self, index):
        """
        Retrieves a KG triple (head, relation, positive tail, negative tail) at a specified index.
        Returns:
            tuple
        """
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail

class TravelDataset(Dataset):
    """
    A custom dataset class for handling travel-related data for machine learning models.
    This class processes and stores data related to points of interest (POIs), regions,
    user transactions, and associated features for use in models focusing on travel data.
    """
    def __init__(self, args, ori_data_path, dst_data_path, trans_data_path):
        ori_raw = list(map(lambda x: x.strip().split('\t'), open(ori_data_path, 'r')))
        dst_raw = list(map(lambda x: x.strip().split('\t'), open(dst_data_path, 'r')))
        trans_raw = list(map(lambda x: x.strip().split('\t'), open(trans_data_path, 'r')))
        self.args = args
        self.poi_idx = {}
        self.region_idx = {}
        self.tag_idx = {}
        self.region_poi = defaultdict(set)

        self.trans = []
        self.feats = []
        self.uids = []

        with open(f"../{self.args.dataset_name}/city_tz_mapping.pkl", "rb") as f:
            city_tz_mapping = pickle.load(f)

        for i in trans_raw:
            uid, cuid, ori_region, dst_region = i
            if ori_region not in self.region_idx: 
                self.region_idx[ori_region] = len(self.region_idx) 
            if dst_region not in self.region_idx:
                self.region_idx[dst_region] = len(self.region_idx)
            self.trans.append((self.region_idx[ori_region], self.region_idx[dst_region]))
            self.uids.append(int(uid))


        # for i in ori_raw + dst_raw:
        #     uid, cuid, _, bid, timestamp, std_tag = i
        #     if bid not in self.poi_idx:
        #         self.poi_idx[bid] = len(self.poi_idx) + 1
        #     if std_tag not in self.tag_idx:
        #         self.tag_idx[std_tag] = len(self.tag_idx)
        bid_counter = Counter([i[3] for i in (ori_raw + dst_raw)])
        tag_counter = Counter([i[5] for i in (ori_raw + dst_raw)])

        sorted_bids = [bid for bid, freq in bid_counter.most_common()]
        sorted_tags = [tag for tag, count in tag_counter.most_common()]

        self.poi_idx = {bid: idx for idx, bid in enumerate(sorted_bids, start=1)}
        self.tag_idx = {tag: idx for idx, tag in enumerate(sorted_tags, start=1)}

        with open(f"../{self.args.dataset_name}/poi_coord.pkl", 'rb') as f:
            poi_coord = pickle.load(f)

        all_lats = [coord[0] for coord in poi_coord.values()]
        all_lons = [coord[1] for coord in poi_coord.values()]
        global_lat_min = min(all_lats)
        global_lat_max = max(all_lats)
        global_lon_min = min(all_lons)
        global_lon_max = max(all_lons)
        lat_range = global_lat_max - global_lat_min if global_lat_max != global_lat_min else 1
        lon_range = global_lon_max - global_lon_min if global_lon_max != global_lon_min else 1

        def normalize_coord(coord):
            lat, lon = coord
            return np.array(((lat - global_lat_min) / lat_range, (lon - global_lon_min) / lon_range))

        self.poi_coord_norm = {poi: normalize_coord(coord) for poi, coord in poi_coord.items()}

        self.oris = []
        self.dsts = []

        ori_buffer = []
        train_buffer = []
        dst_buffer = []

        last_uid = '0'
        for i in ori_raw:
            uid, cuid, rid, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][0]].add(self.poi_idx[bid])
            local_time, local_hour = convert_timestamp(rid, timestamp, city_tz_mapping)
            if uid != last_uid:
                self.oris.append(ori_buffer)
                ori_buffer = []
                train_buffer = []
                last_uid = uid
            ori_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], float(timestamp), local_hour))
            train_buffer.append(self.poi_idx[bid])
        self.oris.append(ori_buffer)

        last_uid = '0'
        for i in dst_raw:
            uid, cuid, rid, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][1]].add(self.poi_idx[bid])
            local_time, local_hour = convert_timestamp(rid, timestamp, city_tz_mapping)
            if uid != last_uid:
                self.dsts.append(dst_buffer)
                dst_buffer = []
                last_uid = uid
            dst_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], float(timestamp), local_hour))
        self.dsts.append(dst_buffer)
        self.oris_norm = []
        self.oris_duration = []
        for traj in self.oris:
            timestamps = np.array([item[2] for item in traj])
            t_min, t_max = timestamps.min(), timestamps.max()
            duration = t_max - t_min
            t_range = duration if duration != 0 else 1
            norm_traj = []
            norm_times = (timestamps - t_min) / t_range
            for idx, (poi, tag, ts, local_hour) in enumerate(traj):
                norm_time = norm_times[idx]
                norm_traj.append((poi, tag, ts, local_hour, norm_time, self.poi_coord_norm[poi]))
            self.oris_norm.append(norm_traj)
            self.oris_duration.append(duration)

        self.dsts_norm = []
        self.dsts_duration = []
        for traj in self.dsts:
            timestamps = np.array([item[2] for item in traj])
            t_min, t_max = timestamps.min(), timestamps.max()
            duration = t_max - t_min
            t_range = duration if duration != 0 else 1
            norm_traj = []
            norm_times = (timestamps - t_min) / t_range
            for idx, (poi, tag, ts, local_hour) in enumerate(traj):
                norm_time = norm_times[idx]
                norm_traj.append((poi, tag, ts, local_hour, norm_time, self.poi_coord_norm[poi]))
            self.dsts_norm.append(norm_traj)
            self.dsts_duration.append(duration)

        if not os.path.exists(f'../{self.args.dataset_name}/poi_id.pkl'):
            os.makedirs(os.path.dirname(f'../{self.args.dataset_name}/poi_id.pkl'), exist_ok=True)
            with open(f'../{self.args.dataset_name}/poi_id.pkl', 'wb') as f:
                pickle.dump(self.poi_idx, f)
        if not os.path.exists(f'../{self.args.dataset_name}/region_poi.pkl'):
            os.makedirs(os.path.dirname(f'../{self.args.dataset_name}/region_poi.pkl'), exist_ok=True)
            with open(f'../{self.args.dataset_name}/region_poi.pkl', 'wb') as f:
                pickle.dump(self.region_poi, f)

        
    def __getitem__(self, index):
        """
        Retrieves an item at a specified index in the dataset.
        Returns:
            tuple
        """
        uid = self.uids[index]
        o = self.oris_norm[index]
        d = self.dsts_norm[index]
        t = self.trans[index]
        ori_ck = torch.LongTensor(list(map(lambda y: y[0], o)))
        # dst_ck = torch.LongTensor(list(map(lambda y: y[0], d))).unique()
        dst_ck = torch.LongTensor(list(map(lambda y: y[0], d)))
        o_hour = torch.LongTensor(list(map(lambda y: y[3], o)))
        d_hour = torch.LongTensor(list(map(lambda y: y[3], d)))
        # change hour 0 to 24
        o_hour[o_hour == 0] = 24
        d_hour[d_hour == 0] = 24
        # timestamp
        o_t = torch.FloatTensor(list(map(lambda y: y[4], o)))
        d_t = torch.FloatTensor(list(map(lambda y: y[4], d)))
        # location
        o_l = torch.FloatTensor(list(map(lambda y: y[5], o)))
        d_l = torch.FloatTensor(list(map(lambda y: y[5], d)))
        # Mask the intermediate POI IDs and hour IDs (excluding start and end POIs)
        d_mask_indices = torch.arange(1, len(dst_ck) - 1)
        masked_d_ck = dst_ck.clone()
        masked_d_ck[d_mask_indices] = 0.0
        masked_d_h = d_hour.clone()
        masked_d_h[d_mask_indices] = 0.0
        ori_rg = t[0]
        dst_rg = t[1]
        return uid, ori_ck, dst_ck, masked_d_ck, o_hour, d_hour, masked_d_h, o_t, d_t, o_l, d_l, ori_rg, dst_rg
    
    def __len__(self):
        """
        Returns the total number of transactions (user movements) in the dataset.
        Returns:
            int
        """
        return len(self.trans)


def random_split(dataset, dataset_name, split_path, ratios=[0.8, 0.1, 0.1]):
    """
    Splits a dataset into training, validation, and testing subsets randomly.
    Returns:
        tuple
    """
    trans = dataset.trans
    trans_by_pair = defaultdict(list)
    for u, t in enumerate(trans):
        trans_by_pair[t].append(u)
    
    train_indice, valid_indice, test_indice = [], [], []

    # if os.path.exists(split_path):
    #     train_indice, valid_indice, test_indice = np.load(split_path, allow_pickle=True)
    # else:
    for t, us in trans_by_pair.items():
        us_shuf = copy(us)
        np.random.shuffle(us_shuf)
        us_len = len(us)

        train_offset = int(us_len * ratios[0])
        valid_offset = int(us_len * (ratios[0] + ratios[1]))

        train_indice.extend(us_shuf[:train_offset])
        valid_indice.extend(us_shuf[train_offset:valid_offset])
        test_indice.extend(us_shuf[valid_offset:])

    with open(f'../{dataset_name}/data_split.pkl', 'wb') as file:
        pickle.dump([train_indice, valid_indice, test_indice], file)

    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice) # train_indices 是训练数据的索引列表

class TravelTextDataset(Dataset):
    """
    用于生成旅游轨迹文本描述的数据集类
    生成用于GRPO训练的prompt和reference对
    """
    def __init__(self, args, home_data_path, oot_data_path, travel_data_path):
        """
        初始化文本数据集
        
        Args:
            args: 配置参数
            home_data_path: 原始地数据路径
            oot_data_path: 目的地数据路径 
            travel_data_path: 旅行数据路径
        """
        self.args = args
        
        # 读取数据文件
        home_raw = list(map(lambda x: x.strip().split('\t'), open(home_data_path, 'r')))
        oot_raw = list(map(lambda x: x.strip().split('\t'), open(oot_data_path, 'r')))
        travel_raw = list(map(lambda x: x.strip().split('\t'), open(travel_data_path, 'r')))
        
        # 加载pickle文件
        with open(f"../{self.args.dataset_name}/poi_coord.pkl", "rb") as f:
            self.poi_coord = pickle.load(f)
        
        with open(f"../{self.args.dataset_name}/city_tz_mapping.pkl", "rb") as f:
            self.city_tz_mapping = pickle.load(f)

        with open(f"../{self.args.dataset_name}/poi_id.pkl", "rb") as f:
            self.poi_idx = pickle.load(f)
        
        # 按用户ID组织数据
        self.home_data = self._organize_data_by_user(home_raw)
        self.oot_data = self._organize_data_by_user(oot_raw)
        self.travel_data = {int(row[0]): (row[2], row[3]) for row in travel_raw}
        
        # 生成文本对
        self.text_pairs = self._generate_text_pairs()
    
    def _organize_data_by_user(self, raw_data):
        """按用户ID组织数据"""
        user_data = defaultdict(list)
        for row in raw_data:
            uid, cuid, rid, bid, timestamp, std_tag = row
            user_data[int(uid)].append({
                'cuid': cuid,
                'region': rid,
                'poi_id': bid,
                'timestamp': float(timestamp),
                'category': std_tag
            })
        return user_data
    
    def _format_trajectory_text(self, trajectory_data, region_name, query_type="hometown"):
        """
        格式化轨迹数据为文本描述
        
        Args:
            trajectory_data: 轨迹数据列表
            region_name: 区域名称
            query_type: 查询类型 ("hometown" 或 "destination")
        
        Returns:
            格式化的文本描述
        """
        if not trajectory_data:
            return ""
        
        # 按时间排序
        trajectory_data = sorted(trajectory_data, key=lambda x: x['timestamp'])
        
        poi_descriptions = []
        for i, point in enumerate(trajectory_data):
            # 限制最多50个POI
            if i >= 50:
                break
            poi_id = point['poi_id']
            category = point['category']
            timestamp = point['timestamp']
            
            # 获取POI坐标
            poi_num_id = self.poi_idx.get(poi_id, None)
            coord = self.poi_coord.get(poi_num_id, (0.0, 0.0))
            lat, lon = coord
            
            # 转换时间戳为UTC时间
            dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            utc_time_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            poi_desc = f"POI {i+1}: Category={category}, Location=({lat:.4f}, {lon:.4f}), Time={utc_time_str}"
            poi_descriptions.append(poi_desc)
        
        trajectory_text = "\n".join(poi_descriptions)
        
        if query_type == "hometown":
            prompt = f"""User's travel trajectory in {region_name} (hometown):
{trajectory_text}

Based on this hometown travel pattern, what would be the user's likely travel style when visiting a destination city? Please describe their preferences within 200 words for:
1. Types of attractions they would visit
2. Activity patterns and pace
3. Overall travel behavior

Travel style prediction:"""
        
        else:  # destination
            prompt = f"""User's actual travel trajectory in {region_name} (destination):
{trajectory_text}

Based on this actual travel behavior in the destination city, describe the user's travel style within 200 words including:
1. Types of attractions they prefer
2. Activity patterns and pace  
3. Overall travel behavior

Travel style description:"""
        
        return prompt
    
    def _generate_text_pairs(self):
        """生成prompt和reference文本对"""
        text_pairs = []
        
        for uid in self.travel_data:
            if uid not in self.home_data or uid not in self.oot_data:
                continue
            
            ori_region, dst_region = self.travel_data[uid]
            home_trajectory = self.home_data[uid]
            oot_trajectory = self.oot_data[uid]
            
            # 生成hometown prompt (用于训练时的输入)
            hometown_prompt = self._format_trajectory_text(
                home_trajectory, ori_region, "hometown"
            )
            
            # 生成destination reference (用于训练时的参考答案)
            destination_prompt = self._format_trajectory_text(
                oot_trajectory, dst_region, "destination"
            )
            
            if hometown_prompt and destination_prompt:
                text_pairs.append({
                    'uid': uid,
                    'hometown_prompt': hometown_prompt,
                    'destination_prompt': destination_prompt,
                    'ori_region': ori_region,
                    'dst_region': dst_region
                })
        
        return text_pairs
    
    def get_prompt_reference_pairs(self):
        """
        获取用于GRPO训练的prompt和reference对
        
        Returns:
            tuple: (prompts列表, references列表)
        """
        prompts = []
        references = []
        
        for pair in self.text_pairs:
            prompts.append(pair['hometown_prompt'])
            # 这里我们需要一个LLM来处理destination_prompt并生成reference
            # 暂时使用destination_prompt作为占位符
            references.append(pair['destination_prompt'])
        
        return prompts, references
    
    def __len__(self):
        """返回文本对的数量"""
        return len(self.text_pairs)
    
    def __getitem__(self, index):
        """获取指定索引的文本对"""
        return self.text_pairs[index]

def create_travel_text_dataset(args, dataset_name):
    """
    创建旅游文本数据集的便捷函数
    
    Args:
        args: 配置参数
        dataset_name: 数据集名称
    
    Returns:
        TravelTextDataset实例
    """
    home_path = f"../{dataset_name}/home.txt"
    oot_path = f"../{dataset_name}/oot.txt"
    travel_path = f"../{dataset_name}/travel.txt"
    
    return TravelTextDataset(args, home_path, oot_path, travel_path)
