from copy import copy
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import pandas as pd
import collections
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

    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice)  # train_indices holds the indices for the training split

class TravelTextDataset(Dataset):
    """
    Dataset class that produces textual descriptions of travel trajectories.
    Generates prompt and reference pairs for GRPO training.
    """
    def __init__(self, args, home_data_path, oot_data_path, travel_data_path):
        """
        Initialize the text dataset.

        Args:
            args: Configuration arguments.
            home_data_path: Path to the hometown data file.
            oot_data_path: Path to the destination data file.
            travel_data_path: Path to the travel metadata file.
        """
        self.args = args
        
        # Load raw data files
        home_raw = list(map(lambda x: x.strip().split('\t'), open(home_data_path, 'r')))
        oot_raw = list(map(lambda x: x.strip().split('\t'), open(oot_data_path, 'r')))
        travel_raw = list(map(lambda x: x.strip().split('\t'), open(travel_data_path, 'r')))
        
        # Load supporting pickle files
        with open(f"../{self.args.dataset_name}/poi_coord.pkl", "rb") as f:
            self.poi_coord = pickle.load(f)
        
        with open(f"../{self.args.dataset_name}/city_tz_mapping.pkl", "rb") as f:
            self.city_tz_mapping = pickle.load(f)

        with open(f"../{self.args.dataset_name}/poi_id.pkl", "rb") as f:
            self.poi_idx = pickle.load(f)
        
        # Organize data by user ID
        self.home_data = self._organize_data_by_user(home_raw)
        self.oot_data = self._organize_data_by_user(oot_raw)
        self.travel_data = {int(row[0]): (row[2], row[3]) for row in travel_raw}
        
        # Generate prompt-reference pairs
        self.text_pairs = self._generate_text_pairs()
    
    def _organize_data_by_user(self, raw_data):
        """Organize raw trajectory rows by user ID."""
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
        Format a trajectory into a textual description.

        Args:
            trajectory_data: List of trajectory entries.
            region_name: Name of the region associated with the trajectory.
            query_type: Query mode ("hometown" or "destination").

        Returns:
            A formatted text description of the trajectory.
        """
        if not trajectory_data:
            return ""
        
        # Sort by timestamp
        trajectory_data = sorted(trajectory_data, key=lambda x: x['timestamp'])
        
        poi_descriptions = []
        # Retrieve the timezone string
        tz_str = self.city_tz_mapping.get(region_name, "UTC")
        try:
            local_tz = pytz.timezone(tz_str)
        except Exception:
            local_tz = timezone.utc
        for i, point in enumerate(trajectory_data):
            # Limit to at most 50 POIs
            if i >= 50:
                break
            poi_id = point['poi_id']
            category = point['category']
            timestamp = point['timestamp']
            # Look up the POI coordinates
            poi_num_id = self.poi_idx.get(poi_id, None)
            coord = self.poi_coord.get(poi_num_id, (0.0, 0.0))
            lat, lon = coord
            # Convert the timestamp to local time
            dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            dt_local = dt_utc.astimezone(local_tz)
            local_time_str = dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            poi_desc = f"POI {i+1}: Category={category}, Time={local_time_str}."
            poi_descriptions.append(poi_desc)
        
        trajectory_text = "\n".join(poi_descriptions)
        
        if query_type == "hometown":
            prompt = f"""As a professional travel analyst and user profiling expert, analyze the user's POI (Point of Interest) trajectory data from their hometown to predict their potential travel style at their destination. Based on their hometown patterns, determine whether they are likely to engage in activities such as cultural exploration, outdoor adventure, historical site visits, shopping, or relaxation. Provide a clear and concise prediction of their travel style without additional details within 200 words.
User's travel trajectory in {region_name} (hometown):
{trajectory_text}

Travel style prediction:"""
        
        else:  # destination
            prompt = f"""As a professional travel analyst and user profiling expert, analyze the user's POI (Point of Interest) trajectory data at the destination to infer their travel style. Based on this data, predict their primary travel activities, such as cultural exploration, outdoor adventure, historical site visits, shopping, or relaxation. Provide a clear and concise prediction of their travel stylewithout additional details within 200 words, which can be used to further forecast their future POI trajectories at the destination.
User's actual travel trajectory in {region_name} (destination):
{trajectory_text}

Travel style description:"""
        
        return prompt
    
    def _generate_text_pairs(self):
        """Generate paired hometown prompts and destination references."""
        text_pairs = []
        
        for uid in self.travel_data:
            if uid not in self.home_data or uid not in self.oot_data:
                continue
            
            ori_region, dst_region = self.travel_data[uid]
            home_trajectory = self.home_data[uid]
            oot_trajectory = self.oot_data[uid]
            
            # Build the hometown prompt (model input during training)
            hometown_prompt = self._format_trajectory_text(
                home_trajectory, ori_region, "hometown"
            )
            
            # Build the destination reference (training target)
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
        Retrieve prompt and reference pairs for GRPO training.

        Returns:
            tuple: (list of prompts, list of references)
        """
        prompts = []
        references = []
        
        for pair in self.text_pairs:
            prompts.append(pair['hometown_prompt'])
            # The destination prompt can be treated as a placeholder reference
            # until an LLM-generated description is available.
            references.append(pair['destination_prompt'])
        
        return prompts, references
    
    def __len__(self):
        """Return the number of prompt-reference pairs."""
        return len(self.text_pairs)
    
    def __getitem__(self, index):
        """Return the prompt-reference pair at the specified index."""
        return self.text_pairs[index]

    def get_data(self, trajectory_data, region_name, query_type="hometown"):
        """
        Format trajectory data into structured components.

        Args:
            trajectory_data: List of trajectory entries.
            region_name: Associated region name.
            query_type: Query mode ("hometown" or "destination").

        Returns:
            Tuple of POI IDs, categories, and timestamp strings.
        """
        if not trajectory_data:
            return ""

        # Sort by timestamp
        trajectory_data = sorted(trajectory_data, key=lambda x: x['timestamp'])

        poi_descriptions = []
        # Retrieve timezone information
        tz_str = self.city_tz_mapping.get(region_name, "UTC")
        try:
            local_tz = pytz.timezone(tz_str)
        except Exception:
            local_tz = timezone.utc

        poi_ids = []
        categories = []
        timestamps = []

        for i, point in enumerate(trajectory_data):
            poi_id = point['poi_id']
            category = point['category']
            timestamp = point['timestamp']
            # Look up the POI coordinates
            poi_num_id = self.poi_idx.get(poi_id, None)
            coord = self.poi_coord.get(poi_num_id, (0.0, 0.0))
            lat, lon = coord
            # Convert the timestamp to local time
            dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            dt_local = dt_utc.astimezone(local_tz)
            local_time_str = dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            poi_ids.append(poi_num_id)
            categories.append(category)
            timestamps.append(local_time_str)


        return poi_ids, categories, timestamps

    def generate_json(self):
        """Generate JSON-ready trajectory data pairs."""
        text_pairs = []

        for uid in self.travel_data:
            if uid not in self.home_data or uid not in self.oot_data:
                continue

            ori_region, dst_region = self.travel_data[uid]
            home_trajectory = self.home_data[uid]
            oot_trajectory = self.oot_data[uid]

            # Build hometown prompt components (model input)
            hometown_poi_ids, hometown_categories, hometown_timestamps = self.get_data(
                home_trajectory, ori_region, "hometown"
            )

            # Build destination reference components (training target)
            destination_poi_ids, destination_categories, destination_timestamps = self.get_data(
                oot_trajectory, dst_region, "destination"
            )

            if hometown_poi_ids and destination_poi_ids:
                text_pairs.append({
                    'uid': uid,
                    'ori_region': ori_region,
                    'dst_region': dst_region,
                    'hometown_poi_ids': hometown_poi_ids,
                    'hometown_categories': hometown_categories,
                    'hometown_timestamps': hometown_timestamps,
                    'destination_poi_ids': destination_poi_ids,
                    'destination_categories': destination_categories,
                    'destination_timestamps': destination_timestamps
                })

        return text_pairs


def create_travel_text_dataset(args, dataset_name):
    """
    Convenience function for building a TravelTextDataset instance.

    Args:
        args: Configuration arguments.
        dataset_name: Dataset identifier.

    Returns:
        TravelTextDataset instance.
    """
    home_path = f"../{dataset_name}/home.txt"
    oot_path = f"../{dataset_name}/oot.txt"
    travel_path = f"../{dataset_name}/travel.txt"
    
    return TravelTextDataset(args, home_path, oot_path, travel_path)


def extract_and_save_poi_metadata(dataset_name):
    """\
    Extract and persist metadata for each numeric POI identifier
    (original string bid, latitude, longitude, associated regions, categories).

    Source files:
        - ../{dataset_name}/home.txt and ../{dataset_name}/oot.txt
          Row format: uid\tcuid\trid\tbid\ttimestamp\tstd_tag
        - ../{dataset_name}/poi_id.pkl : mapping from original bid to numeric POI id
        - ../{dataset_name}/poi_coord.pkl : mapping from numeric POI id to (lat, lon)

    Outputs:
        - ../{dataset_name}/poi_meta.pkl  (dictionary serialized with pickle)
        - ../{dataset_name}/poi_meta.json (UTF-8 JSON for human inspection)

    Example structure for each POI:
        poi_meta[num_id] = {
            'bid': 'original_string_id',
            'lat': 31.2345,
            'lon': 121.4567,
            'regions': ['shanghai'],            # All unique regions encountered
            'categories': ['Food', 'Coffee'],    # All unique categories encountered
            'main_region': 'shanghai',          # Region with highest frequency
            'main_category': 'Food'             # Most frequent category
        }
    """
    import pickle, json, os
    from collections import defaultdict, Counter

    base_dir = f"../{dataset_name}"
    home_path = os.path.join(base_dir, 'home.txt')
    oot_path = os.path.join(base_dir, 'oot.txt')
    poi_id_path = os.path.join(base_dir, 'poi_id.pkl')
    poi_coord_path = os.path.join(base_dir, 'poi_coord.pkl')

    # Read required mapping files
    if not (os.path.exists(poi_id_path) and os.path.exists(poi_coord_path)):
        raise FileNotFoundError("Please generate poi_id.pkl and poi_coord.pkl first.")
    with open(poi_id_path, 'rb') as f:
        bid2num = pickle.load(f)  # Original bid -> numeric id
    # Inverse mapping: numeric id -> original bid
    num2bid = {v: k for k, v in bid2num.items()}
    with open(poi_coord_path, 'rb') as f:
        numid2coord = pickle.load(f)  # Numeric id -> (lat, lon)

    # Read trajectory text (home + oot)
    def _read_lines(p):
        if not os.path.exists(p):
            return []
        with open(p, 'r', encoding='utf-8') as f:
            return [ln.strip().split('\t') for ln in f if ln.strip()]

    all_rows = _read_lines(home_path) + _read_lines(oot_path)

    # Aggregate region/category statistics
    region_counter = defaultdict(Counter)   # num_id -> Counter(region)
    category_counter = defaultdict(Counter) # num_id -> Counter(category)

    for row in all_rows:
        if len(row) < 6:
            continue
        uid, cuid, rid, bid, ts, std_tag = row
        num_id = bid2num.get(bid)
        if num_id is None:
            continue
        region_counter[num_id][rid] += 1
        category_counter[num_id][std_tag] += 1

    poi_meta = {}
    for num_id, coord in numid2coord.items():
        lat, lon = coord if isinstance(coord, (list, tuple)) and len(coord) >= 2 else (0.0, 0.0)
        regions = list(region_counter[num_id].keys()) if num_id in region_counter else []
        categories = list(category_counter[num_id].keys()) if num_id in category_counter else []
        main_region = region_counter[num_id].most_common(1)[0][0] if region_counter[num_id] else None
        main_category = category_counter[num_id].most_common(1)[0][0] if category_counter[num_id] else None
        poi_meta[num_id] = {
            'bid': num2bid.get(num_id, None),
            'lat': float(lat),
            'lon': float(lon),
            'regions': regions,
            'categories': categories,
            'main_region': main_region,
            'main_category': main_category
        }

    # Persist metadata to disk
    meta_pkl = os.path.join(base_dir, 'poi_meta.pkl')
    meta_json = os.path.join(base_dir, 'poi_meta.json')
    with open(meta_pkl, 'wb') as f:
        pickle.dump(poi_meta, f)
    # Convert keys to strings for JSON serialization
    json_serializable = {int(k): v for k, v in poi_meta.items()}
    with open(meta_json, 'w', encoding='utf-8') as f:
        json.dump(json_serializable, f, ensure_ascii=False, indent=2)

    return poi_meta


if __name__ == '__main__':

    dataset_name = 'Yelp'
    poi_meta = extract_and_save_poi_metadata(dataset_name)
    print(f"Extracted and saved metadata for {len(poi_meta)} POIs")