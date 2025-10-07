import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import time
# import dgl

try:
    import ipdb
except:
    pass

def set_seeds(seed):
    """
    Sets the seed for various random number generators to ensure reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    # dgl.seed(seed)


def save_model(model, i, save_dir, optimizer=None, scheduler=None):
    """
    Saves the current model state along with its optimizer and scheduler states.
    The function checks if both an optimizer and a scheduler are provided.
    If they are, it saves the model state, optimizer state, and scheduler state together.
    If not, it only saves the model state. The saved file is named 'model_{i}.xhr',
    where {i} is replaced by the provided identifier.
    """
    if optimizer is not None and scheduler is not None:
        torch.save({
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict()
            }, os.path.join(save_dir, 'model_{}.xhr'.format(i)))
    else:
        torch.save({
            "state_dict":model.state_dict(),
            }, os.path.join(save_dir, 'model_{}.xhr'.format(i)))

def path_exist(path):
    """
    Checks if a directory exists, and if not, creates it.
    This function first checks if the directory specified by 'path' exists.
    If the directory does not exist, it creates the directory along with any
    necessary intermediate directories.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def filt_params(named_params, filt_key):
    """
    Filters parameters based on a specified key.
    Returns:
        list
    This function iterates through 'named_params', filtering out and returning
    only those parameters whose names contain the specified 'filt_key'.
    """
    filted = []
    for name, par in named_params:
        if filt_key in name:
            filted.append(par)
    return filted

def delete_models(epochs, base_path):
    """
    Deletes model files corresponding to specific training epochs.
    For each epoch number in 'epochs', this function constructs the file name of
    the corresponding saved model and deletes it from the file system.
    """
    for e in epochs:
        os.remove(os.path.join(base_path, "model_{}.xhr".format(e)))

class Logger(object):
    """
    Logger class for recording training processes and results.

    This class provides functionalities for logging messages both to the console
    and to a file, with time stamps included for each entry. It's useful for
    tracking the progress and results of machine learning experiments.

    Attributes:
        log_file: A file object for writing logs to a file.
        is_write_file (bool): Determines whether to write logs to a file.
    """

    def __init__(self, log_path, name, seed, is_write_file=True):
        cur_time = time.strftime("%m-%d-%H:%M", time.localtime())
        self.is_write_file = is_write_file
        if self.is_write_file:
            self.log_file = open(os.path.join(log_path, "%s %s(%d).log" % (cur_time, name, seed)), 'w')
    
    def log(self, log_str):
        """
        Logs a given string with a time stamp.
        """
        out_str = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {log_str}"
        print(out_str)
        if self.is_write_file:
            self.log_file.write(out_str+'\n')
            self.log_file.flush()
    
    def close_log(self):
        """
        Closes the log file if file logging is enabled.
        """
        if self.is_write_file:
            self.log_file.close()


def cuda():
    """
    Checks if CUDA is available for PyTorch operations.
    Returns:
        bool
    This function is a utility to quickly check the availability of CUDA, which is used for GPU-based computations.
    """
    return torch.cuda.is_available()
    #return False


def collate_fn(batch):
    """
    Custom collation function for batching data in a DataLoader.
    This function is used to process a batch of data items and ensure they are in a consistent format,
    suitable for model training or evaluation.
    Returns:
        tuple
    """
    uid, ori_ck, dst_ck, masked_dst_ck, o_hour, d_hour, masked_d_h, ori_t, dst_t, ori_l, dst_l, ori_rg, dst_rg = zip(*batch)

    pad_ori_ck = pad_sequence(ori_ck, batch_first=True)
    pad_dst_ck = pad_sequence(dst_ck, batch_first=True)
    pad_masked_dst_ck = pad_sequence(masked_dst_ck, batch_first=True)
    pad_o_hour = pad_sequence(o_hour, batch_first=True)
    pad_d_hour = pad_sequence(d_hour, batch_first=True)
    pad_masked_d_hour = pad_sequence(masked_d_h, batch_first=True)
    pad_ori_t = pad_sequence(ori_t, batch_first=True)
    pad_ori_l = pad_sequence(ori_l, batch_first=True)
    pad_dst_t = pad_sequence(dst_t, batch_first=True)
    pad_dst_l = pad_sequence(dst_l, batch_first=True)
    ori_rg = torch.LongTensor(ori_rg)
    dst_rg = torch.LongTensor(dst_rg)
    uid = torch.LongTensor(uid)
    # 为 ori_ck 生成 pad mask：有效数据为 True，padding 为 False
    lens_ori = torch.tensor([len(seq) for seq in ori_ck], dtype=torch.long)
    max_len_ori = pad_ori_ck.size(1)
    ori_pad = torch.arange(max_len_ori).unsqueeze(0).expand(len(ori_ck), max_len_ori) < lens_ori.unsqueeze(1)
    # 为 AGG token 位置增加一列（设为 True）
    ori_pad = torch.cat([ori_pad, torch.ones(len(ori_ck), 1, dtype=torch.bool)], dim=1)

    # 同样，为 dst_ck 生成 pad mask
    lens_dst = torch.tensor([len(seq) for seq in dst_ck], dtype=torch.long)
    max_len_dst = pad_dst_ck.size(1)
    dst_pad = torch.arange(max_len_dst).unsqueeze(0).expand(len(dst_ck), max_len_dst) < lens_dst.unsqueeze(1)
    # # 如果你也想给 dst_ck 增加 AGG token 位置
    dst_pad = torch.cat([dst_pad, torch.ones(len(dst_ck), 1, dtype=torch.bool)], dim=1)

    return uid, pad_ori_ck, pad_dst_ck, pad_masked_dst_ck, pad_o_hour, pad_d_hour, pad_masked_d_hour, pad_ori_t, pad_dst_t, pad_ori_l, pad_dst_l, ori_pad, dst_pad, ori_rg, dst_rg
