# -*- coding: utf-8 -*-
from email import message
from pyexpat.errors import messages
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from collections import namedtuple, defaultdict, Counter
import numpy as np
import os
import sys

from copy import copy

from utils import *
import metrics

import pickle
import numbers

try:
    from tqdm import tqdm
    import ipdb
except:
    pass


def Trans_train(args, dataloader, model, opt):
    """
    Function for training the model with TransE, TransR, or SEEK algorithms.
    Returns:
        float
    """
    model.train()
    kgdataset = dataloader
    kgloader = DataLoader(kgdataset, batch_size=2048, drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(args.device)
        relations = data[1].to(args.device)
        pos_tails = data[2].to(args.device)
        neg_tails = data[3].to(args.device)
        if args.trans == 'seek':
            kg_batch_loss = model.calc_kg_loss_SEEK(heads, relations, pos_tails, neg_tails)
        if args.trans == 'transr':
            kg_batch_loss = model.calc_kg_loss_transR(heads, relations, pos_tails, neg_tails)
        if args.trans == 'transe':
            kg_batch_loss = model.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()


# ===================== decoding methods =========================== #
# Advanced-Greedy:
def find_duplicates_and_indices(input_tensor):
    duplicates_dict = {}
    input_tensor = input_tensor.tolist()
    for index, value in enumerate(input_tensor):
        if input_tensor.count(value) > 1:
            if value not in duplicates_dict:
                duplicates_dict[value] = [index, ]
            else:
                duplicates_dict[value] += [index, ]

    return duplicates_dict


def advanced_greedy_recommendation(batch_candidate, batch_similarity):
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        if len(top_candidates[batch]) == len(np.unique(top_candidates[batch])):
            pass
        else:

            position_top_k = [0] * top_candidates.shape[1]

            while len(top_candidates[batch]) != len(np.unique(top_candidates[batch])):

                repetition_dict = find_duplicates_and_indices(top_candidates[batch])

                for key in repetition_dict.keys():
                    confidence_list = []
                    for item in repetition_dict[key]:
                        confidence = batch_similarity[batch, item, position_top_k[item]].item()
                        confidence_list.append(confidence)

                    max_item = repetition_dict[key][confidence_list.index(max(confidence_list))]
                    left_item_list = list(filter(lambda x: x != max_item, repetition_dict[key]))
                    for left_item in left_item_list:
                        position_top_k[left_item] += 1
                        top_candidates[batch, left_item] = batch_candidate[batch, left_item, position_top_k[left_item]]

    return top_candidates


# Top-N and Top-NP method:
def random_choice_by_probability(probability_list):
    cumulative_probabilities = []
    cumulative_prob = 0
    for prob in probability_list:
        cumulative_prob += prob
        cumulative_probabilities.append(cumulative_prob)

    random_number = random.random()

    for i, cumulative_prob in enumerate(cumulative_probabilities):
        if random_number <= cumulative_prob:
            return i


def select_top_p_indices(probabilities, threshold=0.8):
    sorted_indices = np.argsort(probabilities)[::-1]  # re-order the probability
    cumulative_prob = 0.0
    selected_indices = []

    for idx in sorted_indices:
        cumulative_prob += probabilities[idx]
        selected_indices.append(idx)
        if cumulative_prob >= threshold:
            break

    return selected_indices[-1]


def top_n_recommendation(batch_candidate, batch_similarity, confidence=1):
    # the top_n method to recommend trajectory
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        for middle_index in range(batch_candidate.shape[1]):
            # print(batch_similarity[batch, middle_index])
            batch_similarity[batch, middle_index] = F.softmax(batch_similarity[batch, middle_index] * confidence, dim=0)
            # print(batch_similarity[batch, middle_index])
            new_top_k_index = random_choice_by_probability(batch_similarity[batch, middle_index].tolist())
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_k_index]

    return top_candidates  # [b,l]


def top_np_recommendation(batch_candidate, batch_similarity, confidence=0.5, threshold=0.8):
    # the top_np method to recommend trajectory
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        for middle_index in range(batch_candidate.shape[1]):
            batch_similarity[batch, middle_index] = F.softmax(batch_similarity[batch, middle_index] * confidence, dim=0)

            top_p_indices = select_top_p_indices(batch_similarity[batch, middle_index].tolist(), threshold)
            batch_similarity[batch, middle_index, :(top_p_indices + 1)] = \
                F.softmax(batch_similarity[batch, middle_index, :(top_p_indices + 1)] * confidence, dim=0)

            batch_similarity[batch, middle_index, (top_p_indices + 1):] = torch.tensor(0)

            batch_probability_list = batch_similarity[batch, middle_index].tolist()
            nonzero_probability_list = [x for x in batch_probability_list if x != 0]

            new_top_p_index = random_choice_by_probability(nonzero_probability_list)
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_p_index]

    return top_candidates  # [b,l]


def ad_top_np_recommendation(batch_candidate, batch_similarity, confidence, threshold=0.8):
    # the top_np method to recommend trajectory
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()
    for batch in range(batch_candidate.shape[0]):
        for middle_index in range(batch_candidate.shape[1]):
            batch_similarity[batch, middle_index] = F.softmax(batch_similarity[batch, middle_index] *
                                                              confidence[middle_index], dim=0)
            top_p_indices = select_top_p_indices(batch_similarity[batch, middle_index].tolist(), threshold)
            batch_similarity[batch, middle_index, :(top_p_indices + 1)] = \
                F.softmax(batch_similarity[batch, middle_index, :(top_p_indices + 1)], dim=0)

            batch_similarity[batch, middle_index, (top_p_indices + 1):] = torch.tensor(0)

            batch_probability_list = batch_similarity[batch, middle_index].tolist()
            nonzero_probability_list = [x for x in batch_probability_list if x != 0]

            new_top_p_index = random_choice_by_probability(nonzero_probability_list)
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_p_index]

    return top_candidates


def train_single_phase(model, train_loader, valid_loader, test_loader, args, logger, kg=None, train_am=None, train_pm=None):
    """
    Train the model for a single phase.
    Returns:
        str/int
    """
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    stopping_dict = defaultdict(float)
    flag = True

    from datasets import load_from_disk
    text_dataset = load_from_disk(args.dataset_path)
    prompts, references = [item['prompt'] for item in text_dataset], [item['reference'] for item in text_dataset]

    # 读取 POI 元信息 (用于地理 / 类别 / 区域指标)
    poi_meta = None
    try:
        with open(f'../{args.dataset_name}/poi_meta.pkl', 'rb') as f:
            poi_meta = pickle.load(f)
    except Exception as _e:
        logger.log(f"[warn] 无法加载 poi_meta.pkl: {_e}")

    for e in range(args.epoch):
        # pre-training
        if args.kg and args.train_trans and args.model == 'SPOT-Trip':
            print("[KG_Trans]")
            trans_loss = Trans_train(args, kg, model, optimizer)
            print(f"trans Loss: {trans_loss:.3f}")

        model.train()  # train mode
        model.eval_p = None
        model.eval_r = None
        model.eval_p_big = None
        model.eval_r_big = None
        loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
        number = len(train_loader)
        for b, (
        uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in tqdm(
                enumerate(train_loader), total=len(train_loader)):
            print("batch: %d/%d" % (b, number), end='\r')
            if args.use_target_llm:
                # 提取uid对应的references，uid是tensor，需要转换为Python列表来索引prompts
                batch_messages = [references[uid_item.item()] for uid_item in uid]
            else:
                # 提取uid对应的messages，uid是tensor，需要转换为Python列表来索引prompts
                batch_messages = [prompts[uid_item.item()] for uid_item in uid]
            # 将messages转换为tensor并移到设备上
            # 这里假设prompts已经是字符串，需要根据实际情况进行tokenization
            messages = batch_messages  # 暂时保持为列表格式，具体tokenization在模型内部处理
            uid = uid.to(args.device)
            o_ck = o_ck.to(args.device)
            masked_d_ck = masked_d_ck.to(args.device)
            d_ck = d_ck.to(args.device)
            o_h = o_h.to(args.device)
            masked_d_h = masked_d_h.to(args.device)
            d_h = d_h.to(args.device)
            o_t = o_t.to(args.device)
            d_t = d_t.to(args.device)
            o_l = o_l.to(args.device)
            d_l = d_l.to(args.device)
            o_pad = o_pad.to(args.device)
            d_pad = d_pad.to(args.device)
            o_rg = o_rg.to(args.device)
            d_rg = d_rg.to(args.device)

            optimizer.zero_grad()
            if args.model == 'SPOT-Trip':
                loss = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg,
                             target_seq=d_ck)
            if args.model == 'AR-Trip':
                poi_output, loss = model(masked_d_ck, masked_d_h, train_am, train_pm, d_ck, d_rg)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            # time_loss_sum += time_loss.item()
            # torch.cuda.empty_cache()
        scheduler.step()

        logger.log("Epoch %d/%d : Train Loss %.10f" % (e, args.epoch - 1, loss_sum / (b + 1)))
        if e % args.save_step == 0 and not args.best_save:
            save_model(model, e, args.save_path, optimizer, scheduler)
        model.eval()

        if flag:
            batch_alt_f1 = []
            batch_alt_pairs_f1 = []
            batch_alt_all_f1 = []
            batch_alt_pairs_all_f1 = []
            # 新指标收集
            batch_hit = []
            batch_recall = []
            batch_lev = []
            batch_dtw = []
            batch_geo = []
            batch_cat = []
            batch_region = []
            batch_path_err = []
            batch_centroid = []
            # full 序列
            full_hit = []
            full_recall = []
            full_lev = []
            full_dtw = []
            full_geo = []
            full_cat = []
            full_region = []
            full_path_err = []
            full_centroid = []

            for b, (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg,
                    d_rg) in enumerate(valid_loader):
                if args.use_target_llm:
                    # 提取uid对应的references，uid是tensor，需要转换为Python列表来索引prompts
                    batch_messages = [references[uid_item.item()] for uid_item in uid]
                else:
                    # 提取uid对应的messages，uid是tensor，需要转换为Python列表来索引prompts
                    batch_messages = [prompts[uid_item.item()] for uid_item in uid]
                messages = batch_messages  # 暂时保持为列表格式，具体tokenization在模型内部处理
                uid = uid.to(args.device)
                o_ck = o_ck.to(args.device)
                masked_d_ck = masked_d_ck.to(args.device)
                d_ck = d_ck.to(args.device)
                o_h = o_h.to(args.device)
                masked_d_h = masked_d_h.to(args.device)
                d_h = d_h.to(args.device)
                o_t = o_t.to(args.device)
                d_t = d_t.to(args.device)
                o_l = o_l.to(args.device)
                d_l = d_l.to(args.device)
                o_pad = o_pad.to(args.device)
                d_pad = d_pad.to(args.device)
                o_rg = o_rg.to(args.device)
                d_rg = d_rg.to(args.device)
                if args.model == 'SPOT-Trip':
                    predicted_ids = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                          d_rg, target_seq=None)
                    # Process each sample in the batch separately
                elif args.model == 'AR-Trip':
                    poi_output, _ = model(masked_d_ck, masked_d_h, train_am, train_pm, d_ck, d_rg)
                    guidance_similarity_ratio, guidance_candidate_ids = torch.topk(poi_output,
                                                                                   k=poi_output.shape[1], dim=2)
                    predicted_ids = ad_top_np_recommendation(guidance_candidate_ids, guidance_similarity_ratio,
                                                             confidence=torch.tensor(args.confidence),
                                                             threshold=0.8)
                for i in range(predicted_ids.shape[0]):
                    # Extract the prediction and target for the current sample
                    sample_pred = predicted_ids[i].cpu()  # shape: [seq_len]
                    sample_target = d_ck[i].cpu()  # shape: [seq_len]

                    # Exclude padded values (assuming padding is represented by 0)
                    non_padded_indices = sample_target != 0
                    sample_pred = sample_pred[non_padded_indices]
                    sample_target = sample_target[non_padded_indices]

                    # If the sample length is greater than 1, perform alteration to keep the first and last elements unchanged
                    if sample_target.numel() > 1:
                        alt_sample_pred = torch.cat((sample_target[:1], sample_pred[1:-1], sample_target[-1:]),
                                                    dim=0)
                    else:
                        alt_sample_pred = sample_pred

                    # Calculate F1 score and pairs F1 score for the current sample
                    sample_f1 = metrics.f1_score(sample_target[1:-1], sample_pred[1:-1])
                    sample_pairs_f1 = metrics.pairs_f1_score(sample_target[1:-1], sample_pred[1:-1])
                    # # Calculate F1 score and pairs F1 score for the current sample
                    # sample_f1 = metrics.f1_score(sample_target, alt_sample_pred)
                    # sample_pairs_f1 = metrics.pairs_f1_score(sample_target, alt_sample_pred)
                    full_sample_f1 = metrics.f1_score(torch.cat([torch.tensor([-1]), sample_target[1:-1], torch.tensor([-2])]),
                                                      torch.cat([torch.tensor([-1]), sample_pred[1:-1], torch.tensor([-2])]))
                    full_sample_pairs_f1 = metrics.pairs_f1_score(
                        torch.cat([torch.tensor([-1]), sample_target[1:-1], torch.tensor([-2])]),
                        torch.cat([torch.tensor([-1]), sample_pred[1:-1], torch.tensor([-2])]))

                    batch_alt_f1.append(sample_f1)
                    batch_alt_pairs_f1.append(sample_pairs_f1)
                    batch_alt_all_f1.append(full_sample_f1)
                    batch_alt_pairs_all_f1.append(full_sample_pairs_f1)

                    # 计算新增指标 （包括首尾）
                    inner_pred = sample_pred[1:-1]
                    inner_target = sample_target[1:-1]
                    # 中间序列指标
                    if inner_target.numel() > 0:
                        try:
                            batch_hit.append(metrics.hit_rate(inner_pred, inner_target))
                            batch_recall.append(metrics.recall_rate(inner_pred, inner_target, unique=True))
                            batch_lev.append(metrics.levenshtein_distance(inner_pred, inner_target, normalize=True))
                            batch_dtw.append(metrics.dtw_distance(inner_pred, inner_target, normalize=True))
                            if poi_meta:
                                geo_err = metrics.average_geo_distance_error(inner_pred, inner_target, poi_meta)
                                cat_cons = metrics.category_consistency_rate(inner_pred, inner_target, poi_meta)
                                reg_match = metrics.region_match_rate(inner_pred, inner_target, poi_meta)
                                path_err = metrics.relative_path_distance_error(inner_pred, inner_target, poi_meta, normalize=True)
                                centroid_dist = metrics.centroid_geo_distance(inner_pred, inner_target, poi_meta)
                                if not np.isnan(geo_err): batch_geo.append(geo_err)
                                if not np.isnan(cat_cons): batch_cat.append(cat_cons)
                                if not np.isnan(reg_match): batch_region.append(reg_match)
                                if not np.isnan(path_err): batch_path_err.append(path_err)
                                if not np.isnan(centroid_dist): batch_centroid.append(centroid_dist)
                        except Exception as _me:
                            logger.log(f"[warn] metric calc error(inner): {_me}")
                    # full 序列指标 (包含首尾)
                    try:
                        # 使用 alt_sample_pred 计算包含首尾的 full-* 指标
                        full_hit.append(metrics.hit_rate(alt_sample_pred, sample_target))
                        full_recall.append(metrics.recall_rate(alt_sample_pred, sample_target, unique=True))
                        full_lev.append(metrics.levenshtein_distance(alt_sample_pred, sample_target, normalize=True))
                        full_dtw.append(metrics.dtw_distance(alt_sample_pred, sample_target, normalize=True))
                        if poi_meta:
                            g2 = metrics.average_geo_distance_error(alt_sample_pred, sample_target, poi_meta)
                            c2 = metrics.category_consistency_rate(alt_sample_pred, sample_target, poi_meta)
                            r2 = metrics.region_match_rate(alt_sample_pred, sample_target, poi_meta)
                            p2 = metrics.relative_path_distance_error(alt_sample_pred, sample_target, poi_meta, normalize=True)
                            cent2 = metrics.centroid_geo_distance(alt_sample_pred, sample_target, poi_meta)
                            if not np.isnan(g2): full_geo.append(g2)
                            if not np.isnan(c2): full_cat.append(c2)
                            if not np.isnan(r2): full_region.append(r2)
                            if not np.isnan(p2): full_path_err.append(p2)
                            if not np.isnan(cent2): full_centroid.append(cent2)
                    except Exception as _me:
                        logger.log(f"[warn] metric calc error(full): {_me}")

            alt_f1 = np.mean(batch_alt_f1)
            alt_pairs_f1 = np.mean(batch_alt_pairs_f1)
            alt_all_f1 = np.mean(batch_alt_all_f1)
            alt_pairs_all_f1 = np.mean(batch_alt_pairs_all_f1)
            # 新指标聚合
            hit_mean = np.mean(batch_hit) if batch_hit else float('nan')
            recall_mean = np.mean(batch_recall) if batch_recall else float('nan')
            lev_mean = np.mean(batch_lev) if batch_lev else float('nan')
            dtw_mean = np.mean(batch_dtw) if batch_dtw else float('nan')
            geo_mean = np.mean(batch_geo) if batch_geo else float('nan')
            cat_mean = np.mean(batch_cat) if batch_cat else float('nan')

            region_mean = np.mean(batch_region) if batch_region else float('nan')
            path_err_mean = np.mean(batch_path_err) if batch_path_err else float('nan')
            centroid_mean = np.mean(batch_centroid) if batch_centroid else float('nan')

            # full metrics
            full_hit_mean = np.mean(full_hit) if full_hit else float('nan')
            full_recall_mean = np.mean(full_recall) if full_recall else float('nan')
            full_lev_mean = np.mean(full_lev) if full_lev else float('nan')
            full_dtw_mean = np.mean(full_dtw) if full_dtw else float('nan')
            full_geo_mean = np.mean(full_geo) if full_geo else float('nan')
            full_cat_mean = np.mean(full_cat) if full_cat else float('nan')
            full_region_mean = np.mean(full_region) if full_region else float('nan')
            full_path_err_mean = np.mean(full_path_err) if full_path_err else float('nan')
            full_centroid_mean = np.mean(full_centroid) if full_centroid else float('nan')

            logger.log("[val] Epoch {}/{} F1: {:5.4f} Pairs-F1: {:5.4f} All-F1: {:5.4f} All-Pairs-F1: {:5.4f} | Hit: {} Recall: {} Lev: {} DTW: {} Geo: {} Cat: {} Region: {} PathErr: {} Centroid(km): {} | full-Hit: {} full-Recall: {} full-Lev: {} full-DTW: {} full-Geo: {} full-Cat: {} full-Region: {} full-PathErr: {} full-Centroid(km): {}".format(
                e, args.epoch - 1, alt_f1, alt_pairs_f1, alt_all_f1, alt_pairs_all_f1,
                _fmt(hit_mean), _fmt(recall_mean), _fmt(lev_mean), _fmt(dtw_mean), _fmt(geo_mean), _fmt(cat_mean), _fmt(region_mean), _fmt(path_err_mean), _fmt(centroid_mean),
                _fmt(full_hit_mean), _fmt(full_recall_mean), _fmt(full_lev_mean), _fmt(full_dtw_mean), _fmt(full_geo_mean), _fmt(full_cat_mean), _fmt(full_region_mean), _fmt(full_path_err_mean), _fmt(full_centroid_mean)))
        get_test_result(model, test_loader, args, logger, prompts, references, poi_meta, train_am, train_pm)

        # early stop
        if flag:
            if alt_f1 > stopping_dict['best_f1']:
                stopping_dict['best_f1'] = alt_f1
                stopping_dict['f1_epoch'] = 0
                stopping_dict['best_epoch'] = e
                # if args.best_save:
                #     save_model(model, "best", args.save_path, optimizer, scheduler)
            else:
                stopping_dict['f1_epoch'] += 1

            if alt_pairs_f1 > stopping_dict['best_pairs_f1']:
                stopping_dict['best_pairs_f1'] = alt_pairs_f1
                stopping_dict['pairs_f1_epoch'] = 0
                stopping_dict['best_epoch'] = e
                if args.best_save:
                    save_model(model, "best", args.save_path, optimizer, scheduler)
            else:
                stopping_dict['pairs_f1_epoch'] += 1

            logger.log("early stop: {}|{}".format(stopping_dict['f1_epoch'], stopping_dict["pairs_f1_epoch"]))

            if stopping_dict['f1_epoch'] >= args.stop_epoch and stopping_dict['pairs_f1_epoch'] >= args.stop_epoch:
                flag = False
                logger.log("early stopped! best epoch: {}".format(stopping_dict['best_epoch']))

                best_return = stopping_dict['best_epoch']

        if not flag:
            if args.best_save:
                return "best"
            else:
                return best_return

def _fmt(v):
    if v is None: return 'None'
    if isinstance(v, numbers.Number):
        if np.isnan(v):
            return 'nan'
        return f"{v:.4f}"
    return str(v)


def get_test_result(model, test_loader, args, logger, prompts, references, poi_meta=None, train_am=None, train_pm=None):
    batch_alt_f1 = []
    batch_alt_pairs_f1 = []
    batch_alt_all_f1 = []
    batch_alt_pairs_all_f1 = []
    # for the repetition
    repetition_list = []
    # 新指标收集
    batch_hit = []
    batch_recall = []
    batch_lev = []
    batch_dtw = []
    batch_geo = []
    batch_cat = []
    batch_region = []
    batch_path_err = []
    batch_centroid = []
    full_hit = []
    full_recall = []
    full_lev = []
    full_dtw = []
    full_geo = []
    full_cat = []
    full_region = []
    full_path_err = []
    full_centroid = []

    for b, (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in tqdm(
            enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):
        if args.use_target_llm:
            # 提取uid对应的references，uid是tensor，需要转换为Python列表来索引prompts
            batch_messages = [references[uid_item.item()] for uid_item in uid]
        else:
            # 提取uid对应的messages，uid是tensor，需要转换为Python列表来索引prompts
            batch_messages = [prompts[uid_item.item()] for uid_item in uid]
        messages = batch_messages  # 暂时保持为列表格式，具体tokenization在模型内部处理
        uid = uid.to(args.device)
        o_ck = o_ck.to(args.device)
        masked_d_ck = masked_d_ck.to(args.device)
        d_ck = d_ck.to(args.device)
        o_h = o_h.to(args.device)
        masked_d_h = masked_d_h.to(args.device)
        d_h = d_h.to(args.device)
        o_t = o_t.to(args.device)
        d_t = d_t.to(args.device)
        o_l = o_l.to(args.device)
        d_l = d_l.to(args.device)
        o_pad = o_pad.to(args.device)
        d_pad = d_pad.to(args.device)
        o_rg = o_rg.to(args.device)
        d_rg = d_rg.to(args.device)
        if args.model == 'SPOT-Trip':
            predicted_ids = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg,
                                  target_seq=None)
        # Process each sample in the batch separately
        elif args.model == 'AR-Trip':
            poi_output, _ = model(masked_d_ck, masked_d_h, train_am, train_pm, d_ck, d_rg)
            guidance_similarity_ratio, guidance_candidate_ids = torch.topk(poi_output,
                                                                           k=poi_output.shape[1], dim=2)
            predicted_ids = ad_top_np_recommendation(guidance_candidate_ids, guidance_similarity_ratio,
                                                     confidence=torch.tensor(args.confidence),
                                                     threshold=0.8)
        for i in range(predicted_ids.shape[0]):
            # Extract the prediction and target for the current sample
            sample_pred = predicted_ids[i].cpu()  # shape: [seq_len]
            sample_target = d_ck[i].cpu()  # shape: [seq_len]
            print("UIDs:", uid[i].cpu().item())
            print("Predicted IDs:", sample_pred.tolist())
            print("Target IDs:", sample_target.tolist())

            # Exclude padded values (assuming padding is represented by 0)
            non_padded_indices = sample_target != 0
            sample_pred = sample_pred[non_padded_indices]
            sample_target = sample_target[non_padded_indices]

            # If the sample length is greater than 1, perform alteration to keep the first and last elements unchanged
            if sample_target.numel() > 1:
                alt_sample_pred = torch.cat((sample_target[:1], sample_pred[1:-1], sample_target[-1:]),
                                            dim=0)
            else:
                alt_sample_pred = sample_pred

            # Calculate F1 score and pairs F1 score for the current sample
            sample_f1 = metrics.f1_score(sample_target[1:-1], sample_pred[1:-1])
            sample_pairs_f1 = metrics.pairs_f1_score(sample_target[1:-1], sample_pred[1:-1])
            # # Calculate F1 score and pairs F1 score for the current sample
            # sample_f1 = metrics.f1_score(sample_target, alt_sample_pred)
            # sample_pairs_f1 = metrics.pairs_f1_score(sample_target, alt_sample_pred)
            full_sample_f1 = metrics.f1_score(torch.cat([torch.tensor([-1]), sample_target[1:-1], torch.tensor([-2])]),
                                              torch.cat([torch.tensor([-1]), sample_pred[1:-1], torch.tensor([-2])]))
            full_sample_pairs_f1 = metrics.pairs_f1_score(
                torch.cat([torch.tensor([-1]), sample_target[1:-1], torch.tensor([-2])]),
                torch.cat([torch.tensor([-1]), sample_pred[1:-1], torch.tensor([-2])]))

            batch_alt_f1.append(sample_f1)
            batch_alt_pairs_f1.append(sample_pairs_f1)
            batch_alt_all_f1.append(full_sample_f1)
            batch_alt_pairs_all_f1.append(full_sample_pairs_f1)

            inner_pred = sample_pred[1:-1]
            inner_target = sample_target[1:-1]
            if inner_target.numel() > 0:
                try:
                    batch_hit.append(metrics.hit_rate(inner_pred, inner_target))
                    batch_recall.append(metrics.recall_rate(inner_pred, inner_target, unique=True))
                    batch_lev.append(metrics.levenshtein_distance(inner_pred, inner_target, normalize=True))
                    batch_dtw.append(metrics.dtw_distance(inner_pred, inner_target, normalize=True))
                    if poi_meta:
                        geo_err = metrics.average_geo_distance_error(inner_pred, inner_target, poi_meta)
                        cat_cons = metrics.category_consistency_rate(inner_pred, inner_target, poi_meta)
                        reg_match = metrics.region_match_rate(inner_pred, inner_target, poi_meta)
                        path_err = metrics.relative_path_distance_error(inner_pred, inner_target, poi_meta, normalize=True)
                        centroid_dist = metrics.centroid_geo_distance(inner_pred, inner_target, poi_meta)
                        if not np.isnan(geo_err): batch_geo.append(geo_err)
                        if not np.isnan(cat_cons): batch_cat.append(cat_cons)
                        if not np.isnan(reg_match): batch_region.append(reg_match)
                        if not np.isnan(path_err): batch_path_err.append(path_err)
                        if not np.isnan(centroid_dist): batch_centroid.append(centroid_dist)
                except Exception as _me:
                    logger.log(f"[warn] metric calc error (test-inner): {_me}")
            try:
                # 使用 alt_sample_pred 计算 full-* 指标
                full_hit.append(metrics.hit_rate(alt_sample_pred, sample_target))
                full_recall.append(metrics.recall_rate(alt_sample_pred, sample_target, unique=True))
                full_lev.append(metrics.levenshtein_distance(alt_sample_pred, sample_target, normalize=True))
                full_dtw.append(metrics.dtw_distance(alt_sample_pred, sample_target, normalize=True))
                if poi_meta:
                    g2 = metrics.average_geo_distance_error(alt_sample_pred, sample_target, poi_meta)
                    c2 = metrics.category_consistency_rate(alt_sample_pred, sample_target, poi_meta)
                    r2 = metrics.region_match_rate(alt_sample_pred, sample_target, poi_meta)
                    p2 = metrics.relative_path_distance_error(alt_sample_pred, sample_target, poi_meta, normalize=True)
                    cent2 = metrics.centroid_geo_distance(alt_sample_pred, sample_target, poi_meta)
                    if not np.isnan(g2): full_geo.append(g2)
                    if not np.isnan(c2): full_cat.append(c2)
                    if not np.isnan(r2): full_region.append(r2)
                    if not np.isnan(p2): full_path_err.append(p2)
                    if not np.isnan(cent2): full_centroid.append(cent2)
            except Exception as _me:
                logger.log(f"[warn] metric calc error (test-full): {_me}")

            repetition_ratio = metrics.count_adjacent_repetition_rate(alt_sample_pred)
            repetition_list.append(repetition_ratio)
            # torch.cuda.empty_cache()
    repetition = np.mean(repetition_list)
    alt_f1 = np.mean(batch_alt_f1)
    alt_pairs_f1 = np.mean(batch_alt_pairs_f1)
    hit_mean = np.mean(batch_hit) if batch_hit else float('nan')
    recall_mean = np.mean(batch_recall) if batch_recall else float('nan')
    lev_mean = np.mean(batch_lev) if batch_lev else float('nan')
    dtw_mean = np.mean(batch_dtw) if batch_dtw else float('nan')
    geo_mean = np.mean(batch_geo) if batch_geo else float('nan')
    cat_mean = np.mean(batch_cat) if batch_cat else float('nan')

    region_mean = np.mean(batch_region) if batch_region else float('nan')
    path_err_mean = np.mean(batch_path_err) if batch_path_err else float('nan')
    centroid_mean = np.mean(batch_centroid) if batch_centroid else float('nan')
    full_hit_mean = np.mean(full_hit) if full_hit else float('nan')
    full_recall_mean = np.mean(full_recall) if full_recall else float('nan')
    full_lev_mean = np.mean(full_lev) if full_lev else float('nan')
    full_dtw_mean = np.mean(full_dtw) if full_dtw else float('nan')
    full_geo_mean = np.mean(full_geo) if full_geo else float('nan')
    full_cat_mean = np.mean(full_cat) if full_cat else float('nan')
    full_region_mean = np.mean(full_region) if full_region else float('nan')
    full_path_err_mean = np.mean(full_path_err) if full_path_err else float('nan')
    full_centroid_mean = np.mean(full_centroid) if full_centroid else float('nan')

    logger.log("[test-general] F1: {:5.4f} Pairs-F1: {:5.4f} Repetition: {:5.4f} All-F1: {:5.4f} All-Pairs-F1: {:5.4f} | Hit: {} Recall: {} Lev: {} DTW: {} Geo: {} Cat: {} Region: {} PathErr: {} Centroid(km): {} | full-Hit: {} full-Recall: {} full-Lev: {} full-DTW: {} full-Geo: {} full-Cat: {} full-Region: {} full-PathErr: {} full-Centroid(km): {}".format(
        alt_f1, alt_pairs_f1, repetition, np.mean(batch_alt_all_f1), np.mean(batch_alt_pairs_all_f1),
        _fmt(hit_mean), _fmt(recall_mean), _fmt(lev_mean), _fmt(dtw_mean), _fmt(geo_mean), _fmt(cat_mean), _fmt(region_mean), _fmt(path_err_mean), _fmt(centroid_mean),
        _fmt(full_hit_mean), _fmt(full_recall_mean), _fmt(full_lev_mean), _fmt(full_dtw_mean), _fmt(full_geo_mean), _fmt(full_cat_mean), _fmt(full_region_mean), _fmt(full_path_err_mean), _fmt(full_centroid_mean)))

def test(model, model_path, test_loader, args, logger, n_region, train_am=None, train_pm=None):
    """
    Test the model using the provided test dataset.
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)
    model.eval()
    batch_alt_f1 = []
    batch_alt_pairs_f1 = []
    batch_alt_all_f1 = []
    batch_alt_pairs_all_f1 = []
    # for the repetition
    repetition_list = []
    # 新指标收集
    batch_hit = []
    batch_recall = []
    batch_lev = []
    batch_dtw = []
    batch_geo = []
    batch_cat = []
    batch_region = []
    batch_path_err = []
    batch_centroid = []
    full_hit = []
    full_recall = []
    full_lev = []
    full_dtw = []
    full_geo = []
    full_cat = []
    full_region = []
    full_path_err = []
    full_centroid = []

    from datasets import load_from_disk
    text_dataset = load_from_disk(args.dataset_path)
    prompts, references = [item['prompt'] for item in text_dataset], [item['reference'] for item in text_dataset]

    # 读取 POI 元信息 (用于地理 / 类别 / 区域指标)
    poi_meta = None
    try:
        with open(f'../{args.dataset_name}/poi_meta.pkl', 'rb') as f:
            poi_meta = pickle.load(f)
    except Exception as _e:
        logger.log(f"[warn] 无法加载 poi_meta.pkl: {_e}")

    for b, (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in tqdm(
            enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):
        if args.use_target_llm:
            # 提取uid对应的references，uid是tensor，需要转换为Python列表来索引prompts
            batch_messages = [references[uid_item.item()] for uid_item in uid]
        else:
            # 提取uid对应的messages，uid是tensor，需要转换为Python列表来索引prompts
            batch_messages = [prompts[uid_item.item()] for uid_item in uid]
        messages = batch_messages  # 暂时保持为列表格式，具体tokenization在模型内部处理
        uid = uid.to(args.device)
        o_ck = o_ck.to(args.device)
        masked_d_ck = masked_d_ck.to(args.device)
        d_ck = d_ck.to(args.device)
        o_h = o_h.to(args.device)
        masked_d_h = masked_d_h.to(args.device)
        d_h = d_h.to(args.device)
        o_t = o_t.to(args.device)
        d_t = d_t.to(args.device)
        o_l = o_l.to(args.device)
        d_l = d_l.to(args.device)
        o_pad = o_pad.to(args.device)
        d_pad = d_pad.to(args.device)
        o_rg = o_rg.to(args.device)
        d_rg = d_rg.to(args.device)
        if args.model == 'SPOT-Trip':
            predicted_ids = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg,
                                  target_seq=None)
        # Process each sample in the batch separately
        for i in range(predicted_ids.shape[0]):
            # Extract the prediction and target for the current sample
            sample_pred = predicted_ids[i].cpu()  # shape: [seq_len]
            sample_target = d_ck[i].cpu()  # shape: [seq_len]
            print("UIDs:", uid[i].cpu().item())
            print("Predicted IDs:", sample_pred.tolist())
            print("Target IDs:", sample_target.tolist())

            # Exclude padded values (assuming padding is represented by 0)
            non_padded_indices = sample_target != 0
            sample_pred = sample_pred[non_padded_indices]
            sample_target = sample_target[non_padded_indices]

            # If the sample length is greater than 1, perform alteration to keep the first and last elements unchanged
            if sample_target.numel() > 1:
                alt_sample_pred = torch.cat((sample_target[:1], sample_pred[1:-1], sample_target[-1:]),
                                            dim=0)
            else:
                alt_sample_pred = sample_pred

            # Calculate F1 score and pairs F1 score for the current sample
            sample_f1 = metrics.f1_score(sample_target[1:-1], sample_pred[1:-1])
            sample_pairs_f1 = metrics.pairs_f1_score(sample_target[1:-1], sample_pred[1:-1])
            # # Calculate F1 score and pairs F1 score for the current sample
            # sample_f1 = metrics.f1_score(sample_target, alt_sample_pred)
            # sample_pairs_f1 = metrics.pairs_f1_score(sample_target, alt_sample_pred)
            full_sample_f1 = metrics.f1_score(torch.cat([torch.tensor([-1]), sample_target[1:-1], torch.tensor([-2])]),
                                              torch.cat([torch.tensor([-1]), sample_pred[1:-1], torch.tensor([-2])]))
            full_sample_pairs_f1 = metrics.pairs_f1_score(
                torch.cat([torch.tensor([-1]), sample_target[1:-1], torch.tensor([-2])]),
                torch.cat([torch.tensor([-1]), sample_pred[1:-1], torch.tensor([-2])]))

            batch_alt_f1.append(sample_f1)
            batch_alt_pairs_f1.append(sample_pairs_f1)
            batch_alt_all_f1.append(full_sample_f1)
            batch_alt_pairs_all_f1.append(full_sample_pairs_f1)

            inner_pred = sample_pred[1:-1]
            inner_target = sample_target[1:-1]
            if inner_target.numel() > 0:
                try:
                    batch_hit.append(metrics.hit_rate(inner_pred, inner_target))
                    batch_recall.append(metrics.recall_rate(inner_pred, inner_target, unique=True))
                    batch_lev.append(metrics.levenshtein_distance(inner_pred, inner_target, normalize=True))
                    batch_dtw.append(metrics.dtw_distance(inner_pred, inner_target, normalize=True))
                    if poi_meta:
                        geo_err = metrics.average_geo_distance_error(inner_pred, inner_target, poi_meta)
                        cat_cons = metrics.category_consistency_rate(inner_pred, inner_target, poi_meta)
                        reg_match = metrics.region_match_rate(inner_pred, inner_target, poi_meta)
                        path_err = metrics.relative_path_distance_error(inner_pred, inner_target, poi_meta,
                                                                        normalize=True)
                        centroid_dist = metrics.centroid_geo_distance(inner_pred, inner_target, poi_meta)
                        if not np.isnan(geo_err): batch_geo.append(geo_err)
                        if not np.isnan(cat_cons): batch_cat.append(cat_cons)
                        if not np.isnan(reg_match): batch_region.append(reg_match)
                        if not np.isnan(path_err): batch_path_err.append(path_err)
                        if not np.isnan(centroid_dist): batch_centroid.append(centroid_dist)
                except Exception as _me:
                    logger.log(f"[warn] metric calc error (test-inner): {_me}")
            try:
                # 使用 alt_sample_pred 计算 full-* 指标
                full_hit.append(metrics.hit_rate(alt_sample_pred, sample_target))
                full_recall.append(metrics.recall_rate(alt_sample_pred, sample_target, unique=True))
                full_lev.append(metrics.levenshtein_distance(alt_sample_pred, sample_target, normalize=True))
                full_dtw.append(metrics.dtw_distance(alt_sample_pred, sample_target, normalize=True))
                if poi_meta:
                    g2 = metrics.average_geo_distance_error(alt_sample_pred, sample_target, poi_meta)
                    c2 = metrics.category_consistency_rate(alt_sample_pred, sample_target, poi_meta)
                    r2 = metrics.region_match_rate(alt_sample_pred, sample_target, poi_meta)
                    p2 = metrics.relative_path_distance_error(alt_sample_pred, sample_target, poi_meta, normalize=True)
                    cent2 = metrics.centroid_geo_distance(alt_sample_pred, sample_target, poi_meta)
                    if not np.isnan(g2): full_geo.append(g2)
                    if not np.isnan(c2): full_cat.append(c2)
                    if not np.isnan(r2): full_region.append(r2)
                    if not np.isnan(p2): full_path_err.append(p2)
                    if not np.isnan(cent2): full_centroid.append(cent2)
            except Exception as _me:
                logger.log(f"[warn] metric calc error (test-full): {_me}")

            repetition_ratio = metrics.count_adjacent_repetition_rate(alt_sample_pred)
            repetition_list.append(repetition_ratio)
            # torch.cuda.empty_cache()
    repetition = np.mean(repetition_list)
    alt_f1 = np.mean(batch_alt_f1)
    alt_pairs_f1 = np.mean(batch_alt_pairs_f1)
    hit_mean = np.mean(batch_hit) if batch_hit else float('nan')
    recall_mean = np.mean(batch_recall) if batch_recall else float('nan')
    lev_mean = np.mean(batch_lev) if batch_lev else float('nan')
    dtw_mean = np.mean(batch_dtw) if batch_dtw else float('nan')
    geo_mean = np.mean(batch_geo) if batch_geo else float('nan')
    cat_mean = np.mean(batch_cat) if batch_cat else float('nan')

    region_mean = np.mean(batch_region) if batch_region else float('nan')
    path_err_mean = np.mean(batch_path_err) if batch_path_err else float('nan')
    centroid_mean = np.mean(batch_centroid) if batch_centroid else float('nan')
    full_hit_mean = np.mean(full_hit) if full_hit else float('nan')
    full_recall_mean = np.mean(full_recall) if full_recall else float('nan')
    full_lev_mean = np.mean(full_lev) if full_lev else float('nan')
    full_dtw_mean = np.mean(full_dtw) if full_dtw else float('nan')
    full_geo_mean = np.mean(full_geo) if full_geo else float('nan')
    full_cat_mean = np.mean(full_cat) if full_cat else float('nan')
    full_region_mean = np.mean(full_region) if full_region else float('nan')
    full_path_err_mean = np.mean(full_path_err) if full_path_err else float('nan')
    full_centroid_mean = np.mean(full_centroid) if full_centroid else float('nan')

    logger.log(
        "[test-general] F1: {:5.4f} Pairs-F1: {:5.4f} Repetition: {:5.4f} All-F1: {:5.4f} All-Pairs-F1: {:5.4f} | Hit: {} Recall: {} Lev: {} DTW: {} Geo: {} Cat: {} Region: {} PathErr: {} Centroid(km): {} | full-Hit: {} full-Recall: {} full-Lev: {} full-DTW: {} full-Geo: {} full-Cat: {} full-Region: {} full-PathErr: {} full-Centroid(km): {}".format(
            alt_f1, alt_pairs_f1, repetition, np.mean(batch_alt_all_f1), np.mean(batch_alt_pairs_all_f1),
            _fmt(hit_mean), _fmt(recall_mean), _fmt(lev_mean), _fmt(dtw_mean), _fmt(geo_mean), _fmt(cat_mean),
            _fmt(region_mean), _fmt(path_err_mean), _fmt(centroid_mean),
            _fmt(full_hit_mean), _fmt(full_recall_mean), _fmt(full_lev_mean), _fmt(full_dtw_mean), _fmt(full_geo_mean),
            _fmt(full_cat_mean), _fmt(full_region_mean), _fmt(full_path_err_mean), _fmt(full_centroid_mean)))
