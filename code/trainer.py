import numbers
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import metrics
from cross_city_profile import (
    build_cross_city_profile,
    resolve_model_stage,
    write_cross_city_profile,
)
from utils import save_model

try:
    from tqdm import tqdm
    import ipdb
except:
    pass

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
            batch_similarity[batch, middle_index, :(top_p_indices+1)] = \
                F.softmax(batch_similarity[batch, middle_index, :(top_p_indices+1)] * confidence, dim=0)

            batch_similarity[batch, middle_index, (top_p_indices+1):] = torch.tensor(0)

            batch_probability_list = batch_similarity[batch, middle_index].tolist()
            nonzero_probability_list = [x for x in batch_probability_list if x != 0]

            new_top_p_index = random_choice_by_probability(nonzero_probability_list)
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_p_index]

    return top_candidates  # [b,l]
def train_single_phase(model, train_loader, valid_loader, test_loader, args, logger):
    """
    Train the model for a single phase.
    Returns:
        str/int
    """
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    stopping_dict = defaultdict(float)
    stopping_dict['best_RR'] = float('-inf')
    flag = True

    from datasets import load_from_disk
    text_dataset = load_from_disk(args.dataset_path)
    prompts, references = [item['prompt'] for item in text_dataset], [item['reference'] for item in text_dataset]

    # Load POI metadata (used for geographic/category/region metrics)
    poi_meta = None
    try:
        with open(f'../{args.dataset_name}/poi_meta.pkl', 'rb') as f:
            poi_meta = pickle.load(f)
    except Exception as _e:
        logger.log(f"[warn] Failed to load poi_meta.pkl: {_e}")

    for e in range(args.epoch):

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
                # Extract references for each uid; convert the tensor to a Python list for indexing
                batch_messages = [references[uid_item.item()] for uid_item in uid]
            else:
                # Extract prompts for each uid; convert the tensor to a Python list for indexing
                batch_messages = [prompts[uid_item.item()] for uid_item in uid]
            # Keep messages as strings for now; tokenization is handled inside the model as needed
            messages = batch_messages
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
            loss = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg,
                             target_seq=d_ck)
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
            batch_hit = []
            batch_recall = []
            batch_lev = []
            batch_dtw = []
            batch_cat = []
            batch_region = []


            for b, (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg,
                    d_rg) in enumerate(valid_loader):
                if args.use_target_llm:
                    # Extract references for each uid; convert the tensor to a Python list for indexing
                    batch_messages = [references[uid_item.item()] for uid_item in uid]
                else:
                    # Extract prompts for each uid; convert the tensor to a Python list for indexing
                    batch_messages = [prompts[uid_item.item()] for uid_item in uid]
                # Keep messages as strings for now; tokenization is handled inside the model as needed
                messages = batch_messages
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

                predicted_ids = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                        d_rg, target_seq=None)

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

                    # Compute additional metrics (including start/end elements)
                    inner_pred = sample_pred[1:-1]
                    inner_target = sample_target[1:-1]
                    # Metrics for inner sequence elements
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

                                if not np.isnan(cat_cons): batch_cat.append(cat_cons)
                                if not np.isnan(reg_match): batch_region.append(reg_match)

                        except Exception as _me:
                            logger.log(f"[warn] metric calc error(inner): {_me}")



            # Aggregate the additional metrics
            hit_mean = np.mean(batch_hit) if batch_hit else float('nan')
            recall_mean = np.mean(batch_recall) if batch_recall else float('nan')
            lev_mean = np.mean(batch_lev) if batch_lev else float('nan')
            dtw_mean = np.mean(batch_dtw) if batch_dtw else float('nan')
            cat_mean = np.mean(batch_cat) if batch_cat else float('nan')
            region_mean = np.mean(batch_region) if batch_region else float('nan')

            logger.log(f"[val] Epoch {e}/{args.epoch - 1} | Hit: {hit_mean:.4f} Recall: {recall_mean:.4f} Lev: {lev_mean:.4f} DTW: {dtw_mean:.4f} Cat: {cat_mean:.4f} Region: {region_mean:.4f}")

        get_test_result(model, test_loader, args, logger, prompts, references, poi_meta)

        # early stop
        # Use the recall_mean for early stopping


        if flag:
            if recall_mean > stopping_dict['best_RR']:
                stopping_dict['best_RR'] = recall_mean
                stopping_dict['RR_epoch'] = 0
                stopping_dict['best_epoch'] = e
                if args.best_save:
                    save_model(model, "best", args.save_path, optimizer, scheduler)
            else:
                stopping_dict['RR_epoch'] += 1

            if stopping_dict['RR_epoch'] >= args.stop_epoch:
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


def get_test_result(model, test_loader, args, logger, prompts, references, poi_meta=None):

    # Collect metrics for evaluation
    batch_hit = []
    batch_recall = []
    batch_lev = []
    batch_dtw = []
    batch_geo = []
    batch_cat = []
    batch_region = []
    batch_path_err = []
    batch_centroid = []

    for b, (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in tqdm(
            enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):
        if args.use_target_llm:
            # Extract references for each uid; convert the tensor to a Python list for indexing
            batch_messages = [references[uid_item.item()] for uid_item in uid]
        else:
            # Extract prompts for each uid; convert the tensor to a Python list for indexing
            batch_messages = [prompts[uid_item.item()] for uid_item in uid]
        # Keep messages as strings for now; tokenization is handled inside the model as needed
        messages = batch_messages
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
        predicted_ids = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg,
                                target_seq=None)

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


    hit_mean = np.mean(batch_hit) if batch_hit else float('nan')
    recall_mean = np.mean(batch_recall) if batch_recall else float('nan')
    lev_mean = np.mean(batch_lev) if batch_lev else float('nan')
    dtw_mean = np.mean(batch_dtw) if batch_dtw else float('nan')
    geo_mean = np.mean(batch_geo) if batch_geo else float('nan')
    cat_mean = np.mean(batch_cat) if batch_cat else float('nan')

    region_mean = np.mean(batch_region) if batch_region else float('nan')
    path_err_mean = np.mean(batch_path_err) if batch_path_err else float('nan')
    centroid_mean = np.mean(batch_centroid) if batch_centroid else float('nan')

    logger.log(f"[Test] Hit: {hit_mean:.4f} Recall: {recall_mean:.4f} Lev: {lev_mean:.4f} DTW: {dtw_mean:.4f} Geo: {geo_mean:.4f} Cat: {cat_mean:.4f} Region: {region_mean:.4f} PathErr: {path_err_mean:.4f} Centroid(km): {centroid_mean:.4f}")

def test(model, model_path, test_loader, args, logger, n_region, train_am=None, train_pm=None):
    """
    Test the model using the provided test dataset.
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)
    model.eval()
    batch_hit = []
    batch_recall = []
    batch_lev = []
    batch_dtw = []
    batch_geo = []
    batch_cat = []
    batch_region = []
    batch_path_err = []
    batch_centroid = []

    from datasets import load_from_disk
    text_dataset = load_from_disk(args.dataset_path)
    prompts, references = [item['prompt'] for item in text_dataset], [item['reference'] for item in text_dataset]

    # Load POI metadata (used for geographic/category/region metrics)
    poi_meta = None
    try:
        with open(f'../{args.dataset_name}/poi_meta.pkl', 'rb') as f:
            poi_meta = pickle.load(f)
    except Exception as _e:
        logger.log(f"[warn] Failed to load poi_meta.pkl: {_e}")

    base_dataset = test_loader.dataset
    while hasattr(base_dataset, "dataset"):
        base_dataset = base_dataset.dataset
    region_names = {
        index: name for name, index in getattr(base_dataset, "region_idx", {}).items()
    }
    export_directory = getattr(args, "cross_city_export_dir", None)
    model_stage = resolve_model_stage(
        use_llm=args.use_llm,
        use_lora=args.use_lora,
        secondary_lora_path=args.lora_path2,
        use_vllm=getattr(args, "use_vllm", False),
    )
    style_checkpoint = None
    if model_stage == "base":
        style_checkpoint = args.llm_model_path
    elif model_stage == "sft":
        style_checkpoint = args.lora_path
    elif model_stage == "rl":
        style_checkpoint = args.lora_path2

    for b, (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in tqdm(
            enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):
        if args.use_target_llm:
            # Extract references for each uid; convert the tensor to a Python list for indexing
            batch_messages = [references[uid_item.item()] for uid_item in uid]
        else:
            # Extract prompts for each uid; convert the tensor to a Python list for indexing
            batch_messages = [prompts[uid_item.item()] for uid_item in uid]
        # Keep messages as strings for now; tokenization is handled inside the model as needed
        messages = batch_messages
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
        predicted_ids = model(uid, messages, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg, d_rg,
                                target_seq=None)
        generated_styles = getattr(model, "last_generated_texts", [])
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

            if export_directory:
                style_source = "unavailable"
                travel_style = None
                if i < len(generated_styles):
                    travel_style = generated_styles[i]
                    style_source = (
                        "destination_reference"
                        if args.use_target_llm
                        else "model_generated"
                    )
                profile = build_cross_city_profile(
                    user_id=uid[i].cpu().item(),
                    dataset_name=args.dataset_name,
                    origin_city=region_names.get(o_rg[i].cpu().item(), "Unknown"),
                    destination_city=region_names.get(d_rg[i].cpu().item(), "Unknown"),
                    model_stage=model_stage,
                    predicted_poi_ids=sample_pred.tolist(),
                    poi_meta=poi_meta or {},
                    travel_style=travel_style,
                    style_source=style_source,
                    sequence_checkpoint=model_path,
                    style_checkpoint=style_checkpoint,
                )
                output_path = os.path.join(
                    export_directory, f"user_{uid[i].cpu().item()}.json"
                )
                write_cross_city_profile(profile, output_path)

            # If the sample length is greater than 1, perform alteration to keep the first and last elements unchanged
            if sample_target.numel() > 1:
                alt_sample_pred = torch.cat((sample_target[:1], sample_pred[1:-1], sample_target[-1:]),
                                            dim=0)
            else:
                alt_sample_pred = sample_pred

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


    hit_mean = np.mean(batch_hit) if batch_hit else float('nan')
    recall_mean = np.mean(batch_recall) if batch_recall else float('nan')
    lev_mean = np.mean(batch_lev) if batch_lev else float('nan')
    dtw_mean = np.mean(batch_dtw) if batch_dtw else float('nan')
    geo_mean = np.mean(batch_geo) if batch_geo else float('nan')
    cat_mean = np.mean(batch_cat) if batch_cat else float('nan')

    region_mean = np.mean(batch_region) if batch_region else float('nan')
    path_err_mean = np.mean(batch_path_err) if batch_path_err else float('nan')
    centroid_mean = np.mean(batch_centroid) if batch_centroid else float('nan')

    logger.log(f"[test] Hit: {hit_mean:.4f} Recall: {recall_mean:.4f} Lev: {lev_mean:.4f} DTW: {dtw_mean:.4f} Geo: {geo_mean:.4f} Cat: {cat_mean:.4f} Region: {region_mean:.4f} PathErr: {path_err_mean:.4f} Centroid(km): {centroid_mean:.4f}")
