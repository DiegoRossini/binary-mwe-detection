from __future__ import annotations
from typing import List, Tuple, Dict, Any, Set
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Sentence, CoAMTokenDataset, collate_fn
from config import DEVICE, BATCH_SIZE, MAX_SEQ_LEN, MAX_MEMBER_LEN, WINDOW_MAX, THRESH_GRID


def reconstruct_candidates(sent: Sentence, start_scores: List[float], end_scores: List[float],
                           inside_scores: List[float], th_start: float, th_end: float,
                           th_inside: float, window_max: int = WINDOW_MAX,
                           max_member_len: int = MAX_MEMBER_LEN,
                           use_dependency: bool = True) -> List[Tuple[int, ...]]:
    n = len(sent.tokens)
    starts = [i for i, v in enumerate(start_scores) if v >= th_start]
    ends = [i for i, v in enumerate(end_scores) if v >= th_end]

    candidates = []
    for s in starts:
        for e in ends:
            if e <= s or (e - s + 1) > window_max:
                continue

            members = {s, e}
            for t in range(s + 1, e):
                if inside_scores[t] >= th_inside:
                    members.add(t)

            if use_dependency and len(members) < (e - s + 1) and sent.dep_distances:
                member_list = sorted(members)
                max_dep_dist = 0
                for i in range(len(member_list) - 1):
                    key = (member_list[i], member_list[i + 1])
                    dist = sent.dep_distances.get(key, 10)
                    max_dep_dist = max(max_dep_dist, dist)
                if max_dep_dist > 3:
                    continue

            if 2 <= len(members) <= max_member_len:
                candidates.append(tuple(sorted(members)))

    return list(set(candidates))


def evaluate_exact(model, dataset: CoAMTokenDataset, thresholds: Tuple[float, float, float],
                   device: str = DEVICE, batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
    model.eval()
    ts, te, ti = thresholds
    preds_map = {}

    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            chunk_feat = batch["chunk_feat"].to(device)

            start_p, end_p, inside_p, _ = model(ids, attn, chunk_feat)

            for bi, sent_idx in enumerate(batch["sent_idx"]):
                s = dataset.sents[sent_idx]
                words = [t.surface for t in s.tokens]
                enc = dataset.tokenizer(
                    words, is_split_into_words=True, truncation=True,
                    max_length=MAX_SEQ_LEN, add_special_tokens=True
                )
                word_ids = enc.word_ids()

                ws = [0.0] * len(s.tokens)
                we = [0.0] * len(s.tokens)
                wi = [0.0] * len(s.tokens)
                used = set()

                for ti2, wid in enumerate(word_ids):
                    if wid is None:
                        continue
                    if wid not in used:
                        ws[wid] = float(start_p[bi, ti2])
                        we[wid] = float(end_p[bi, ti2])
                        wi[wid] = float(inside_p[bi, ti2])
                        used.add(wid)

                cand = reconstruct_candidates(s, ws, we, wi, ts, te, ti)
                preds_map[s.sent_id] = cand

    tp = gold_total = pred_total = 0
    type_tp, type_fp, type_fn = Counter(), Counter(), Counter()
    cont_tp = cont_fp = cont_fn = cont_gold = cont_pred = 0
    disc_tp = disc_fp = disc_fn = disc_gold = disc_pred = 0

    for s in dataset.sents:
        gold_spans = set(m.tokens for m in s.mwes)
        pred_spans = set(preds_map.get(s.sent_id, []))
        gold_total += len(gold_spans)
        pred_total += len(pred_spans)
        inter = gold_spans & pred_spans
        tp += len(inter)

        gold_type_map = {m.tokens: m.type for m in s.mwes}

        for g in gold_spans:
            tlabel = gold_type_map[g] or "OTHER"
            if g in inter:
                type_tp[tlabel] += 1
            else:
                type_fn[tlabel] += 1

        for p in pred_spans:
            if p not in gold_spans:
                if gold_type_map:
                    tlabel = max(set(gold_type_map.values()), key=list(gold_type_map.values()).count)
                else:
                    tlabel = "OTHER"
                type_fp[tlabel] += 1

        for g in gold_spans:
            is_disc = any(b - a != 1 for a, b in zip(g, g[1:]))
            if is_disc:
                disc_gold += 1
                if g in inter:
                    disc_tp += 1
                else:
                    disc_fn += 1
            else:
                cont_gold += 1
                if g in inter:
                    cont_tp += 1
                else:
                    cont_fn += 1

        for p in pred_spans:
            is_disc = any(b - a != 1 for a, b in zip(p, p[1:]))
            if is_disc:
                disc_pred += 1
                if p not in gold_spans:
                    disc_fp += 1
            else:
                cont_pred += 1
                if p not in gold_spans:
                    cont_fp += 1

    precision = tp / pred_total if pred_total else 0
    recall = tp / gold_total if gold_total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    type_precision, type_recall, type_f1 = {}, {}, {}
    all_types = sorted(type_tp.keys() | type_fp.keys() | type_fn.keys())
    for t in all_types:
        tp_t, fp_t, fn_t = type_tp[t], type_fp[t], type_fn[t]
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
        type_precision[t], type_recall[t], type_f1[t] = prec_t, rec_t, f1_t

    cont_precision = cont_tp / (cont_tp + cont_fp) if (cont_tp + cont_fp) > 0 else 0
    cont_recall = cont_tp / (cont_tp + cont_fn) if (cont_tp + cont_fn) > 0 else 0
    cont_f1 = 2 * cont_precision * cont_recall / (cont_precision + cont_recall) if (cont_precision + cont_recall) > 0 else 0

    disc_precision = disc_tp / (disc_tp + disc_fp) if (disc_tp + disc_fp) > 0 else 0
    disc_recall = disc_tp / (disc_tp + disc_fn) if (disc_tp + disc_fn) > 0 else 0
    disc_f1 = 2 * disc_precision * disc_recall / (disc_precision + disc_recall) if (disc_precision + disc_recall) > 0 else 0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "type_precision": type_precision, "type_recall": type_recall, "type_f1": type_f1,
        "continuous_precision": cont_precision, "continuous_recall": cont_recall, "continuous_f1": cont_f1,
        "discontinuous_precision": disc_precision, "discontinuous_recall": disc_recall, "discontinuous_f1": disc_f1,
        "pred_total": pred_total, "gold_total": gold_total,
        "continuous_gold": cont_gold, "continuous_pred": cont_pred,
        "discontinuous_gold": disc_gold, "discontinuous_pred": disc_pred
    }


def tune_thresholds(model, dataset: CoAMTokenDataset, grid: List[float] = THRESH_GRID,
                    device: str = DEVICE) -> Dict[str, Any]:
    best = {"f1": -1}
    total = len(grid) ** 3
    progress = tqdm(total=total, desc="Tuning thresholds", ncols=80)

    for a in grid:
        for b in grid:
            for c in grid:
                m = evaluate_exact(model, dataset, (a, b, c), device=device)
                if m["f1"] > best["f1"]:
                    best = {**m, "thresholds": (a, b, c)}
                progress.update(1)

    progress.close()
    return best


def collect_detailed_predictions(model, dataset: CoAMTokenDataset,
                                 thresholds: Tuple[float, float, float],
                                 device: str = DEVICE,
                                 batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
    model.eval()
    ts, te, ti = thresholds

    results = {
        "summary": {
            "total_sentences": 0, "sentences_with_gold_mwes": 0,
            "sentences_with_predicted_mwes": 0, "perfect_predictions": 0,
            "continuous_correct": [], "continuous_false_positives": [],
            "continuous_false_negatives": [], "discontinuous_correct": [],
            "discontinuous_false_positives": [], "discontinuous_false_negatives": []
        }
    }

    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            chunk_feat = batch["chunk_feat"].to(device)
            start_p, end_p, inside_p, _ = model(ids, attn, chunk_feat)

            for bi, sent_idx in enumerate(batch["sent_idx"]):
                s = dataset.sents[sent_idx]
                words = [t.surface for t in s.tokens]
                enc = dataset.tokenizer(
                    words, is_split_into_words=True, truncation=True,
                    max_length=MAX_SEQ_LEN, add_special_tokens=True
                )
                word_ids = enc.word_ids()

                ws, we, wi = [0.0] * len(s.tokens), [0.0] * len(s.tokens), [0.0] * len(s.tokens)
                used = set()

                for ti2, wid in enumerate(word_ids):
                    if wid is None:
                        continue
                    if wid not in used:
                        ws[wid] = float(start_p[bi, ti2])
                        we[wid] = float(end_p[bi, ti2])
                        wi[wid] = float(inside_p[bi, ti2])
                        used.add(wid)

                predicted_spans = reconstruct_candidates(s, ws, we, wi, ts, te, ti)
                gold_spans = [m.tokens for m in s.mwes]
                gold_type_map = {m.tokens: m.type for m in s.mwes}

                pred_set, gold_set = set(predicted_spans), set(gold_spans)

                results["summary"]["total_sentences"] += 1
                if gold_spans:
                    results["summary"]["sentences_with_gold_mwes"] += 1
                if predicted_spans:
                    results["summary"]["sentences_with_predicted_mwes"] += 1
                if pred_set == gold_set:
                    results["summary"]["perfect_predictions"] += 1

                for p in predicted_spans:
                    is_disc = any(b - a != 1 for a, b in zip(p, p[1:]))
                    mwe_text = " ".join([s.tokens[i].surface for i in p])

                    if p in gold_set:
                        info = {"sent_id": s.sent_id, "text": mwe_text, "span": p,
                                "type": gold_type_map.get(p, "UNKNOWN")}
                        key = "discontinuous_correct" if is_disc else "continuous_correct"
                    else:
                        info = {"sent_id": s.sent_id, "text": mwe_text, "span": p}
                        key = "discontinuous_false_positives" if is_disc else "continuous_false_positives"
                    results["summary"][key].append(info)

                for g in gold_spans:
                    if g not in pred_set:
                        is_disc = any(b - a != 1 for a, b in zip(g, g[1:]))
                        mwe_text = " ".join([s.tokens[i].surface for i in g])
                        info = {"sent_id": s.sent_id, "text": mwe_text, "span": g,
                                "type": gold_type_map.get(g, "UNKNOWN")}
                        key = "discontinuous_false_negatives" if is_disc else "continuous_false_negatives"
                        results["summary"][key].append(info)

    summary = results["summary"]
    summary["continuous_correct_count"] = len(summary["continuous_correct"])
    summary["continuous_fp_count"] = len(summary["continuous_false_positives"])
    summary["continuous_fn_count"] = len(summary["continuous_false_negatives"])
    summary["discontinuous_correct_count"] = len(summary["discontinuous_correct"])
    summary["discontinuous_fp_count"] = len(summary["discontinuous_false_positives"])
    summary["discontinuous_fn_count"] = len(summary["discontinuous_false_negatives"])

    return results
