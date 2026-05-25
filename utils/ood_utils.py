# 能量检测模块 (Energy-based OOD Detection)
# 基于 NeurIPS 2020 论文《Energy-based Out-of-distribution Detection》
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import sys
import os
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


recall_level_default = 0.95


def _stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError(
            "cumsum was found to be unstable: its last element does not correspond to sum"
        )
    return out


def _fpr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    # Compute FPR at a given recall rate, consistent with the logic in display_results.fpr_and_fdr_at_recall.
    classes = np.unique(y_true)
    if (
        pos_label is None
        and not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        )
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values (descending)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Find positions where scores change, and add the last position
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate TP / FP as threshold decreases
    tps = _stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )

    cutoff = np.argmin(np.abs(recall - recall_level))

    # FPR = FP / N_neg，where N_neg = total negative samples = sum(~y_true)
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def _extract_img_from_batch(batch):
    if isinstance(batch, dict):
        if "img" in batch:
            return batch["img"]
        if "image" in batch:
            return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def get_energy_score(model, img, labels=None, temperature: float = 1.0, energy_alpha: float = 0.7):
    """Extract energy scores from the model for an image.
    
    Parameters:
        model: Model instance (can be DataParallel wrapped)
        img: Input image tensor
        labels: Labels (optional)
        temperature: Temperature coefficient
        energy_alpha: Energy combination weight for dual-head models
    
    Returns:
        energy: Energy score
    """
    # Handle DataParallel wrapped models
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Ensure model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # For models with explicit second head (ELP linear probe / ETFhead),
        # use explicit feature and head interfaces.
        if (
            hasattr(model_module, "_forward_feats")
            and hasattr(model_module, "_forward_logits_similarity")
            and hasattr(model_module, "_forward_logits_linear_probe")
        ):
            # ELP series: linear probe head
            text_feats, img_feats = model_module._forward_feats(img)

            # Compute similarity logits
            logits = model_module._forward_logits_similarity(text_feats, img_feats)

            # Compute linear probe logits (compatible with ETFLinear and other head implementations)
            # _forward_logits_linear_probe returns (logits_lp, labels_lp, features_lp)
            logits_lp, _, _ = model_module._forward_logits_linear_probe(
                text_feats, img_feats, labels
            )

            # Compute dual-head energy
            energy_main = -temperature * torch.logsumexp(logits / temperature, dim=1)
            energy_lp = -temperature * torch.logsumexp(logits_lp / temperature, dim=1)
            energy = energy_alpha * energy_lp + (1 - energy_alpha) * energy_main

        elif (
            hasattr(model_module, "_forward_feats")
            and hasattr(model_module, "_forward_logits_similarity")
            and hasattr(model_module, "_forward_logits_etf")
        ):
            # ETF series: ETF head (e.g., ETFCoOp / OpenSetETFCoOp)
            text_feats, img_feats = model_module._forward_feats(img)

            # Compute similarity logits
            logits = model_module._forward_logits_similarity(text_feats, img_feats)

            # Compute ETF head logits
            # _forward_logits_etf returns (logits_etf, labels_etf, features_etf)
            logits_lp, _, _ = model_module._forward_logits_etf(
                text_feats, img_feats, labels
            )

            # Compute dual-head energy
            energy_main = -temperature * torch.logsumexp(logits / temperature, dim=1)
            energy_lp = -temperature * torch.logsumexp(logits_lp / temperature, dim=1)
            energy = energy_alpha * energy_lp + (1 - energy_alpha) * energy_main

        else:
            # For CoOp / MaPLe and other single-head models, forward directly returns final logits
            out = model_module(img)
            # Compatible with forward returning (loss, logits) or similar structures
            if isinstance(out, (tuple, list)) and len(out) > 0:
                logits = out[0] if out[0].ndim >= 2 else out[-1]
            else:
                logits = out
            
            # Single-head model: only compute energy for main logits
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)

    return energy


def get_MCM_score(model, img, labels=None, temperature: float = 1.0):
    """Get MCM (Maximum Concept Matching) score from the model.
    
    MCM score calculation process:
    1. Get image features and text features
    2. Normalize features
    3. Calculate similarity (image_features @ text_features.T)
    4. Apply softmax scaling
    5. Take the negative of the maximum value as the score
    
    Args:
        model: Model instance (can be DataParallel wrapped)
        img: Input image tensor
        labels: Labels (optional)
        temperature: Softmax temperature coefficient
    
    return:
        mcm_score: MCM score (smaller is more likely to be ID)
    """
    # Handle DataParallel wrapped models
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    model.eval()

    with torch.no_grad():
        # For models with _forward_feats (ELP / ETF series), directly get features
        if hasattr(model_module, "_forward_feats"):
            text_feats, img_feats = model_module._forward_feats(img)

            # Normalize image features and text features
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            # Calculate similarity
            output = img_feats @ text_feats.T

        else:
            # For CoOp / MaPLe etc. single-head models, need to extract features separately
            # Extract image features
            img_feats = model_module.image_encoder(img.type(model_module.dtype))
            
            # Extract text features
            prompts = model_module.prompt_learner()
            tokenized_prompts = model_module.tokenized_prompts
            text_feats = model_module.text_encoder(prompts, tokenized_prompts)
            
            # Normalize image features and text features
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            
            # Calculate similarity (without logit_scale, keep original similarity)
            output = img_feats @ text_feats.T
        
        # Apply softmax scaling
        smax = torch.softmax(output / temperature, dim=1)
        
        # MCM score: -max(softmax_scores, dim=1)
        mcm_score = -torch.max(smax, dim=1)[0]

    return mcm_score


def calibrate_threshold_from_loader(model, id_val_dataloader, args, percentile: float = 95):
    """Based on the ID (base class) validation set to calibrate the OOD score threshold.
    
    Args:
        model: Model instance
        id_val_dataloader: ID validation data DataLoader
        args: Parameter object, must contain score field ('energy' or 'MCM')
        percentile: Percentile, default 95
    
    returns:
        threshold: Calculated threshold
    """
    # Get device
    if isinstance(model, torch.nn.DataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    print(f"[calibrate_threshold_from_loader] Start calibrating threshold (score={args.score}, percentile={percentile})...")

    id_scores = []
    model.eval()
    with torch.no_grad():
        for batch in id_val_dataloader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(device)

            # Choose calculation method based on args.score
            if args.score == 'energy':
                score = get_energy_score(model, img)
            elif args.score == 'MCM':
                score = get_MCM_score(model, img)
            else:
                raise ValueError(f"Unsupported score type: {args.score}, only support 'energy' or 'MCM'")

            score_np = score.cpu().numpy()
            if score_np.ndim == 0:
                score_np = np.array([score_np])
            elif score_np.ndim > 1:
                score_np = score_np.flatten()
            id_scores.append(score_np)

    # Check if there is data
    if len(id_scores) == 0:
        raise ValueError("ID data loader is empty, cannot calibrate threshold. Please check if the validation set has data.")

    # Concatenate all scores
    id_scores = np.concatenate(id_scores)

    # Calculate threshold (using percentile)
    threshold = float(np.percentile(id_scores, percentile))

    return threshold


def evaluate_ood_metrics(model, id_data_loader, ood_data_loader, args, threshold=None, log_file: Optional[str] = None):
    """Evaluate OOD detection metrics.
    
    parameters:
        model: Model instance
        id_data_loader: ID data DataLoader
        ood_data_loader: OOD data DataLoader
        args: Parameter object, must contain score field ('energy' or 'MCM')
        threshold: OOD score threshold (optional, used to calculate OOD detection ratio)
        log_file: Log file path (optional)
    
    returns:
        auroc, aupr, fpr95
    """
    if isinstance(model, torch.nn.DataParallel):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    id_scores = []
    ood_scores = []

    model.eval()

    # Collect ID scores
    with torch.no_grad():
        for batch in id_data_loader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(device)

            # Choose calculation method based on args.score
            if args.score == 'energy':
                score = get_energy_score(model, img)
            elif args.score == 'MCM':
                score = get_MCM_score(model, img)
            else:
                raise ValueError(f"Unsupported score type: {args.score}, only support 'energy' or 'MCM'")

            score_np = score.cpu().numpy()
            if score_np.ndim == 0:
                score_np = np.array([score_np])
            elif score_np.ndim > 1:
                score_np = score_np.flatten()
            id_scores.append(score_np)

    # Collect OOD scores
    with torch.no_grad():
        for batch in ood_data_loader:
            img = _extract_img_from_batch(batch)

            if isinstance(img, torch.Tensor):
                img = img.to(device)

            if args.score == 'energy':
                score = get_energy_score(model, img)
            elif args.score == 'MCM':
                score = get_MCM_score(model, img)
            else:
                raise ValueError(f"Unsupported score type: {args.score}, only support 'energy' or 'MCM'")

            score_np = score.cpu().numpy()
            if score_np.ndim == 0:
                score_np = np.array([score_np])
            elif score_np.ndim > 1:
                score_np = score_np.flatten()
            ood_scores.append(score_np)

    if len(id_scores) == 0 or len(ood_scores) == 0:
        raise ValueError("ID or OOD data is empty, cannot calculate OOD metrics.")

    id_scores = np.concatenate(id_scores)
    ood_scores = np.concatenate(ood_scores)

    # Use ID as positive class:
    #  - Positive samples (pos) = ID samples
    #  - Negative samples (neg) = OOD samples
    # Since higher scores indicate OOD likelihood, we use negative scores to make higher scores indicate ID likelihood
    pos = (-id_scores).reshape(-1, 1)
    neg = (-ood_scores).reshape(-1, 1)
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1  # First half is ID (positive=1), second half is OOD (negative=0)

    # AUROC / AUPR: ID as positive class, higher score indicates more ID-like
    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)

    # FPR@95%TPR: consistent with _fpr_at_recall logic, TPR/Recall is for ID
    fpr95 = float(_fpr_at_recall(labels, examples, recall_level=recall_level_default))

    # Additional print based on current threshold (consistent with original logic)
    if threshold is not None:
        threshold_val = float(threshold)
        flagged = float((ood_scores > threshold_val).sum())
        total_ood = float(len(ood_scores))
        ratio_msg = f"[OOD] ood_train Total={int(total_ood)}, Flagged={int(flagged)}, Ratio={flagged/total_ood:.3f}"
        print(ratio_msg)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(ratio_msg + "\n")
            except Exception:
                pass

    summary_msg = f"[OOD] AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR95={fpr95*100:.2f}%"
    print(summary_msg)
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(summary_msg + "\n")
        except Exception:
            pass

    return auroc, aupr, fpr95



