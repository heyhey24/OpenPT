import sys
sys.path.append('.')

import argparse
import os
import os.path as osp
import traceback
import torch

from dassl.utils import set_random_seed, collect_env_info
from dassl.config import get_cfg_default, clean_cfg
from dassl.engine import build_trainer
from dassl.data import build_data_loader

from utils.logger import setup_logger, print

from utils.ood_utils import calibrate_threshold_from_loader, evaluate_ood_metrics

# register datasets and trainers
import datasets
import trainers


def run_ood_detection(trainer, cfg, args, stage):
    dataset = getattr(getattr(trainer, "dm", None), "dataset", None)
    if dataset is None:
        raise AttributeError("trainer.dm.dataset is not available")

    num_base = len(getattr(dataset, "classnames", []))
    if num_base <= 0:
        raise ValueError("Cannot determine base class count from dataset.classnames")

    
    n_prev_ood = max(0, stage)
    id_max_label = num_base + n_prev_ood

    if not hasattr(trainer, "val_loader") or not hasattr(trainer, "test_loader"):
        raise AttributeError("trainer must have val_loader and test_loader")

    log_path = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = osp.join(args.output_dir, "ood_results.log")


    threshold = calibrate_threshold_from_loader(
        trainer.model, 
        trainer.val_loader, 
        args,
        percentile=float(args.ood_percentile)
    )
    

    threshold_msg = f"[calibrate_threshold_from_loader]: threshold={threshold:.4f}"
    print(threshold_msg)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(threshold_msg + "\n")

    # Prefer a dedicated OOD test split if the dataset provides one; otherwise
    # fall back to the trainer's default test_loader.
    ood_test = getattr(dataset, "ood_test", None)

    if ood_test is not None and len(ood_test) > 0:
        base_ds = getattr(trainer.test_loader, "dataset", None)
        base_tfm = getattr(base_ds, "transform", None)

        test_loader_for_ood = build_data_loader(
            cfg,
            sampler_type="SequentialSampler",
            data_source=ood_test,
            batch_size=getattr(cfg.TEST, "BATCH_SIZE", 64),
            tfm=base_tfm,          
            is_train=False,
        )
    else:
        test_loader_for_ood = trainer.test_loader

    def _iter_filtered(loader, is_id: bool):
        for batch in loader:
            if isinstance(batch, dict) and "img" in batch and "label" in batch:
                img = batch["img"]
                label = batch["label"]
                if not isinstance(img, torch.Tensor) or not isinstance(label, torch.Tensor):
                    raise ValueError("img/label must be torch.Tensor")
                mask = label < id_max_label if is_id else label >= id_max_label
                if mask.any():
                    new_batch = dict(batch)
                    new_batch["img"] = img[mask]
                    new_batch["label"] = label[mask]
                    yield new_batch
            else:
                raise ValueError("test loader batch must be a dict with 'img' and 'label'")

    id_loader = _iter_filtered(test_loader_for_ood, is_id=True)
    ood_loader = _iter_filtered(test_loader_for_ood, is_id=False)
    # ood_loader = trainer.train_loader_x
    
    auroc, aupr, fpr95 = evaluate_ood_metrics(
        trainer.model, id_loader, ood_loader, args, threshold=threshold, log_file=log_path
    )

    msg = (
        f"[OOD Detection] AUROC={auroc:.6f}, AUPR={aupr:.6f}, FPR95={fpr95:.6f} "
        f"(score={args.score}, T={float(args.ood_temperature):.4f}, alpha={float(args.ood_alpha):.4f}, pct={float(args.ood_percentile):.2f}), "
        f"stage={getattr(cfg.DATASET, 'CURRENT_STAGE', 0) + 1}"
    )
    print(msg)

    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    # optim settings, new layers' lr will be setted to 0.0035 * 6.5
    cfg.OPTIM.LR_EXP = 2.0
    cfg.OPTIM.STAGED_LR = True
    cfg.OPTIM.NEW_LAYERS = ['prompt_learner.clsname_seq']
    # cfg.OPTIM.NEW_LAYERS = ['prompt_learner', 'film']
    # cfg.OPTIM.NEW_LAYERS = ['linear_probe', 'film']
    cfg.OPTIM.LR = 0.007   #0.0035
    cfg.OPTIM.BASE_LR_MULT = 1.0
    cfg.OPTIM.MAX_EPOCH = 10
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.NUM_OOD_SAMPLES = 5  
    cfg.DATASET.CURRENT_STAGE = 0
    cfg.DATASET.TOTAL_NUM_CLASSES = 1
    cfg.TRAIN.CHECKPOINT_FREQ = -1

    # modules which need to update
    cfg.TRAINER.NAMES_TO_UPDATE = ['prompt_learner.clsname_seq', 'prompt_learner.ctx', 'film']
    # cfg.TRAINER.NAMES_TO_UPDATE = ['prompt_learner', 'film']
    # linear classifier settings
    cfg.TRAINER.LINEAR_PROBE = CN()
    cfg.TRAINER.LINEAR_PROBE.TYPE = 'linear'
    cfg.TRAINER.LINEAR_PROBE.WEIGHT = 0.7
    cfg.TRAINER.LINEAR_PROBE.TEST_TIME_FUSION = True

    cfg.TRAINER.ETFhead = CN()
    cfg.TRAINER.ETFhead.TYPE = 'etf'
    cfg.TRAINER.ETFhead.WEIGHT = 0.7
    cfg.TRAINER.ETFhead.TEST_TIME_FUSION = True
    # cwT module settings
    cfg.TRAINER.FILM = CN()
    cfg.TRAINER.FILM.LINEAR_PROBE = True
    cfg.TRAINER.FILM.ETF = True

    # CoOp settings
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.COOP.ALPHA = 1.0 # for KgCoOp but NOT USE
    cfg.TRAINER.COOP.W = 2.0 # for KgCoOp

    # CoCoOp settings
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # MaPLe settings
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    
    
    

def setup_cfg(args):
    cfg = get_cfg_default()

    clean_cfg(cfg, 'COOP')
    
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    exception_path = osp.join(args.output_dir, 'exceptions.txt')
    if osp.exists(exception_path):
        os.remove(exception_path)
    
    try:
        cfg = setup_cfg(args)
        from yacs.config import CfgNode as CN
        cfg.defrost()
        if not hasattr(cfg, "TRAIN"):
            cfg.TRAIN = CN()
        cfg.TRAIN.EVAL_ONLY = args.eval_only
        cfg.freeze()
        # setup_logger(cfg.OUTPUT_DIR)
        
        if cfg.SEED >= 0:
            print("Setting fixed seed: {}".format(cfg.SEED))
            set_random_seed(cfg.SEED)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True

        if not args.no_train and args.model_dir:
            ood_trainer = build_trainer(cfg)
            ood_trainer.load_model(args.model_dir, epoch=args.load_epoch)
            run_ood_detection(ood_trainer, cfg, args, cfg.DATASET.CURRENT_STAGE)
            del ood_trainer
            if torch.cuda.is_available() and cfg.USE_CUDA:
                torch.cuda.empty_cache()

    except:
        # handle exception, contents of exception will be saved to exception_path
        e = traceback.format_exc()
        with open(exception_path, 'w') as f:
            f.write(e)
        raise Exception('Training task does not run successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument("--ood-temperature", type=float, default=1.0, help="temperature for energy/MCM score")
    parser.add_argument("--ood-alpha", type=float, default=0.7, help="energy alpha for dual-head models")
    parser.add_argument("--ood-percentile", type=float, default=95.0, help="percentile for threshold calibration")
    parser.add_argument("--score", type=str, default="energy", choices=["energy", "MCM"], help="OOD score type: energy or MCM")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
