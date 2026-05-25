import os
import pickle
import math
import random
import json
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OpenSetOxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        self.cfg = cfg
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED

        # Construct a file to store the OOD class indices to be selected
        self.ood_class_file = os.path.join(
            self.dataset_dir,
            f"openset_ood_class.json",
        )

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        current_stage = cfg.DATASET.CURRENT_STAGE
        
        if subsample == "base":
           # This branch is used to construct the baseline ACC test set under class incremental setting, consisting of base classes and OOD class samples
            train, val, test = self.subsample_classes(
                train, val, test, subsample=subsample, stage=current_stage
            )

            super().__init__(train_x=train, val=val, test=test)
        else:
            # For the openset setting, if it is stage 0, directly obtain the current stage's train/val/test according to subsample_classes,
            # and initialize with them (no OOD split). Stage 0 mainly constructs the validation and test sets for OOD detection to be used in subsequent evaluation.
            if current_stage == 0:
                _, _, _, ood_test = self._build_stage_split(
                    train, val, test, stage=current_stage + 1
                )
                # Additionally expose the OOD test dataset for use in OOD detection, etc.
                self.ood_test = ood_test
                train, val, test = self.subsample_classes(
                    train, val, test, subsample=subsample, stage=current_stage
                )
                
                super().__init__(train_x=train, val=val, test=test)
            else:
               # Starting from stage 1, use the openset procedure:
                # 1) First extract the base training set from the original train according to the current stage
                (train_base,) = self.subsample_classes(
                    train, subsample=subsample, stage=current_stage
                )

                # 2) Then, according to the stage, select an OOD class from the full class set to construct ood_train / new val and test
                ood_train, val, test, ood_test = self._build_stage_split(
                    train, val, test, stage=current_stage
                )

                self.ood_test = ood_test
                # First initialize with train_base to ensure that classnames only contain base classes (matching the model during training)
                # Ensure that the model trained on the base set can be loaded correctly and the dimensions match
                super().__init__(train_x=train_base, val=val, test=test)
                self._train_x = ood_train   # Use the OOD samples of the current stage for training


    def _build_stage_split(self, train, val, test, stage=0):
        """
            Construct openset train/validation/test splits per stage, and construct ood_test_all on the first call.

            Returns:
                ood_train: OOD training samples of the current stage (relabeled)
                new_val:   Validation set for threshold calibration (ID only, relabeled)
                new_test:  Test set of the current stage (base + historical OOD + current OOD, relabeled)
                ood_test:  Test set for OOD detection
            """

        # Read the number of sampled OOD samples from the configuration
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            num_ood_samples = 16
        else:
            num_ood_samples = int(getattr(cfg.DATASET, "NUM_OOD_SAMPLES", 16))

        # Get all classes in the current data (using the full train)
        labels = sorted({item.label for item in train})
        # Build label -> classname mapping for debugging information
        label_to_classname = {}
        for item in train:
            if item.label not in label_to_classname:
                label_to_classname[item.label] = item.classname
        n = len(labels)
        if n == 0:
            return [], val, test

        # Read the pre-defined OOD class indices from openset_ood_class.json
        # Assume the file content is an integer list, e.g., [10, 3, 25, ...]
        ood_indices = None
        if hasattr(self, "ood_class_file") and os.path.exists(self.ood_class_file):
            try:
                with open(self.ood_class_file, "r") as f:
                    state = json.load(f)
                # Compatible with two formats:
                # 1) Directly a list
                # 2) A dictionary containing "indices" or "ood_indices" keys
                if isinstance(state, list):
                    ood_indices = state
                elif isinstance(state, dict):
                    if "ood_indices" in state:
                        ood_indices = state["ood_indices"]
                    elif "indices" in state:
                        ood_indices = state["indices"]
            except Exception:
                ood_indices = None

        # If the OOD index list is not successfully read, degrade to no OOD case
        if not ood_indices:
            return [], val, test

        # stage starts from 1:
        #  - The stage-th index corresponds to the OOD class for this stage's training
        #  - The first stage indices are used for nonbase_selected (accumulated OOD classes)
        if stage <= 0 or stage > len(ood_indices):
            # When the stage exceeds the predefined range, degrade to no OOD
            return [], val, test

        current_ood_index = ood_indices[stage - 1]
        # The current OOD class's specific label value in the entire label set
        ood_label = current_ood_index

        # Extract training samples of the OOD class from train (up to NUM_OOD_SAMPLES), which will be relabeled later with the same rules as val/test
        ood_candidates = [it for it in train if it.label == ood_label]
        n_ood = min(num_ood_samples, len(ood_candidates))
        ood_raw = ood_candidates[:n_ood]

        # Split base / OOD candidates in half: the first half is base, the second half is OOD candidates
        m = math.ceil(n / 2)
        base_labels = labels[:m]

        # Build new val/test using base classes + accumulated OOD classes, and relabel the labels
        # Base class labels are relabeled to [0..|base|-1]; selected OOD classes are mapped in order as
        # len(base_labels)+0, len(base_labels)+1, ...
        base_relabeler = {y: y_new for y_new, y in enumerate(base_labels)}

        # Accumulated OOD classes: the first stage indices in the file
        # nonbase_selected = []
        # nonbase_selected = ood_indices[:4]
        nonbase_selected = ood_indices[:stage]
        # nonbase_selected.append(ood_label)
        nonbase_relabeler = {
            y: (len(base_labels) + idx) for idx, y in enumerate(nonbase_selected)
        }
        selected_for_eval = set(base_labels) | set(nonbase_selected)

        # Debug information: print current stage OOD class and historical OOD classes
        # Current stage OOD class
        current_classname = label_to_classname.get(ood_label, "<unknown>")
        current_remapped = None
        if ood_label in base_relabeler:
            current_remapped = base_relabeler[ood_label]
        elif ood_label in nonbase_relabeler:
            current_remapped = nonbase_relabeler[ood_label]
        print(f"[OpenSetOxfordPets][stage={stage}] current OOD class: "
              f"label={ood_label}, remapped={current_remapped}, classname={current_classname}")

        # Historical OOD classes (including the current stage)
        history_info = []
        for y in nonbase_selected:
            cls_name = label_to_classname.get(y, "<unknown>")
            if y in base_relabeler:
                y_new = base_relabeler[y]
            elif y in nonbase_relabeler:
                y_new = nonbase_relabeler[y]
            else:
                y_new = None
            history_info.append(
                {
                    "label": y,
                    "remapped": y_new,
                    "classname": cls_name,
                }
            )
        print(f"[OpenSetOxfordPets][stage={stage}] history OOD classes: {history_info}")

        def _remap(items):
            out = []
            for it in items:
                if it.label not in selected_for_eval:
                    continue
                if it.label in base_relabeler:
                    new_label = base_relabeler[it.label]
                elif it.label in nonbase_relabeler:
                    new_label = nonbase_relabeler[it.label]
                else:
                    continue
                out.append(
                    Datum(
                        impath=it.impath,
                        label=new_label,
                        classname=it.classname,
                    )
                )
            return out

        # Relabel ood_train to ensure the same label space as val/test
        ood_train = _remap(ood_raw)
        new_val = _remap(val)
        new_test = _remap(test)

        # Construct an additional cross-stage OOD detection test set ood_test_all:
        # Contains base class test samples and all OOD class test samples for the current stage.
        # Only construct once when first available, to avoid repeated calculation.
        
        if ood_indices and len(ood_indices) >= 1:
            k = min(5, len(ood_indices))
            nonbase_selected = ood_indices[:k]
            nonbase_relabeler = {
                y: (len(base_labels) + idx) for idx, y in enumerate(nonbase_selected)
            }
            selected_for_eval = set(base_labels) | set(nonbase_selected)
            ood_test = _remap(test)

        return ood_train, new_val, new_test, ood_test


    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    def subsample_classes(self, *args, subsample="base", stage=0):
        """Divide classes into groups.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        # base represents baseline, this branch constructs base classes + new OOD classes for testing baseline accuracy
        # openset represents openPT, this branch constructs the data set for openPT
        assert subsample in ["base", "openset"]


        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES! (stage={stage})")

        if subsample == "base":
            selected = labels[:m]

            # Additional: According to the first stage indices in openset_ood_class.json,
            # add the corresponding classes to selected
            if stage > 0 and hasattr(self, "ood_class_file") and os.path.exists(self.ood_class_file):
                try:
                    with open(self.ood_class_file, "r") as f:
                        state = json.load(f)
                    ood_indices = None
                    if isinstance(state, list):
                        ood_indices = state
                    elif isinstance(state, dict):
                        if "ood_indices" in state:
                            ood_indices = state["ood_indices"]
                        elif "indices" in state:
                            ood_indices = state["indices"]
                    if ood_indices:
                        extra = []
                        for idx in ood_indices[:stage]:
                            # Only add labels that exist in the current data and have not been selected
                            if idx in labels and idx not in selected:
                                extra.append(idx)
                        selected = selected + extra
                        print(f"[OpenSetOxfordPets][subsample_classes][stage={stage}] "
                              f"append OOD indices to base selected: {extra}")
                except Exception:
                    # If reading fails, ignore additional OOD classes
                    pass
        else:  # subsample == "openset"
            # Only select the class at index m (the first new class) as the openset class
            selected = labels[:m]

        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output