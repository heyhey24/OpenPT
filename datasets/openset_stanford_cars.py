import os
import pickle
from scipy.io import loadmat

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .openset_oxford_pets import OpenSetOxfordPets


@DATASET_REGISTRY.register()
class OpenSetStanfordCars(DatasetBase):

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        """Open-set variant of StanfordCars with staged OOD logic."""

        self.cfg = cfg

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OpenSetOxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            train, val = OxfordPets.split_trainval(trainval)
            OpenSetOxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED

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
        current_stage = getattr(cfg.DATASET, "CURRENT_STAGE", 0)
        # index = getattr(cfg.DATASET, "INDEX", 0)
        self.ood_class_file = os.path.join(
            self.dataset_dir,
            "openset_ood_class.json",
        )

        if subsample == "base":
            train, val, test = OpenSetOxfordPets.subsample_classes(
                self, train, val, test, subsample=subsample, stage=current_stage
            )
            super().__init__(train_x=train, val=val, test=test)
        else:
            if current_stage == 0:
                _, _, _, ood_test = OpenSetOxfordPets._build_stage_split(
                    self, train, val, test, stage=current_stage + 1
                )
                self.ood_test = ood_test
                train, val, test = OpenSetOxfordPets.subsample_classes(
                    self, train, val, test, subsample=subsample, stage=current_stage
                )
                super().__init__(train_x=train, val=val, test=test)
            else:
                (train_base,) = OpenSetOxfordPets.subsample_classes(
                    self, train, subsample=subsample, stage= current_stage
                )

                ood_train, val, test, ood_test = OpenSetOxfordPets._build_stage_split(
                    self, train, val, test, stage= current_stage
                )

                self.ood_test = ood_test
                super().__init__(train_x=train_base, val=val, test=test)
                self._train_x = ood_train

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items
