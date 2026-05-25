# result parser for base to new and cross dataset tasks, result will be saved as csv
import argparse
import os
import os.path as osp
import pandas as pd
import re


ORDERS_BASE_TO_NEW = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
                        'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101']

ORDERS_CROSS_DATASET = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
                          'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101', 
                          'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r',]


class ResultParser(object):
    def __init__(self, mode, dir_, save_path):
        self.mode = mode
        self.dir_ = dir_
        self.save_path = save_path
    
    def parse_and_save(self):
        if self.mode == 'b2n':
            self.read_accs_base_to_new()
        elif self.mode == 'openset':
            self.read_results_openset_staged()
        elif self.mode == 'baselines':
            self.read_results_baselines()
        elif self.mode == 'xd':
            self.read_accs_cross_dataset()
        
        self.save()

    def load_property(self, dir_):
        """ get property (trainer, datasets, num_shots, cfg, seeds) from directory """
        trainer = [subdir for subdir in os.listdir(dir_) if osp.isdir(osp.join(dir_, subdir))][0]
        
        dir_ = osp.join(dir_, trainer)
        datasets = os.listdir(dir_)

        if self.mode == 'b2n':
            datasets = [dataset for dataset in ORDERS_BASE_TO_NEW if dataset in datasets]
        elif self.mode == 'xd':
            datasets = [dataset for dataset in ORDERS_CROSS_DATASET if dataset in datasets]
        else:
            raise NotImplementedError
        
        dir_ = osp.join(dir_, datasets[0])
        num_shots = int(os.listdir(dir_)[0][5:])
        
        dir_ = osp.join(dir_, f'shots{num_shots}')
        cfg = os.listdir(dir_)[0]
        
        dir_ = osp.join(dir_, cfg)
        seeds = list(sorted([int(name[4:]) for name in os.listdir(dir_)]))
        
        self.prop = dict(
            trainer=trainer,
            datasets=datasets,
            num_shots=num_shots,
            cfg=cfg,
            seeds=seeds)

    def _discover_openset_layout(self, dir_):
        """ discover (trainer, datasets, num_shots, cfg, stages, seeds) from openset layout """

        dir_ = osp.join(dir_, 'openset_train')
        trainer = [subdir for subdir in os.listdir(dir_) if osp.isdir(osp.join(dir_, subdir))][0]

        dir_ = osp.join(dir_, trainer)
        datasets = sorted([dataset for dataset in os.listdir(dir_)
                           if osp.isdir(osp.join(dir_, dataset))])

        dir_ = osp.join(dir_, datasets[0])
        shots_dir = [name for name in os.listdir(dir_)
                     if name.startswith('shots') and osp.isdir(osp.join(dir_, name))][0]
        num_shots = int(shots_dir[5:])

        dir_ = osp.join(dir_, shots_dir)
        cfg = [name for name in os.listdir(dir_)
               if osp.isdir(osp.join(dir_, name))][0]

        dir_ = osp.join(dir_, cfg)
        stage_dirs = [name for name in os.listdir(dir_)
                      if name.startswith('stage') and osp.isdir(osp.join(dir_, name))]
        stages = sorted([int(name[5:]) for name in stage_dirs])

        dir_ = osp.join(dir_, stage_dirs[0])
        seed_dirs = [name for name in os.listdir(dir_)
                     if name.startswith('seed') and osp.isdir(osp.join(dir_, name))]
        seeds = sorted([int(name[4:]) for name in seed_dirs])

        return trainer, datasets, num_shots, cfg, stages, seeds

    # ---------------------------------------------------------------------
    # Common helpers
    # ---------------------------------------------------------------------

    def _parse_ood_results_file(self, ood_path):
        """Parse (auroc, fpr95) from ood_results.log for each stage.

        Returns: {stage: [(auroc, fpr95), ...]} with list order matching log order.
        """
        if not osp.isfile(ood_path):
            return {}

        with open(ood_path, encoding='utf-8') as f:
            content = f.read()

        pattern = re.compile(
            r"\[OOD Detection\]\s+AUROC=(?P<auroc>\d+\.\d+)\s*,\s*[^,]*,\s*FPR95=(?P<fpr95>\d+\.\d+)"  # metrics
            r".*?stage=(?P<stage>\d+)"  # stage
        )

        results = {}
        for m in pattern.finditer(content):
            s = int(m.group("stage"))
            auroc = float(m.group("auroc"))
            fpr95 = float(m.group("fpr95"))
            results.setdefault(s, []).append((auroc, fpr95))
        return results

    @staticmethod
    def _mean_or_nan(values):
        vals = [v for v in values if isinstance(v, float) and not pd.isna(v)]
        if len(vals) == 0:
            return float('nan')
        return round(sum(vals) / len(vals), 4)

    def _discover_baselines_layout(self, dir_):
        """Discover (trainer, datasets, num_shots, cfg, stages, seeds) from baselines layout.

        Directory structure reference (with outputs/baselines as self.dir_):

        - Baseline ACC results:
            {root}/acc/{trainer}/{dataset}/shots{shots}/{cfg}/stage{S}/seed{seed}/log.txt

        - OOD detection results (aggregated across all stages):
            {root}/ood/{trainer}/{dataset}/shots{shots}/{cfg}/ood_results.log
        """

        dir_acc = osp.join(dir_, 'acc')

        trainer = [subdir for subdir in os.listdir(dir_acc)
                   if osp.isdir(osp.join(dir_acc, subdir))][0]

        dir_trainer = osp.join(dir_acc, trainer)

        # dataset layer (e.g., openset_oxford_pets)
        datasets = sorted([d for d in os.listdir(dir_trainer)
                           if osp.isdir(osp.join(dir_trainer, d))])

        # Infer shots/cfg/stage/seed structure from the first dataset
        dir_dataset0 = osp.join(dir_trainer, datasets[0])
        shots_dir = [name for name in os.listdir(dir_dataset0)
                     if name.startswith('shots') and osp.isdir(osp.join(dir_dataset0, name))][0]
        num_shots = int(shots_dir[5:])

        dir_shots = osp.join(dir_dataset0, shots_dir)
        cfg = [name for name in os.listdir(dir_shots)
               if osp.isdir(osp.join(dir_shots, name))][0]

        dir_cfg = osp.join(dir_shots, cfg)
        stage_dirs = [name for name in os.listdir(dir_cfg)
                      if name.startswith('stage') and osp.isdir(osp.join(dir_cfg, name))]
        stages = sorted([int(name[5:]) for name in stage_dirs])

        dir_stage0 = osp.join(dir_cfg, stage_dirs[0])
        seed_dirs = [name for name in os.listdir(dir_stage0)
                     if name.startswith('seed') and osp.isdir(osp.join(dir_stage0, name))]
        seeds = sorted([int(name[4:]) for name in seed_dirs])

        return trainer, datasets, num_shots, cfg, stages, seeds

    def read_accs_base_to_new(self):
        dir_ = self.dir_

        base_dir = osp.join(dir_, 'train_base')
        new_dir = osp.join(dir_, 'test_new')
        
        self.load_property(base_dir)
        prop = self.prop

        trainer = prop['trainer']
        datasets = prop['datasets']
        num_shots = prop['num_shots']
        cfg = prop['cfg']
        seeds = prop['seeds']

        headers = ['dataset', 
                   'base_acc_seed1', 'new_acc_seed1', 'H_seed1', 
                   'base_acc_seed2', 'new_acc_seed2', 'H_seed2',
                   'base_acc_seed3', 'new_acc_seed3', 'H_seed3']
        rows = []
        
        for dataset in datasets:
            row = [dataset]
            
            for seed in seeds:
                base_path = osp.join(base_dir, trainer, dataset, f'shots{num_shots}', cfg, f'seed{seed}', 'log.txt')
                new_path = osp.join(new_dir, trainer, dataset, f'shots{num_shots}', cfg, f'seed{seed}', 'log.txt')

                base_acc = self._read_acc(base_path)
                new_acc = self._read_acc(new_path)
                H = 2 / (1 / base_acc + 1 / new_acc)
                
                row += [base_acc, new_acc, H]
            
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=headers)
        df['base_acc'] = (df['base_acc_seed1'] + df['base_acc_seed2'] + df['base_acc_seed3']) / 3
        df['new_acc'] = (df['new_acc_seed1'] + df['new_acc_seed2'] + df['new_acc_seed3']) / 3

        df.loc[len(df.index)] = ['average'] + df.drop(columns=['dataset']).mean().tolist()
        df['H'] = 2 / (1 / df['base_acc'] + 1 / df['new_acc'])
        
        self.df = df

    def read_results_openset_staged(self):
        """Read openset staged training results and corresponding OOD / zeroshot results.

        Directory structure reference:

        - Training accuracy (per stage, per seed):
            {root}/openset_train/{trainer}/{dataset}/shots{shots}/{cfg}/stage{S}/seed{seed}/log.txt

        - OOD detection results (per stage, possibly aggregated across multiple runs):
            {root}/ood/{trainer}/{dataset}/shots{shots}/{cfg}/ood_results.log
          Each line in the file follows the format:
            [OOD Detection] AUROC=..., AUPR=..., FPR95=... (..), stage=S

        - Zeroshot results (per stage, per seed):
            {root}/zsclip/{dataset}/stage{S}/log.txt

        """

        dir_ = self.dir_

        (trainer, datasets, num_shots, cfg, stages, seeds) = self._discover_openset_layout(dir_)

        headers = ['dataset', 'stage',
           'acc_seed1', 'acc_seed2', 'acc_seed3', 'acc', 
           'ood_auroc_seed1', 'ood_fpr95_seed1',
           'ood_auroc_seed2', 'ood_fpr95_seed2', 
           'ood_auroc_seed3', 'ood_fpr95_seed3',
           'ood_auroc', 'ood_fpr95', 
           'zs_acc']

        all_dfs = []  # To save DataFrames for all datasets
        
        for dataset in datasets:
            rows = []  # Reset rows list for each dataset
            # OOD results file path (aggregated across all stages)
            ood_results_path = osp.join(
                dir_, 'ood', dataset, f'shots{num_shots}', 'ood_results.log',
            )
            ood_stage_results = self._parse_ood_results_file(ood_results_path)

            for stage in stages:
                row = [dataset, stage]

                # 1) openset_train per seed ACC
                acc_list = []
                for seed in seeds:
                    log_path = osp.join(dir_, 'openset_train', trainer, dataset, f'shots{num_shots}',
                        cfg, f'stage{stage}', f'seed{seed}', 'log.txt',)

                    acc = self._read_acc(log_path)
                    acc_list.append(acc)
                    row.append(acc)

                # multi-seed mean accuracy (openset_train)
                row.append(sum(acc_list) / len(acc_list))

                # 2) OOD metrics: per-seed + per-stage mean (if a stage has no results, fill with NaN)
                stage_ood_list = ood_stage_results.get(stage, [])

                # per-seed columns
                per_seed_aurocs, per_seed_fpr95s = [], []  # Removed per_seed_auprs
                for i, seed in enumerate(seeds):
                    if i < len(stage_ood_list):
                        auroc_i, fpr95_i = stage_ood_list[i]  # Removed aupr_i
                    else:
                        auroc_i = fpr95_i = float('nan')  # Removed aupr_i
                    per_seed_aurocs.append(auroc_i)
                    per_seed_fpr95s.append(fpr95_i)  # Removed per_seed_auprs.append(aupr_i)

                    row.extend([auroc_i, fpr95_i])  # Removed aupr_i from extend

                # per-stage mean over available seed results
                row.append(self._mean_or_nan(per_seed_aurocs))
                row.append(self._mean_or_nan(per_seed_fpr95s))

                # 3) ZeroshotCLIP per seed ACC
                zs_acc_list = []
                zs_log_path = osp.join(
                    dir_, 'zsclip', dataset, f'stage{stage}', 'log.txt', )

                if osp.isfile(zs_log_path):
                    zs_acc = self._read_acc(zs_log_path)
                else:
                    zs_acc = float('nan')
                zs_acc_list.append(zs_acc)
                row.append(zs_acc)

                rows.append(row)

            # Create DataFrame for current dataset
            if rows:  # Ensure there is data
                df = pd.DataFrame(rows, columns=headers)

                # Append overall average row for current dataset at the end
                avg_row = ['average', '-']
                # openset_train acc per seed + mean
                for seed in seeds:
                    avg_row.append(df[f'acc_seed{seed}'].mean())
                avg_row.append(df['acc'].mean())

                # OOD per-seed metrics mean
                for seed in seeds:
                    avg_row.append(df[f'ood_auroc_seed{seed}'].mean())
                    avg_row.append(df[f'ood_fpr95_seed{seed}'].mean())

                # OOD metrics overall mean (keep 4 decimal places)
                avg_row.append(round(df['ood_auroc'].mean(), 4))
                avg_row.append(round(df['ood_fpr95'].mean(), 4))

                # ZeroshotCLIP acc per seed + mean
                if 'zs_acc' in df.columns:
                    avg_row.append(df['zs_acc'].mean())
                else:
                    avg_row.append(float('nan'))

                df.loc[len(df.index)] = avg_row
                all_dfs.append(df)  # Add current dataset's DataFrame to the list

        # Merge all datasets' DataFrames
        if all_dfs:
            self.df = pd.concat(all_dfs, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=headers)


    def read_results_baselines(self):
        """Read baselines (base ACC + OOD) results.

        Directory structure reference (with outputs/baselines as self.dir_):

        - Baseline ACC results（per stage, per seed）：
            {root}/acc/{trainer}/{dataset}/shots{shots}/{cfg}/stage{S}/seed{seed}/log.txt

        - OOD detection results（per stage，aggregated to a single file）：
            {root}/ood/{trainer}/{dataset}/shots{shots}/{cfg}/ood_results.log
          Each line in the file follows the format:
            [OOD Detection] AUROC=..., FPR95=..., (...), stage=S
        """

        dir_ = self.dir_

        (trainer, datasets, num_shots, cfg, stages, seeds) = self._discover_baselines_layout(dir_)

        headers = [
            'dataset', 'stage',
            'acc_seed1', 'acc_seed2', 'acc_seed3', 'acc',
            'ood_auroc_seed1', 'ood_fpr95_seed1',
            'ood_auroc_seed2', 'ood_fpr95_seed2',
            'ood_auroc_seed3', 'ood_fpr95_seed3',
            'ood_auroc', 'ood_fpr95',
        ]

        all_dfs = []

        for dataset in datasets:
            rows = []

            # OOD results file (all stages aggregated in one file for this dataset)
            ood_results_path = osp.join(
                dir_, 'ood', dataset, f'shots{num_shots}', 'ood_results.log',
            )
            ood_stage_results = self._parse_ood_results_file(ood_results_path)

            for stage in stages:
                row = [dataset, stage]

                # 1) baseline ACC per seed
                acc_list = []
                for seed in seeds:
                    log_path = osp.join(
                        dir_, 'acc', trainer, dataset, f'shots{num_shots}', cfg,
                        f'stage{stage}', f'seed{seed}', 'log.txt',
                    )

                    acc = self._read_acc(log_path)
                    acc_list.append(acc)
                    row.append(acc)

                # multi-seed mean accuracy
                row.append(sum(acc_list) / len(acc_list))

                # 2) OOD metrics: per-seed + per-stage mean
                stage_ood_list = ood_stage_results.get(stage, [])

                per_seed_aurocs, per_seed_fpr95s = [], []
                for i, seed in enumerate(seeds):
                    if i < len(stage_ood_list):
                        auroc_i, fpr95_i = stage_ood_list[i]
                    else:
                        auroc_i = fpr95_i = float('nan')
                    per_seed_aurocs.append(auroc_i)
                    per_seed_fpr95s.append(fpr95_i)

                    row.extend([auroc_i, fpr95_i])

                row.append(self._mean_or_nan(per_seed_aurocs))
                row.append(self._mean_or_nan(per_seed_fpr95s))

                rows.append(row)

            if rows:
                df = pd.DataFrame(rows, columns=headers)

                # Append overall average row for current dataset at the end
                avg_row = ['average', '-']

                for seed in seeds:
                    avg_row.append(df[f'acc_seed{seed}'].mean())
                avg_row.append(df['acc'].mean())

                for seed in seeds:
                    avg_row.append(df[f'ood_auroc_seed{seed}'].mean())
                    avg_row.append(df[f'ood_fpr95_seed{seed}'].mean())

                avg_row.append(round(df['ood_auroc'].mean(), 4))
                avg_row.append(round(df['ood_fpr95'].mean(), 4))

                df.loc[len(df.index)] = avg_row
                all_dfs.append(df)

        if all_dfs:
            self.df = pd.concat(all_dfs, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=headers)


    def read_accs_cross_dataset(self):
        dir_ = self.dir_

        self.load_property(dir_)
        prop = self.prop

        trainer = prop['trainer']
        datasets = prop['datasets']
        num_shots = prop['num_shots']
        cfg = prop['cfg']
        seeds = prop['seeds']

        headers = ['dataset', 'acc_seed1', 'acc_seed2', 'acc_seed3']
        rows = []
        
        datasets = [dataset for dataset in ORDERS_CROSS_DATASET if dataset in datasets]
        for dataset in datasets:
            row = [dataset]
            
            for seed in seeds:
                path = osp.join(dir_, trainer, dataset, f'shots{num_shots}', cfg, f'seed{seed}', 'log.txt')
                acc = self._read_acc(path)
                row.append(acc)
            
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=headers)
        df['acc'] = (df['acc_seed1'] + df['acc_seed2'] + df['acc_seed3']) / 3

        dg_datasets = [dataset for dataset in datasets
                      if 'imagenet' in dataset and dataset != 'imagenet']
        xd_datasets = [dataset for dataset in datasets
                      if dataset not in dg_datasets and dataset != 'imagenet']
        
        dg_df = df.loc[df['dataset'].isin(dg_datasets)].copy().reset_index(drop=True)
        xd_df = df.loc[df['dataset'].isin(xd_datasets)].copy().reset_index(drop=True)
        img_net_df = df.loc[df['dataset'] == 'imagenet'].copy().reset_index(drop=True)

        dg_df.loc[len(dg_df.index)] = ['average_dg'] + dg_df.drop(columns=['dataset']).mean().tolist()
        xd_df.loc[len(xd_df.index)] = ['average_xd'] + xd_df.drop(columns=['dataset']).mean().tolist()

        df = pd.concat([img_net_df, xd_df, dg_df]).reset_index(drop=True)
        
        self.df = df
        
    def save(self):
        save_path = self.save_path
        save_dir = osp.join(*save_path.replace('\\', '/').split('/')[:-1])

        os.makedirs(save_dir, exist_ok=True)
        self.df.round(3).to_csv(save_path, index=None)
        # self.df.to_csv(save_path, index=None)
    def _read_acc(self, path):
        with open(path, encoding='utf-8') as f:
            content = ''.join(f.readlines())
        try:
            acc = float(re.findall(r'accuracy\: (\d+\.\d*)\%', content)[-1])
            return acc
        except BaseException as e:
            print(f'Key word "accuracy" not found in file {path}!')
            raise e
    

def main(args):
    parser = ResultParser(args.mode, args.dir, args.save_path)
    parser.parse_and_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='mode, b2n or xd or openset')
    parser.add_argument('--dir', type=str, help='directory which need to stats')
    parser.add_argument('--save-path', type=str, help='directory to save statistics')
    args = parser.parse_args()
    main(args)
