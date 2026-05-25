import argparse
import datetime
import numpy as np
import os
import os.path as osp
import shutil

from utils.gpu_allocater import GPUAllocater
from utils.logger import setup_logger, print
from utils.mail import MailClient
from utils.result_parser import ResultParser

from configs import get_config

from templates import (
    get_command,
    get_ood_command,
    get_zsclip_eval_command,
    get_openset_train_command,
    TEST_BASELINES_ACC_TEMPLATE,
    OOD_CMD_TEMPLATE,
)

class ParallelRunner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.data_cfg = cfg['data']
        self.train_cfg = cfg['train']
        self.grid_search_cfg = cfg['grid_search']
        self.output_cfg = cfg['output']
        self.mail_cfg = cfg['mail']
        
        self.allocater = GPUAllocater(cfg['gpu_ids'])
        self.mail = MailClient(self.mail_cfg)

    def run(self):
        """ main method """
        grid_search_cfg = self.grid_search_cfg
        output_cfg = self.output_cfg
        
        # remove useless directories
        remove_dirs = [output_cfg[name] for name in output_cfg['remove_dirs']]
        
        for dir_ in remove_dirs:
            if osp.exists(dir_):
                print(f'Remove directory >>> {dir_}')
                shutil.rmtree(dir_)
                
            os.makedirs(dir_)

        setup_logger(osp.join(output_cfg['root'], 'log.txt'), write_to_console=True)

        start_time = datetime.datetime.now()

        try:
            # main
            if grid_search_cfg['enable']:
                result_paths = self.run_grid_serach() 
            else:
                result_paths = [self.run_single()]
        except:
            # handle exception, contents of exception will be sent to your email
            end_time = datetime.datetime.now()
            contents = [f'<b>Training tasks FAILED!</b> Time cost: {end_time - start_time}\n\n', 
                        '<b>Exception is following above:</b>\n']

            exception_path = osp.join(output_cfg['root'], 'exceptions.txt')
            with open(exception_path) as f:
                contents += f.readlines()
            
            print('Training tasks FAILED! Mail will be sent >>> {}'.format(self.mail_cfg['to']))
            self.mail.send('Training Tasks FAILED!', contents)
            return

        # after finished, results will be sent to your email
        end_time = datetime.datetime.now()
        contents = [f'<b>Training tasks FINISHED!</b> Time cost: {end_time - start_time}\n\n', 
                    '<b>Results are following above:</b>\n']
        
        for result_path in result_paths:
            contents += [f'\n<b>{result_path}</b>\n']
            with open(result_path) as f:
                contents += f.readlines()

        print('Training tasks FINISHED! Mail will be sent >>> {}'.format(self.mail_cfg['to']))
        self.mail.send('Training Tasks FINISHED!', contents)
            
    def run_grid_serach(self):
        """ run if grid search is enabled """
        output_cfg = self.output_cfg
        root = output_cfg['root']

        # parse gird search params
        dirnames, opts_list = self.get_grid_search_opts()
        
        print('Grid search opts:')
        for opts in opts_list:
            print(opts)
        print()
        
        result_paths = []
        
        for idx, (dirname, opts) in enumerate(zip(dirnames, opts_list)):
            # run single task for each grid search param group
            print(f'[{idx + 1} / {len(dirnames)}] Running task {opts}\n')
            output_cfg['root'] = osp.join(root, dirname)
            
            self.dirname = dirname
            result_paths.append(self.run_single(opts))
        
        return result_paths
            
    def run_single(self, opts=[]):
        cfg = self.cfg
        train_cfg = self.train_cfg
        grid_search_cfg = self.grid_search_cfg
        output_cfg = self.output_cfg
        
        # get command
        if cfg['mode'] == 'b2n':
            commands = self.get_base_to_new_commands(opts)
        elif cfg['mode'] == 'openset':
            commands = self.get_openset_commands(opts)
        elif cfg['mode'] == 'baselines':
            commands = self.get_baselines_commands(opts)
        else:
            commands = self.get_cross_dataset_commands(opts)
        
        # add command
        for command in commands:
            self.allocater.add_command(command)
        
        # run command
        self.allocater.run()
        
        # save result
        if not grid_search_cfg['enable']:
            filename = '{}-{}-{}.csv'.format(cfg['mode'], train_cfg['trainer'], train_cfg['cfg'])
        else:
            filename = '{}-{}-{}-{}.csv'.format(cfg['mode'], train_cfg['trainer'], train_cfg['cfg'], self.dirname)
            
        os.makedirs(output_cfg['result'], exist_ok=True)
        result_path = osp.join(output_cfg['result'], filename)

        print(f'Results will be save >>> {result_path}')
        parser = ResultParser(cfg['mode'], output_cfg['root'], result_path)
        parser.parse_and_save()
        
        return result_path
    
    def get_base_to_new_commands(self, opts=[]):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg

        data_root = data_cfg['root']
        datasets = data_cfg['datasets_base_to_new']
        
        trainer = train_cfg['trainer']
        cfg = train_cfg['cfg']
        seeds = train_cfg['seeds']
        loadep = train_cfg['loadep']
        shots = train_cfg['shots']
        opts += train_cfg['opts']
        
        root = output_cfg['root']

        commands = []

        # training on all datasets
        for dataset in datasets:
            for seed in seeds:
                cmd = get_command(data_root, seed, trainer, dataset, cfg, root, 
                                  shots, dataset, loadep, opts, mode='b2n', train=True)
                commands.append(cmd)
        # testing on all datasets
        for dataset in datasets:
            for seed in seeds:
                cmd = get_command(data_root, seed, trainer, dataset, cfg, root, 
                                  shots, dataset, loadep, opts, mode='b2n', train=False)
                commands.append(cmd)
                
        return commands
    
    def get_baselines_commands(self, opts=[]):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg

        data_root = data_cfg['root']
        datasets = data_cfg['datasets_openset']
        
        trainer = train_cfg['trainer']
        cfg = train_cfg['cfg']
        seeds = train_cfg['seeds']
        loadep = train_cfg['loadep']
        shots = train_cfg['shots']
        stages = train_cfg['stages']
        opts += train_cfg['opts']
        
        root = output_cfg['root']
        base_root = output_cfg['base_root']
        commands = []

        # training on all datasets
        for dataset in datasets:
            for seed in seeds:
                for stage in stages:
                    base_dataset = dataset.replace('openset_', '', 1)
                    # Baseline ACC eval on base model for this stage
                    cmd_base_acc = TEST_BASELINES_ACC_TEMPLATE.format(data_root, seed, trainer, dataset, cfg, root,
                        shots, loadep, stage, base_dataset, base_root,)
                    commands.append(cmd_base_acc)

                    warm_start = (
                        f"--model-dir {base_root}/train_base/{trainer}/{base_dataset}/shots{shots}/{cfg}/seed{seed} \\\n"  # noqa: E501
                        f"--load-epoch {loadep} \\\n"  # noqa: E501
                    )
                    cmd_base_ood = OOD_CMD_TEMPLATE.format(data_root, seed, trainer, dataset, cfg, root,
                        shots, loadep, stage-1, warm_start, )
                    commands.append(cmd_base_ood)
                
        return commands

    def get_openset_commands(self, opts=[]):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg

        data_root = data_cfg['root']
        datasets = data_cfg['datasets_openset']
        trainer = train_cfg['trainer']
        cfg = train_cfg['cfg']
        seeds = train_cfg['seeds']
        loadep = train_cfg['loadep']
        shots = train_cfg['shots']
        stages = train_cfg['stages']

        root = output_cfg['root']
        
        commands = []

        
        use_openset_template = str(trainer).startswith("OpenSet")

        # training and testing on all datasets across stages
        for dataset in datasets:
            for stage in stages:
                for seed in seeds:
                    all_opts = []
                    all_opts += train_cfg['opts']
                    all_opts += opts

                    # 1) OOD detection on current stage (before training)
                    cmd_ood = get_ood_command(data_root, seed, trainer, dataset, cfg, root, shots,
                     loadep, prev_stage=stage-1, opts=all_opts)
                    commands.append(cmd_ood)

                    # 2) Train command for this stage
                
                        # For trainers in the OpenSet* series, use the staged openset training template.
                        # Convention: stages are consecutive integers (e.g., [1, 2, 3,...]), 
                        # and for stage > min_stage, continue training from the previous stage's checkpoint.
                        # min_stage = min(stages) if len(stages) > 0 else stage
                        
                    prev_stage = stage - 1

                    cmd_tr = get_openset_train_command(data_root, seed, trainer, dataset, cfg, root, 
                    shots, stage, prev_stage, loadep, opts=all_opts,)
                    commands.append(cmd_tr)
                    
                cmd_zs = get_zsclip_eval_command(data_root, dataset, cfg, root, stage, opts=all_opts,)
                commands.append(cmd_zs)
                
        return commands
    

    def get_cross_dataset_commands(self, opts):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg

        data_root = data_cfg['root']
        datasets = data_cfg['datasets_cross_dataset']
        
        trainer = train_cfg['trainer']
        cfg = train_cfg['cfg']
        seeds = train_cfg['seeds']
        loadep = train_cfg['loadep']
        shots = train_cfg['shots']
        opts += train_cfg['opts']
        
        root = output_cfg['root']

        commands = []
        
        # training on image
        load_dataset = 'imagenet'
        for seed in seeds:
            cmd = get_command(data_root, seed, trainer, load_dataset, cfg, root,
                              shots, load_dataset, loadep, opts, mode='xd', train=True)
            commands.append(cmd)

        # testing on other datasets
        for dataset in datasets:
            for seed in seeds:
                cmd = get_command(data_root, seed, trainer, dataset, cfg, root,
                                  shots, load_dataset, loadep, opts, mode='xd', train=False)
                commands.append(cmd)
                
        return commands
    
    def get_grid_search_opts(self):
        grid_search_cfg = self.grid_search_cfg
        mode = grid_search_cfg['mode']
        params = grid_search_cfg['params']

        names = [param['name'] for param in params]
        aliases = [param['alias'] for param in params]
        values_list = [param['values'] for param in params]
        
        # grid to sequential
        if mode == 'grid' and len(names) > 1:
            values_list = [list(arr.flatten()) for arr in np.meshgrid(*values_list)]
        
        # build opts
        dirnames, grid_search_opts_list = [], []
        for i in range(len(values_list[0])):
            values = [values[i] for values in values_list]

            dirname, opts = [], []
            for name, alias, value in zip(names, aliases, values):
                dirname.append(f'{alias}{value}')
                opts += [name, value]
            
            dirname = '_'.join(dirname)
            dirnames.append(dirname)
            grid_search_opts_list.append(opts)
            
        return dirnames, grid_search_opts_list


def main(args):
    cfg = get_config(args.cfg)
    runner = ParallelRunner(cfg)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()
    main(args)
