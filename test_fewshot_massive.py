#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging

def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'d'}])


def to_cmd(cfg):
    
    command = f'python weaklabeler/fewShot/train.py --experiment_name fewShot_currated --feat_extractor {cfg["model_name"]} ' \
              f'--training_path weaklabeler/Data/covid_currated.csv --model_save_path weaklabeler/models/ --batch_size 16 --epochs 3 --learning_rate 0.01' \
              f'--shot_num {cfg["shot_num"]} --val_step 100 --text_col Text --target_col Class' \
              f'--target_config_path weaklabeler/configs/covid_currated_config.json ' \
    
    return command


def to_logfile(c, path):
    outfile = "{}/temperature_beaker_v2.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space_all = dict(
        model_name=['roberta-base','roberta-large', 'xlm-roberta-base', 'xlm-roberta-large'],
        shot_num=[4,8,16,32,64],
    )

    configurations = list(cartesian_product(hyp_space_all))

    commands = []
    for cfg in configurations:
        commands.append(to_cmd(cfg))
    
    #write commands in a file

    with open("commands_fewshot_currated.sh", "w") as f:
        for command in commands:
            f.write(command + "\n")




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
