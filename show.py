#!/usr/bin/env python
# encoding: utf8

import json
import os
import sys


def show_experiment(experiment_name, experiment_param, path):
    with open(path) as f_in:
        data = json.loads(f_in.read())
    #print data['mean_score']
    print "%s\t%s\t%.4f" % (experiment_name,
                            experiment_param,
                            data['mean_score'], )


def main():
    base_dir = sys.argv[1]
    for exp in sorted(os.listdir(base_dir)):
        dir_name = os.path.join(base_dir, exp)

        # Only directories are considered to be experiments.
        if not os.path.isdir(dir_name):
            continue

        exp_val = exp
        #print ">", dir_name

        for cnt, exp_res in sorted((int(x[:x.index('.')]), x) for x in os.listdir(dir_name)):
            exp_res_val = exp_res
            if len(exp_res.rstrip('.json')) == 2:
                exp_res_val = '0' + exp_res.rstrip('.json')

            if exp_res.endswith('.json'):
                show_experiment(
                    exp_val.replace('experiment_DataSize_slotvals=', ''),
                    int(exp_res.rstrip('.json')),
                    os.path.join(dir_name, exp_res)
                )


if __name__ == '__main__':
    main()
