#!/usr/bin/env python
# encoding: utf8

import json
import os
import sys


def read_data(path):
    with open(path) as f_in:
        data = json.loads(f_in.read())
    return data


def main():
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument('base_dir')
    argp.add_argument('--order_by', default='')

    args = argp.parse_args()

    res = []

    base_dir = args.base_dir
    for exp in sorted(os.listdir(base_dir)):
        dir_name = os.path.join(base_dir, exp)

        # Only directories are considered to be experiments.
        if not os.path.isdir(dir_name):
            continue

        exp_val = exp
        #print ">", dir_name
        exp_parsed = {}
        vars = exp.split(',')
        for var in vars:
            name, value = var.split('=')
            exp_parsed[name] = value

        for exp_json in os.listdir(dir_name):
            if '.json' in exp_json:
                data = read_data(os.path.join(dir_name, exp_json))
                res.append((exp_parsed, float(data['mean_score'])))


    for data, value in sorted(res,
                              key=lambda d: (int(d[0]["n_vars_per_slot"]),
                                             int(d[0]["n"]))
                              ):
        #print "%s\t%s" % (json.dumps(data), "%.4f" % value, )
        print "%s\t%s\t%4f" % (data['n_vars_per_slot'], data['n'], value, )


if __name__ == '__main__':
    main()
