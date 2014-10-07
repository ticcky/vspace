#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from multiprocessing import Pool
import os

from vspace2 import *


def run_experiment(n_vars_per_slot, n):
    out_root = "out/experiment_DataSize_slotvals=%d/" % n_vars_per_slot

    vspace = VSpace1(learning_iters=learning_iters,
                    n_vars_per_slot=n_vars_per_slot,
                    dialog_cnt=n)
    vspace.prepare_training()
    vspace.train()
    vspace.visualize(os.path.join(out_root, "%d.html" % n))
    logger.debug("[n_vars_per_slot=%d,n=%d] Done." % (n_vars_per_slot, n))


if __name__ == '__main__':
    git_commit()

    learning_iters = 1000

    try:
        os.mkdir(out_root)
    except OSError, e:
        if not "File exists" in e.strerror:
            raise e

    experiment_set = []
    for n_vars_per_slot in [10, 15, 20]:
        for n in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            experiment_set.append((n_vars_per_slot, n, ))

    pool = Pool(10)
    pool.map(run_experiment, experiment_set)
