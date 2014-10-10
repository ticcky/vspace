#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from multiprocessing import Pool
import os

from vspace2 import *


def run_experiment((n_vars_per_slot, n, )):
    out_root = "out/experiment_DataSize_slotvals=%d/" % n_vars_per_slot
    try:
        os.mkdir(out_root)
    except OSError, e:
        if not "File exists" in e.strerror:
            raise e

    logger.debug("Creating VSpace instance.")
    vspace = VSpace1(learning_iters=learning_iters,
                    n_vars_per_slot=n_vars_per_slot,
                    dialog_cnt=n)
    logger.debug("Preparing VSpace training.")
    vspace.prepare_training()
    logger.debug("Running VSpace training.")
    vspace.train()
    logger.debug("Visualizing.")
    vspace.visualize(os.path.join(out_root, "%d.html" % n),
                     os.path.join(out_root, "%d.pickle" % n))
    logger.debug("[n_vars_per_slot=%d,n=%d] Done." % (n_vars_per_slot, n))


if __name__ == '__main__':
    logger.info("Experiment DataSize started.")

    git_commit()

    learning_iters = 1000

    experiment_set = []
    for n_vars_per_slot in [15, 20, 25, 30]:
        for n in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            experiment_set.append((n_vars_per_slot, n, ))

    pool = Pool(1)
    pool.map(run_experiment, experiment_set)
    pool.join()
