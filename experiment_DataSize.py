#!/usr/bin/env python
import logging
import logging.config
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

import tempfile
from multiprocessing import Pool
import signal
import os
import sys
import argparse


def signal_handler(signal, frame):
    print 'You pressed Ctrl+C!'
    # for p in jobs:
    #     p.terminate()
    sys.exit(0)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_experiment((n_vars_per_slot, n, )):
    tmp_dir = tempfile.mkdtemp(prefix='tmp_vspace_')
    os.environ['THEANO_FLAGS'] += ",base_compiledir=%s" % tmp_dir
    from vspace2 import VSpace1
    #signal.signal(signal.SIGINT, signal_handler)

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

    argp = argparse.ArgumentParser()
    argp.add_argument('--nworkers', type=int, default=1)
    argp.add_argument('--debug', type=bool, default=False)

    args = argp.parse_args()

    #git_commit()
    n_workers = args.nworkers

    learning_iters = 1000

    # Prepare all experiments.
    experiment_set = []
    if not args.debug:
        for n_vars_per_slot in [15, 20, 25, 30]:
            for n in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
                experiment_set.append((n_vars_per_slot, n, ))
    else:
        experiment_set.append((15, 50))


    pool = Pool(n_workers)
    try:
        pool.map_async(run_experiment, experiment_set).get(9999999)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()
