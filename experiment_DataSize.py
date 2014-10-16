#!/usr/bin/env python
import logging
import logging.config
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

import argparse
import tempfile
from multiprocessing import Pool
import signal
import os
import sys
import traceback


def signal_handler(signal, frame):
    print 'You pressed Ctrl+C!'
    # for p in jobs:
    #     p.terminate()
    sys.exit(0)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_experiment(*args, **kwargs):
    try:
        _run_experiment(*args, **kwargs)
    except Exception, e:
        logger.error("Process with the following args failed:" + str(args))
        
        for ln in traceback.format_exc().split("\n"):
            logger.error(ln)
        logger.error("Terminating it.")



def _run_experiment((args, n_vars_per_slot, n, )):
    tmp_dir = "/tmp/tmp_vspace_%d" % os.getpid()  #hash((n_vars_per_slot, n, ))
    os.environ['THEANO_FLAGS'] += ",base_compiledir=%s" % tmp_dir
    from vspace2 import VSpace1
    #signal.signal(signal.SIGINT, signal_handler)

    out_root = "/ha/work/people/zilka/out%s/experiment_DataSize_slotvals=%d/" % (args.out_tag, n_vars_per_slot)
    try:
        os.makedirs(out_root)
    except OSError, e:
        if not "File exists" in e.strerror:
            raise e

    logger.debug("Creating VSpace instance.")
    vspace = VSpace1(
        learning_iters=learning_iters,
        n_vars_per_slot=n_vars_per_slot,
        dialog_cnt=n,
        init_b=args.init_b,
        lat_dims=args.ndims_lat,
        proj_dims=args.ndims_proj
    )
    logger.debug("Preparing VSpace training.")
    vspace.prepare_training()
    logger.debug("Running VSpace training.")
    vspace.train()
    logger.debug("Visualizing.")
    vspace.visualize(os.path.join(out_root, "%d.html" % n),
                     os.path.join(out_root, "%d.json" % n))
    logger.debug("[n_vars_per_slot=%d,n=%d] Done." % (n_vars_per_slot, n))


if __name__ == '__main__':
    logger.info("Experiment DataSize started.")

    argp = argparse.ArgumentParser()
    argp.add_argument('--nworkers', type=int, default=1)
    argp.add_argument('--debug', action='store_true', default=False)
    argp.add_argument('--init_b', action='store_true', default=False)
    argp.add_argument('--ndims_lat', type=int, default=5)
    argp.add_argument('--ndims_proj', type=int, default=1)
    argp.add_argument('--out_tag', default='')

    args = argp.parse_args()

    #git_commit()
    n_workers = args.nworkers

    learning_iters = 500

    # Prepare all experiments.
    experiment_set = []
    if not args.debug:
        logger.info("Running full experiment.")
        #experiment_set.append((args, 5, 10))
        #for n_vars_per_slot in [15, 20, 25, 30]:
        for n_vars_per_slot in [5, 7, 9, 11, 13]:
            for n in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
                experiment_set.append((args, n_vars_per_slot, n, ))
    else:
        logger.info("Running short debug experiment.")
        experiment_set.append((args, 5, 10))


    pool = Pool(n_workers)
    try:
        pool.map(run_experiment, experiment_set)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()
