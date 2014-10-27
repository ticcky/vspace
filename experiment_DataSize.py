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
sys.setrecursionlimit(50000) 

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
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu," \
                                 "base_compiledir=%s" % tmp_dir
    from vspace2 import VSpace1
    #signal.signal(signal.SIGINT, signal_handler)
    exp_dir = "n=%d,n_vars_per_slot=%d,init_b=%s,ndims_lat=%d,ndims_proj=%d" \
              ",loss=%s" % (n, n_vars_per_slot, args.init_b, args.ndims_lat, args.ndims_proj,
                 args.loss)

    out_root = os.path.join("%s%s" % (args.out_root, args.out_tag, ), exp_dir)
    # "/ha/work/people/zilka/out%s/experiment_DataSize_slotvals=%d/"
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
        proj_dims=args.ndims_proj,
        loss=args.loss,
        n_neg_samples=args.n_neg_samples
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
    argp.add_argument('--init_b', action='store_true', default=False)
    argp.add_argument('--ndims_lat', type=int, default=5)
    argp.add_argument('--ndims_proj', type=int, default=1)
    argp.add_argument('--n_neg_samples', type=int, default=5)
    argp.add_argument('--out_tag', default='')
    argp.add_argument('--vars_per_slot', default='')
    argp.add_argument('--data_sizes', default='')
    argp.add_argument('--out_root', default='"out%s/experiment_DataSize_slotvals=%d/')
    argp.add_argument('--loss', default='')
    argp.add_argument('--debug', action='store_true', default=False)

    args = argp.parse_args()

    #git_commit()
    n_workers = args.nworkers

    learning_iters = 1000

    # Prepare all experiments.
    experiment_set = []

    logger.info("Running full experiment.")
    vars_per_slot = [int(i) for i in args.vars_per_slot.split(",")]
    data_sizes = [int(i) for i in args.data_sizes.split(",")]
    for n_vars_per_slot in vars_per_slot:
        for n in data_sizes:
            experiment_set.append((args, n_vars_per_slot, n, ))

    if args.debug:
        logger.warning("Running in debug mode. Only ONE experiment!")
        _run_experiment(experiment_set.pop())
    else:
        pool = Pool(n_workers)
        try:
            pool.map_async(run_experiment, experiment_set).get(9999999)
            pool.close()
            pool.join()

        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
