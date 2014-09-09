#!/usr/bin/env python

import os

from vspace2 import *


if __name__ == '__main__':
    git_commit()

    n_vars_per_slot = 30
    learning_iters = 1000

    out_root = "out/experiment_DataSize_slotvals=%d/" % n_vars_per_slot

    try:
        os.mkdir(out_root)
    except OSError, e:
        if not "File exists" in e.strerror:
            raise e

    for n in [10, 100, 1000]:
        print "> Running with parameters (n=%d)" % n

        vspace = VSpace1(learning_iters=learning_iters,
                         n_vars_per_slot=n_vars_per_slot,
                         dialog_cnt=n)
        vspace.prepare_training()
        vspace.train()
        vspace.visualize(os.path.join(out_root, "%d.html" % n))