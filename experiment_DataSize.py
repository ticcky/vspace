#!/usr/bin/env python

import os

from vspace2 import *


if __name__ == '__main__':
    git_commit()

    try:
        os.mkdir("out/experiment_DataSize/")
    except OSError, e:
        if not "File exists" in e.strerror:
            raise e

    for n in [10, 100, 1000]:
        print "> Running with parameters (n=%d)" % n

        vspace = VSpace1(learning_iters=1000, dialog_cnt=n)
        vspace.prepare_training()
        vspace.train()
        vspace.visualize("out/experiment_DataSize/%d.html" % n)