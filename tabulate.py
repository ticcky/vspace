#!/usr/bin/env python
#  encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
    tables = []
    for in_file in sys.argv[1:]:
        tables.append(np.loadtxt(in_file))

    assert len(tables) > 0

    tbl_ref = tables[0]

    n_entries = len(tables[0])
    n_tables = len(tables)

    tbl_sum = np.zeros(n_entries)
    tbl_sum2 = np.zeros(n_entries)
    for tbl in tables:
        if (tbl[:, 0:1] != tbl_ref[:, 0:1]).any():
            raise Exception("Something's not good. First cols of tables do not "
                            "match.")
        tbl_sum += tbl[:,2]
        tbl_sum2 += tbl[:,2]**2

    mean = tbl_sum / n_tables
    std_dev = np.sqrt(tbl_sum2 / n_tables - mean ** 2)

    out_table = tbl_ref.copy()
    out_table[:, 2] = mean
    out_table = np.c_[out_table, std_dev]

    np.savetxt('/dev/stdout', out_table, fmt="%.4f")

    categories = set(out_table[:, 0])

    for category in sorted(categories):
        vals = out_table[out_table[:,0] == category][:, 1:]

        plt.errorbar(vals[:, 0], vals[:, 1], vals[:, 2])

    plt.show()


    #plt.errorbar(out_table[:, 2], y, e, linestyle='None', marker='^')

    #plt.show()





if __name__ == '__main__':
    main()