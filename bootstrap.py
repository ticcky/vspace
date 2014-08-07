from collections import defaultdict

import numpy as np

from confusion_table import ConfusionTable


def from_all_confusion_tables(cts):
    cts_by_slot = defaultdict(list)
    for ct in cts:
        for slot, slot_ct in ct.iteritems():
            cts_by_slot[slot].append(slot_ct)

    res = {}
    for slot, slot_cts in cts_by_slot.iteritems():
        res['slot'] = from_confusion_tables(slot_cts)

    return res



def from_confusion_tables(cts):
    n_rows = len(cts[0].rows)
    n_cols = len(cts[0].rows[0])
    values = cts[0].values

    mean_ct = np.zeros((n_rows, n_cols))
    stddev_ct = np.zeros((n_rows, n_cols))

    # Compute means.
    #import ipdb; ipdb.set_trace()
    for ct in cts:
        for row_id, row in enumerate(ct.rows):
            for col_id, val in enumerate(row):
                mean_ct[row_id, col_id] += val * 1.0 / len(cts)

    # Compute standard deviations.
    for ct in cts:
        for row_id, row in enumerate(ct.rows):
            for col_id, val in enumerate(row):
                stddev_ct[row_id, col_id] += (val - mean_ct[row_id, col_id])**2 / (len(cts) - 1)

    # Synthetize output table.
    res_ct = np.ndarray((n_rows, n_cols), dtype=object)
    for row_id in range(n_rows):
        for col_id in range(n_cols):
            res_ct[row_id, col_id] = "%.2f (+-%.2f)" % (mean_ct[row_id, col_id], stddev_ct[row_id, col_id])

    return ConfusionTable(res_ct, values)