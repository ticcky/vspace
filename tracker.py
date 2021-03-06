# encoding: utf8

import copy

import numpy as np

from sklearn.metrics import confusion_matrix

import theano

from common import rand
from confusion_table import ConfusionTable


class TrackerState:
    threshold = 0.1
    def __init__(self, scores, true_state):
        self.scores = scores

        self.true_state = copy.copy(true_state)
        self.best_vals = {}
        for slot, vals in self.scores.iteritems():
            score, val = min((score, val) for val, score in vals.iteritems() if score is not None)
            #if score < self.threshold:
            self.best_vals[slot] = val
            #else:
            #    self.best_vals[slot] = None

    def iter_confusion_entries(self):
        for slot in self.scores:
            y_true = self.true_state[slot]
            y_pred = self.best_vals[slot]
            yield slot, y_true, y_pred


class BasicTracker:
    def __init__(self, ontology):
        self.ontology = ontology
        self.state = None

    def get_state(self):
        return self.state

    def new_dialog(self):
        self.state = {slot: vals[0] for slot, vals in self.ontology.iteritems()}

    def next(self, act):
        if act.act == "inform":
            self.state[act.slot] = act.value
        else:
            raise NotImplemented(act.act)
        return self.state


class Tracker:
    def __init__(self, model, inv=False):
        self.model = model
        self.true_state = None
        self.state = None

        # Invert projection matrix if needed.
        if inv:
            P_inv = []
            for P_slot in self.model.P.get_value():
                P_inv.append(np.linalg.pinv(P_slot).T)

            self.P = P_inv

        else:
            self.P = self.model.P.get_value()

    def get_state(self): return self.state

    def new_dialog(self):
        self.state = np.zeros(self.model.lat_dims).astype(theano.config.floatX)
        #for null in self.model.nulls:
        #    self.state = self.model.f_s_new(self.state, self.model.acts[null])

        self.true_state = {slot: self.model.ontology[slot][0] for slot in self.model.slots}

    def next(self, act):
        o = self.model.acts[act]
        self.state = self.model.f_s_new(self.state, o)
        if act.act == "inform":
            self.true_state[act.slot] = act.value
        else:
            raise NotImplemented(act.act)
        return self.state, self.true_state

    def decode(self):
        slot_proj_cache = {}
        slot_scores = {}
        for slot in self.model.slots:
            slot_ndx = self.model.slots[slot]

            proj_vector = self.model.f_proj_curr(self.state, slot_ndx, self.P)

            scores = {}
            for val in self.model.ontology[slot]:
                val_ndx = self.model.values[val]
                val_vector = self.model.b.get_value()[val_ndx]

                if self.model.decode_type == self.model.DECODE_SUB:
                    scores[val] = np.linalg.norm(val_vector - proj_vector, 2)
                elif self.model.decode_type == self.model.DECODE_DOT:
                    scores[val] = abs(np.dot(val_vector, proj_vector))
                else:
                    raise Exception('Unknown model decoder.')

            slot_scores[slot] = scores

        return TrackerState(slot_scores, self.true_state)



    def simulate(self, dialogs):
        self.out_data = {
            'simulation': []
        }

        ct_y_true = {slot: [] for slot in self.model.slots}
        ct_y_pred = {slot: [] for slot in self.model.slots}

        for dialog in dialogs:
            self.new_dialog()

            dialog_out = []
            for ndx, act in enumerate(dialog):
                self.next(act)

                decoded_state = self.decode()

                # Add decoded dialog to output.
                dialog_out.append(
                        (act, decoded_state))

                # Compute confusion table entry.
                if ndx > 2:  # Ignore first 3 bootstrap entries.
                    for slot, y_true, y_pred in decoded_state.iter_confusion_entries():
                        ct_y_true[slot].append(unicode(y_true))
                        ct_y_pred[slot].append(unicode(y_pred))


            self.out_data['simulation'].append(dialog_out)

        # Build confusion tables.
        ct = {}
        for slot in self.model.slots:
            vals = self.model.ontology[slot]
            #print ct_y_true[slot]
            #print ct_y_pred[slot]
            print slot
            ct[slot] = ConfusionTable(
                    confusion_matrix(ct_y_true[slot], ct_y_pred[slot],
                            vals),
                    vals, -1.0)

        self.out_data['confusion_tables'] = ct