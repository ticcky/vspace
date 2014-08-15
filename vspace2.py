# encoding: utf8

from collections import OrderedDict
import itertools
import os
import pickle
import random  #; random.seed(0)
import sys

from jinja2 import Environment, FileSystemLoader

#import matplotlib.pyplot as plt

import numpy as np  #; np.random.seed(0)

import progressbar

# import pygit2

import theano
from theano import (function, tensor as T)

# Project libs.
import bootstrap
from generator import (DialogGenerator, Act)
from tracker import (Tracker, BasicTracker)
from common import rand


class Model:
    # Current state.
    s_curr = T.vector(name='s')

    # Observation index into acts.
    act = T.iscalar(name='act')

    # Index into slots.
    slot = T.iscalar(name='slot')

    # Index into values.
    val = T.ivector(name='val')

    # Transformation matrices in the update.
    U = theano.shared(value=None, name="U")

    # Translation vector in the update.
    u = theano.shared(value=None, name="u")

    # Projection matrix for reading the state by hyperplane projection.
    P = theano.shared(value=None, name="P")

    # Hyperplane translation vectors.
    b_value = theano.shared(value=None, name="b")

    params = [U, u, P, b_value]


class VSpace1:
    dialog_cnt = 100
    lat_dims = 10
    proj_dims = 1
    learning_iters = 20000
    learning_rate = 0.1
    rprop_plus = 1.2
    rprop_minus = 0.5
    n_processes = 4


    def __init__(self):
        # Generate some dialogs and prepare training data.
        self.gen = DialogGenerator()
        self.prepare_data()

        # Create new model.
        self.model = Model()



    def prepare_data(self):
        """Build training data from training dialogs."""

        # Build mapping of items from the generated dialogs to indexes.
        self.acts = OrderedDict((dai, ndx) for dai, ndx in
                zip(self.gen.iterate_dais(), itertools.count()))

        self.values = OrderedDict((value, ndx) for value, ndx in
                zip(self.gen.iterate_values(), itertools.count()))

        self.slots = OrderedDict((slot, ndx) for slot, ndx in
                zip(self.gen.iterate_slots(), itertools.count()))

        tracker = BasicTracker(self.gen.ontology)
        tracker.new_dialog()
        blank_state = tracker.get_state()

        training_dialogs = self.gen.generate_dialogs(self.dialog_cnt)
        training_data = []
        for dialog in training_dialogs:
            tracker.new_dialog()

            for dai in dialog:
                true_state = tracker.next(dai)
                true_state_ndx = [self.values[true_state[slot]] for slot in
                                  self.slots]

                training_data.append((self.acts[dai], true_state_ndx,))


            # Insert reset after each dialog so that the whole training data
            # can be modelled like one sequence.
            training_data.append((self.acts[Act("reset", None, None)],
                                  blank_state, ))

        self.training_data = training_data



if __name__ == '__main__':
    VSpace1()