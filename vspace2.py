# encoding: utf8

# !!! NEFUNGUJE GRADIENT KDYZ JE DIALOG CNT > 3

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

# Save floatX so that we save ourselves some typing.
floatx = theano.config.floatX


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
    U = None  #theano.shared(value=None, name="U")

    # Translation vector in the update.
    u = None  #theano.shared(value=None, name="u")

    # Projection matrix for reading the state by hyperplane projection.
    P = None  #theano.shared(value=None, name="P")

    # Hyperplane translation vectors.
    b = None  #theano.shared(value=None, name="b")

    def get_params(self):
        return [self.U, self.u, self.P, self.b]


class VSpace1:
    dialog_cnt = 100
    lat_dims = 10
    proj_dims = 1
    learning_iters = 20000
    learning_rate = 0.1
    rprop_plus = 1.2
    rprop_minus = 0.5

    def __init__(self):
        # Generate some dialogs and prepare training data.
        self.gen = DialogGenerator()
        self.acts = None
        self.values = None
        self.slots = None
        self.prepare_data()

        # Create new model.
        self.model = Model()

        reset_act = self.acts[Act("reset", None, None)]

        U_val = self.urand(len(self.acts), self.lat_dims, self.lat_dims)
        U_val[reset_act] = 0.0
        U_val *= 1.0 / len(self.training_labels)
        self.model.U = theano.shared(value=U_val, name="U")
        u_val = self.urand(len(self.acts), self.lat_dims)
        u_val[reset_act] = 0.0
        u_val /= self.lat_dims
        self.model.u = theano.shared(value=u_val, name="u")
        P_val = self.urand(len(self.slots), self.lat_dims, self.proj_dims)
        self.model.P = theano.shared(value=P_val, name="P")
        b_val = self.urand(len(self.values), self.proj_dims)
        self.model.b = theano.shared(value = b_val, name="b")

        """s = self.zeros(self.lat_dims)
        for i, a in enumerate(self.training_acts):
            s = np.dot(U_val[a], s) + u_val[a]
            print i, s

        import ipdb; ipdb.set_trace()"""

        data_cnt = len(self.training_labels)

        t_acts = T.ivector(name="t_acts")

        # Build the progression of states.
        s0 = T.as_tensor_variable(np.asarray(np.zeros(self.lat_dims), floatx))
        def next_state_fn(a, last_state, U, u):
            U_act = U[a]
            u_act = u[a]
            return T.tensordot(U_act, last_state, [[0], [0]]) + u_act

        states, updates = theano.scan(next_state_fn,
                             sequences=[t_acts],
                             non_sequences=[self.model.U,
                                            self.model.u],
                             outputs_info=[s0]
                         )

        states_projection = T.tensordot(self.model.P, states, [[1], [1]])
        states_projectionx = states_projection.dimshuffle(2, 0, 1)

        fn_states_projectionx = function([t_acts], states_projectionx)
        print fn_states_projectionx(self.training_acts)
        #import ipdb; ipdb.set_trace()
        # P: 4slots, 10latdims, 1projdim
        # states: 36data, 10latdims
        # states_projection: 4slots, 1projdim, 36data
        #print 'P', self.model.P.shape.eval()
        #print 'states', states.shapdae.eval()
        #print 'proj', states_projection.shape.eval()
        #print 'b', self.model.b.shape.eval()
        #print 'projx', states_projectionx.shape.eval()
        #print 'data', len(self.training_labels)
        #test = (states_projection.dimshuffle(2, 1, 0, 'x')
        #        -self.model.b_value.dimshuffle('x', 0, 1))
        # Udelat pro kazdy slot jiny.
        #import ipdb; ipdb.set_trace()
        def loss_fn(proj, data, b, data_cnt):
            loss = 0.0
            for slot, slot_ndx in self.slots.iteritems():
                #loss += ((proj[slot_ndx] - b[data[slot_ndx]])**2).sum()
                #continue
                for val in self.gen.ontology[slot]:
                    val_ndx = self.values[val]
                    score = ((proj[slot_ndx] - b[val_ndx])**2).sum()
                    loss += T.eq(data[slot_ndx], val_ndx) * score
                    loss += T.neq(data[slot_ndx], val_ndx) * \
                            T.nnet.softplus(1 - score)

            return loss

        t_labels = T.imatrix(name="t_labels")
        losses, updates = theano.scan(loss_fn,
                                      sequences=[states_projectionx, t_labels],
                                      #states_projectionx,
                                      #T.as_tensor_variable(
                                      # self.training_labels)],
                                      non_sequences=[
                                                     self.model.b,
                                                     data_cnt]
        )

        total_loss = losses.mean()
        f_losses = function([t_acts, t_labels], losses)
        print f_losses(self.training_acts, self.training_labels)

        f_total_loss = function([t_acts, t_labels], total_loss)
        print f_total_loss(self.training_acts, self.training_labels)

        #import ipdb; ipdb.set_trace()

        # Loss grad.
        #self.loss_grads = []
        self.shapes = []
        grads = []
        grads_history = []
        grads_rprop = []
        grads_rprop_new = []
        for param in self.model.get_params():
            print 'param', param.name
            shape = param.shape.eval()
            self.shapes.append(shape)
            grad = T.grad(total_loss, wrt=param)
            grads.append(grad)

            # Save gradients histories for RProp.
            grad_hist = theano.shared(self.ones(shape), name="%s_hist" % param)
            grads_history.append(
                grad_hist
            )

            # Create variables where rprop rates will be stored.
            grad_rprop = theano.shared(self.ones(shape) * self.learning_rate,
                                  name="%s_rprop" % param)
            grads_rprop.append(grad_rprop)

            # Compute the new RProp coefficients.
            rprop_sign = T.sgn(grad_hist * grad)
            grad_rprop_new = grad_rprop * (
                T.eq(rprop_sign, 1) * self.rprop_plus
                + T.neq(rprop_sign, 1) * self.rprop_minus
            )
            grads_rprop_new.append(grad_rprop_new)


        # Build training function.
        self.train = function(
            inputs=[t_acts, t_labels],
            outputs=[total_loss, grads[3]],
            updates=[
                # Update parameters according to the RProp update rule.
                #(p, p - lr * g) for p, g in zip(self.model.get_params(), grads)
                (p, p - rg * T.sgn(g)) for p, g, rg in zip(
                #(p, p - 0.1) for p, g, rg in zip(
                    self.model.get_params(),
                    grads,
                    grads_rprop_new)
            ] + [
                # Save current gradient for the next step..
                (hg, g) for hg, g in zip(grads_history, grads)
            ] + [
                # Save the new rprop grads.
                (rg, rg_new) for rg, rg_new in zip(grads_rprop, grads_rprop_new)
            ]
        )

        print '>> Training:'
        for i in range(1000):
            train_res = self.train(self.training_acts, self.training_labels)
            print i, "loss:", float(train_res[0])
            #print train_res[1]
            #print grads_rprop[3].get_value()
            #print self.model.b.get_value()


        #print self.loss_grads[3](self.training_acts, self.training_labels)
        #import ipdb; ipdb.set_trace()

    def urand(self, *args):
        return np.random.rand(*args).astype(theano.config.floatX)

    def ones(self, shape):
        return np.ones(shape).astype(theano.config.floatX)

    def zeros(self, shape):
        return np.zeros(shape).astype(theano.config.floatX)

    def ndxify_state(self, state):
        return [self.values[state[slot]] for slot in self.slots]


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
        blank_state = self.ndxify_state(tracker.get_state())
        reset_act = self.acts[Act("reset", None, None)]

        training_dialogs = self.gen.generate_dialogs(self.dialog_cnt)

        training_acts = []
        training_labels = []
        for dialog in training_dialogs:
            tracker.new_dialog()

            for dai in dialog:
                true_state = tracker.next(dai)
                true_state_ndx = self.ndxify_state(true_state)

                training_acts.append(self.acts[dai])
                training_labels.append(true_state_ndx)

            # Insert reset after each dialog so that the whole training data
            # can be modelled like one sequence.
            training_acts.append(reset_act)
            training_labels.append(blank_state)

        self.training_acts = np.asarray(training_acts, dtype=np.int32)
        self.training_labels= np.asarray(training_labels, dtype=np.int32)



if __name__ == '__main__':
    VSpace1()


"""TRASH



        # Build the loss function.
        loss = 0.0
        for i, d in enumerate(self.training_data):
            print i
            act_ndx = d[0]
            vals_ndx = d[1:]
            for slot, slot_ndx in self.slots.iteritems():
                for val in self.gen.ontology[slot]:
                    val_ndx = self.values[val]
                    projection = states_projection[i]

                    # Compute score for the current value. Lower is better.
                    score = (projection - self.model.b[val_ndx])
                    score = (score**2).sum()

                    if vals_ndx[slot_ndx] == val_ndx:
                        loss += score
                    else:
                        loss += T.nnet.softplus(1 - score)




"""
