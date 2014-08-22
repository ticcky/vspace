# encoding: utf8

from collections import OrderedDict
import itertools
import os
import pickle
import random  #; random.seed(0)
import sys
import time

from jinja2 import Environment, FileSystemLoader

#import matplotlib.pyplot as plt

import numpy as np  #; np.random.seed(0)

import progressbar

# import pygit2

import theano
import theano.gradient
from theano import (function, tensor as T)

# Project libs.
import bootstrap
from generator import (DialogGenerator, Act)
from tracker import (Tracker, BasicTracker)
from common import rand

# Save floatX so that we save ourselves some typing.
floatx = theano.config.floatX


# Helper functions
def urand(*args):
    return np.random.rand(*args).astype(theano.config.floatX)

def ones(shape):
    return np.ones(shape).astype(theano.config.floatX)

def zeros(shape):
    return np.zeros(shape).astype(theano.config.floatX)


class Model:
    DECODE_SUB = 1
    DECODE_DOT = 2
    decode_type = None

    acts = None
    slots = None
    values = None

    f_s_new = None

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

    def next_state_fn(self, a, last_state, U, u):
        U_act = U[a]
        u_act = u[a]
        return T.tensordot(
            U_act,
            (last_state), [[0], [0]]
        ) + u_act

    def proj_fn(self, slot_ndx, state, P):
        return T.tensordot(P[slot_ndx], state, [[0], [0]])

    def __init__(self, n_data, lat_dims, proj_dims, ontology, acts, slots, \
                                                            values):
        # Store parameters in the model.
        self.n_data = n_data
        self.lat_dims = lat_dims
        self.proj_dims = proj_dims
        self.ontology = ontology
        self.acts = acts
        self.slots = slots
        self.values = values

        # Initialize the other model parameters.
        reset_act = acts[Act("reset", None, None)]

        U_val = urand(len(acts), lat_dims, lat_dims)
        U_val[reset_act] = 0.0
        U_val *= 1.0 / n_data
        self.U = theano.shared(value=U_val, name="U")

        u_val = urand(len(acts), lat_dims)
        u_val[reset_act] = 0.0
        u_val /= lat_dims
        self.u = theano.shared(value=u_val, name="u")

        P_val = urand(len(slots), lat_dims, proj_dims)
        self.P = theano.shared(value=P_val, name="P")

        #b_val = urand(len(values), proj_dims)
        b_val = []
        for slot in self.slots:
            curr_b = -2.5 * (len(self.ontology[slot]) / 2)
            for val in self.ontology[slot]:
                b_val.append(curr_b)
                curr_b += 2.5
        b_val = np.array([b_val]).astype(floatx)
        self.b = theano.shared(value = b_val, name="b")

        a = T.iscalar(name="a")
        state = T.vector(name="state")
        state_new = self.next_state_fn(a, state, self.U, self.u)
        self.f_s_new = function([state, a], state_new)

        slot = T.iscalar(name="slot")
        proj = self.proj_fn(slot, state, self.P)
        self.f_proj_curr = function([state, slot], proj)

    def get_params(self):
        return [self.U, self.u, self.P, self.b]


class VSpace1:
    dialog_cnt = 100
    lat_dims = 10
    proj_dims = 1
    learning_rate = 0.1
    rprop_plus = 1.4
    rprop_minus = 0.5

    learning_iters = None

    values = None
    slots = None
    training_metrics = None

    def __init__(self, learning_iters):
        self.learning_iters = learning_iters

        # Generate some dialogs and prepare training data.
        self.gen = DialogGenerator()
        acts, values, slots = self.prepare_data()

        # Create new model.
        self.model = Model(n_data=len(self.training_labels),
                           lat_dims=self.lat_dims,
                           proj_dims=self.proj_dims,
                           ontology=self.gen.ontology,
                           acts=acts,
                           values=values,
                           slots=slots)

        """s = self.zeros(self.lat_dims)
        for i, a in enumerate(self.training_acts):
            s = np.dot(U_val[a], s) + u_val[a]
            print i, s

        import ipdb; ipdb.set_trace()"""

    def prepare_training(self):
        data_cnt = len(self.training_labels)

        t_acts = T.ivector(name="t_acts")

        # Build the progression of states.
        s0 = T.as_tensor_variable(np.asarray(np.zeros(self.lat_dims), floatx))


        states, updates = theano.scan(self.model.next_state_fn,
                             sequences=[t_acts],
                             non_sequences=[self.model.U,
                                            self.model.u],
                             outputs_info=[s0]
                         )


        states_projection = T.tensordot(self.model.P, states, [[1], [1]])
        states_projectionx = states_projection.dimshuffle(2, 0, 1)

        #fn_states_projectionx = function([t_acts], states_projectionx)
        #print fn_states_projectionx(self.training_acts)
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
        #self.model.decode_type = self.model.DECODE_DOT
        self.model.decode_type = self.model.DECODE_SUB
        def loss_fn(proj, state, data, weight, b, data_cnt, ontology):
            loss = 0.0
            for slot, slot_ndx in self.model.slots.iteritems():
                true_proj = b[data[slot_ndx]]
                proj = T.tensordot(true_proj, self.model.P[slot_ndx],
                                 [[0], [0]])
                loss += ((state - proj)**2).sum()
                # Loss for not getting right the correct slot.
                #score_vec = (proj[slot_ndx] - b[ontology[slot_ndx]])**2
                #score = -T.log(score_vec.mean())
                #score = T.nnet.softplus(((proj[slot_ndx] - b[data[
                #    slot_ndx]])**2).sum() - 0.1)
                #score = T.nnet.softplus(abs(T.dot(proj[slot_ndx], b[data[
                #    slot_ndx]])
                #    / (proj[slot_ndx].norm(2) * b[data[slot_ndx]].norm(2))) -
                #                        0.1)
                #**2).sum()
                #loss += -T.log(1.0 / (0.0001 + score))  #T.tanh(score)
                #loss += score  #T.log(1 + score)  #T.tanh(score)



                """
                score = ((proj[slot_ndx] - b[
                    T.max([data[slot_ndx] - 1, 0])
                ])**2).sum()
                loss += T.nnet.softplus(1 - score) * T.neq(data[slot_ndx], 0)

                max_v = len(self.model.values) - 1
                score = ((proj[slot_ndx] - b[
                    T.min([data[slot_ndx] + 1, max_v])
                ])**2).sum()
                loss += T.nnet.softplus(1 - score) * T.neq(data[slot_ndx],
                                                           max_v)
                """
                # Loss for giving credit to randomly selected others.
                #for val in random.sample(self.gen.ontology[slot], 20):
                #    val_ndx = self.model.values[val]
                #    score = ((proj[slot_ndx] - b[val_ndx])**2).sum()
                #    loss += T.nnet.softplus(1 - score) * T.neq(val_ndx,
                #                                               data[slot_ndx])

            return loss * weight

        t_labels = T.imatrix(name="t_labels")
        t_ontology = T.imatrix(name="t_ontology")
        t_weights = T.vector(name="t_weights")
        losses, updates = theano.scan(loss_fn,
                                      sequences=[states_projectionx,
                                                 states,
                                                 t_labels,
                                                 t_weights],
                                      #states_projectionx,
                                      #T.as_tensor_variable(
                                      # self.training_labels)],
                                      non_sequences=[
                                                     self.model.b,
                                                     data_cnt,
                                                     t_ontology]
        )

        total_loss = losses.mean()
        #f_losses = function([t_acts, t_labels], losses)
        #print f_losses(self.training_acts, self.training_labels)

        #f_total_loss = function([t_acts, t_labels], total_loss)
        #print f_total_loss(self.training_acts, self.training_labels)

        #import ipdb; ipdb.set_trace()

        # Loss grad.
        #self.loss_grads = []
        self.shapes = []
        grads = []
        grads_history = []
        self.grads_rprop = grads_rprop = []
        grads_rprop_new = []
        for param in self.model.get_params():
            print 'param', param.name
            shape = param.shape.eval()
            self.shapes.append(shape)
            grad = T.grad(total_loss, wrt=param)
            grads.append(grad)

            # Save gradients histories for RProp.
            grad_hist = theano.shared(ones(shape), name="%s_hist" % param)
            grads_history.append(
                grad_hist
            )

            # Create variables where rprop rates will be stored.
            grad_rprop = theano.shared(ones(shape) * self.learning_rate,
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
        self._train = function(
            inputs=[t_acts, t_labels, t_weights, t_ontology],
            outputs=[total_loss, grads[0], grads[1], grads_rprop_new[0],
                     grads_rprop_new[1]],
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

    def train(self, ctrl_c_hook=None):
        self.training_metrics = {}
        self.training_metrics['begin'] = time.time()
        self.training_metrics['losses'] = losses = []

        print '>> Training:'
        for i in range(self.learning_iters):
            try:
                loss, grads_U, grads_u, rprop_grads_U, rprop_grads_u = \
                    self._train(self.training_acts,
                                self.training_labels,
                                self.training_weights,
                                self.training_ontology)

                losses.append(loss)
                print i, "loss:", loss
                #print 'grads U:', rprop_grads_U
                #print 'grads u:', rprop_grads_u
                #print
            except KeyboardInterrupt:
                if ctrl_c_hook is not None:
                    ctrl_c_hook()

            #print train_res[1]
            #print grads_rprop[3].get_value()
            #print self.model.b.get_value()

        self.training_metrics['end'] = time.time()
        #print self.loss_grads[3](self.training_acts, self.training_labels)
        #import ipdb; ipdb.set_trace()

    def visualize(self, out_filename="out/training_bs.html"):
        tracker = Tracker(self.model)
        tracker.simulate(self.training_dialogs)

        # Do bootstrap for the confusion table.
        n_bs = 1
        widgets = [progressbar.Percentage(),
                   ' ', progressbar.Bar(),
                   ' ', progressbar.ETA(),
                   ' ', progressbar.AdaptiveETA()]
        bs_progress = progressbar.ProgressBar(widgets=widgets).start()

        cts = []
        for bs_iter in bs_progress(range(n_bs)):
            n_dialogs = len(self.training_dialogs)
            if n_bs > 1:
                dataset = [random.choice(self.training_dialogs) for _ in range(
                    n_dialogs)]
            else:
                dataset = self.training_dialogs

            tracker = Tracker(self.model)
            tracker.simulate(dataset)
            cts.append(tracker.out_data['confusion_tables'])

        ct = bootstrap.from_all_confusion_tables(cts)

        context = {}
        context['tracker'] = tracker.out_data
        context['bootstrap_ct'] = ct
        context['model'] = self.model
        context['training_metrics'] = self.training_metrics
        context['training_data'] = self.training_dialogs
        env = Environment(loader=FileSystemLoader('tpl'))
        env.globals.update(zip=zip)

        tpl = env.get_template('training.html')
        with open(out_filename, "w") as f_out:
            f_out.write(tpl.render(**context))



    def ndxify_state(self, state, slots, values):
        return [values[state[slot]] for slot in slots]


    def prepare_data(self):
        """Build training data from training dialogs."""

        # Build mapping of items from the generated dialogs to indexes.
        acts = OrderedDict((dai, ndx) for dai, ndx in
                zip(self.gen.iterate_dais(), itertools.count()))

        values = OrderedDict((value, ndx) for value, ndx in
                zip(self.gen.iterate_values(), itertools.count()))

        slots = OrderedDict((slot, ndx) for slot, ndx in
                zip(self.gen.iterate_slots(), itertools.count()))

        tracker = BasicTracker(self.gen.ontology)
        tracker.new_dialog()
        blank_state = self.ndxify_state(state=tracker.get_state(),
                                        slots=slots,
                                        values=values)
        reset_act = acts[Act("reset", None, None)]

        self.training_dialogs = training_dialogs = self.gen.generate_dialogs(
            self.dialog_cnt)

        training_acts = []
        training_labels = []
        training_weights = []
        for dialog in training_dialogs:
            tracker.new_dialog()

            for i, dai in enumerate(dialog):
                true_state = tracker.next(dai)
                true_state_ndx = self.ndxify_state(state=true_state,
                                                   slots=slots,
                                                   values=values)

                training_acts.append(acts[dai])
                training_labels.append(true_state_ndx)
                if i > 2:
                    training_weights.append(1.0)
                else:
                    training_weights.append(0.0)


            # Insert reset after each dialog so that the whole training data
            # can be modelled like one sequence.
            training_acts.append(reset_act)
            training_labels.append(blank_state)

        self.training_acts = np.asarray(training_acts, dtype=np.int32)
        self.training_labels= np.asarray(training_labels, dtype=np.int32)
        self.training_weights = np.asarray(training_weights, dtype=floatx)

        t_ontology = np.zeros((len(slots), max(len(x) for x in
                                               self.gen.ontology.values())))
        for slot, vals in self.gen.ontology.iteritems():
            for i, val in enumerate(vals):
                print slots[slot], i, val, values[val]
                t_ontology[slots[slot]][i] = values[val]

        self.training_ontology = np.asarray(t_ontology, dtype=np.int32)

        return (acts, values, slots)



def git_commit():
    os.system("git add *.py")
    #os.system("git add out/*.html")
    os.system("git commit -m 'Automatic.'")


if __name__ == '__main__':
    git_commit()

    vspace = VSpace1(learning_iters=10000)
    def save():
        vspace.visualize("out/vspace2.html")

    def rprop_reset():
        for param, rprop_grad in zip(vspace.model.get_params(),
                                     vspace.grads_rprop):
            rprop_grad.set_value(rprop_grad.get_value() * 10.0)

    def ipdb_invoke():
        import ipdb; ipdb.set_trace()



    vspace.prepare_training()
    vspace.train(ctrl_c_hook=ipdb_invoke)

    save()



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
