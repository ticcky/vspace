from collections import OrderedDict
import copy
import itertools
import os
import pickle
import pprint
import random; random.seed(0)
import sys
import time

from jinja2 import Environment, FileSystemLoader

#import matplotlib.pyplot as plt

import numpy as np; np.random.seed(0)

import progressbar

# import pygit2

import theano
from theano import (function, pp, tensor as T)
from theano.printing import min_informative_str
from theano.tensor.shared_randomstreams import RandomStreams

# Project libs.
import bootstrap
from generator import DialogGenerator
from tracker import (Tracker)
from common import rand



class VSpace1:
    dialog_cnt = 100
    lat_dims = 3
    proj_dims = 1
    learning_iters = 100
    learning_rate = 1.0
    rprop_plus = 1.2
    rprop_minus = 0.5

    def __init__(self):
        self.out_data = {}
        gen = DialogGenerator()
        self.training_dialogs = gen.generate_dialogs(self.dialog_cnt)
        self.acts = OrderedDict((dai, ndx) for dai, ndx in
                zip(gen.iterate_dais(), itertools.count()))

        self.values = OrderedDict((value, ndx) for value, ndx in
                zip(gen.iterate_values(), itertools.count()))

        self.slots = OrderedDict((slot, ndx) for slot, ndx in
                zip(gen.iterate_slots(), itertools.count()))

        class Model:
            ontology = gen.ontology
            slots = self.slots
            values = self.values
            acts = self.acts

            lat_dims = self.lat_dims
            proj_dims = self.proj_dims

            # Current state.
            s_curr = T.vector(name='s')

            # Observation index into acts.
            act = T.iscalar(name='act')

            # Index into slots.
            slot = T.iscalar(name='slot')

            # Index into values.
            val = T.ivector(name='val')

            # Transformation matrices in the update.
            U = theano.shared(value=rand(len(acts), lat_dims, lat_dims),
                    name="U")

            # Translation vector in the update.
            u = theano.shared(value=rand(len(acts), lat_dims),
                    name="u")

            # Projection matrix for reading the state by hyperplane projection.
            P = theano.shared(value=rand(len(slots), lat_dims, proj_dims),
                    name="P")

            # Hyperplane translation vectors.
            b_value = theano.shared(value=rand(len(values), proj_dims),
                    name="b")


            params = [U, u, P, b_value]

            # New state.
            s_new = T.tensordot(U[act], s_curr, [[0], [0]]) + u[act]
            f_s_new = function([s_curr, act], s_new)

            # Projected state.
            def proj(v_P, v_slot, v_state):
                return T.tensordot(v_P[v_slot], v_state, [[0], [0]]) / v_P[v_slot].norm(2)
            proj_curr = proj(P, slot, s_curr)
            proj_new = proj(P, slot, s_new)
            f_proj_curr = function([s_curr, slot], proj_curr)

            # Loss.
            curr_slot_loss = 0.0  #((proj_curr - b_value[val])**2).sum()
            new_slot_loss = 0.0  #((proj_new - b_value[val])**2).sum()
            for i_slot in range(len(self.slots)):
                new_slot_loss += ((proj(P, i_slot, s_new) - b_value[val[i_slot]])**2).sum()
                curr_slot_loss += ((proj(P, i_slot, s_curr) - b_value[val[i_slot]])**2).sum()
            #new_slot_loss +  T.nnet.softplus(1 - (proj_new - b_value[(val + 1) % len(values)]).norm(2))
            #loss += 0.1 * (U.norm(2) + u.norm(2) + P.norm(2) + b_value.norm(2))
            f_curr_slot_loss = function([s_curr, val], curr_slot_loss)

            # Loss grad.
            slot_loss_grads = []
            shapes = []
            for param in params:
                shapes.append(param.shape.eval())
                slot_loss_grads.append(
                        function([s_curr, act, val],
                                T.grad(new_slot_loss, wrt=param)))

            @classmethod
            def save_params(cls, file_name):
                with open(file_name, "w") as f_out:
                    f_out.write(pickle.dumps(cls.params))

            @classmethod
            def load_params(cls, file_name):
                with open(file_name) as f_in:
                    loaded_params = pickle.loads(f_in.read())

                    for l_param, param in zip(loaded_params, cls.params):
                        param.set_value(l_param.get_value())

        self.model = Model

    def learn(self, on_kbd_interrupt=None):
        learning_rate = self.learning_rate

        class rprop:
            g_rprops = []
            g_histories = []

        for shape in self.model.shapes:
            rprop.g_rprops.append(np.ones(shape, dtype=theano.config.floatX) * learning_rate)
            rprop.g_histories.append(np.zeros(shape, dtype=theano.config.floatX))

        self.out_data['losses'] = []
        for step in range(self.learning_iters):
            try:
                curr_loss = self.learning_iter(learning_rate=learning_rate, rprop=rprop)
                self.out_data['losses'].append(curr_loss)
                print curr_loss
            except KeyboardInterrupt:
                if on_kbd_interrupt is not None:
                    on_kbd_interrupt()

        self.out_data['data'] = self.training_dialogs

    def learning_iter(self, learning_rate, rprop, debug=True):
        if debug:
            print "> Starting learning iter"

        # Prepare accumulators for gradient.
        accum_loss_grad = []
        for shape in self.model.shapes:
            accum_loss_grad.append(np.zeros(shape, dtype=theano.config.floatX))

        n_data = sum(len(dialog) for dialog in self.training_dialogs)

        # Compute the gradient over the whole training data.
        total_loss = 0.0
        tracker = Tracker(self.model)
        for dialog in self.training_dialogs:
            if debug:
                print ">> New dialog"

            tracker.new_dialog()
            last_state = tracker.get_state()
            for act in dialog:
                if debug:
                    print ">>>", unicode(act)

                act_ndx = self.model.acts[act]

                # Run tracker to get the new state and the true state.
                curr_state, true_state = tracker.next(act)

                # Prepare array for acummulating the gradient at this time in dialog.
                curr_loss_grads = []
                for shape in self.model.shapes:
                    curr_loss_grads.append(np.zeros(shape, dtype=theano.config.floatX))

                # Compute the loss & gradient of the loss.
                val = [self.values[true_state[slot]] for slot in self.slots]

                if debug:
                    print ">>>> true state:", true_state
                    print ">>>> vals:", val
                    print ">>>> loss:", self.model.f_curr_slot_loss(curr_state, val)

                total_loss += self.model.f_curr_slot_loss(curr_state, val)

                for i, slot_loss_grad in enumerate(self.model.slot_loss_grads):
                    curr_loss_grads[i] += slot_loss_grad(last_state, act_ndx, val)


                for loss_grad, accum in zip(curr_loss_grads, accum_loss_grad):
                    accum += 1.0 / n_data * loss_grad

                last_state = curr_state

        # Update RPROP variables according to the resulting gradient.
        for g_rprop, total_g_loss, g_history in zip(rprop.g_rprops, accum_loss_grad, rprop.g_histories):
            g_rprop[np.where(np.sign(total_g_loss) == np.sign(g_history))] *= self.rprop_plus
            g_rprop[np.where(np.sign(total_g_loss) != np.sign(g_history))] *= self.rprop_minus

        # Save gradients for the next step for RProp.
        rprop.g_histories = accum_loss_grad

        # Update the gradient.
        for acumm, param, g_rprop in zip(accum_loss_grad, self.model.params, rprop.g_rprops):
            param.set_value(param.get_value() - g_rprop * (1 * np.sign(acumm)))

        return total_loss





def git_commit():
    os.system("git add *.py")
    os.system("git add out/*.html")
    os.system("git commit -am 'Automatic.'")
    return
    # Commit code to git.
    repo = pygit2.Repository(".")
    index = repo.index
    index.read()
    for python_file in [x for x in os.listdir('.') if x.endswith('.py')]:
        index.add(python_file)
    index.add("out/training.html")
    index.write()
    tree = index.write_tree()

    head = repo.lookup_reference('HEAD')
    head = head.resolve()

    author = pygit2.Signature('autogit', 'autogit@zilka.me')
    repo.create_commit('refs/heads/master', author, author, 'automatic',
            tree, [head.target])

def main():
    git_commit()

    vs = train_and_visualize()

    git_commit()

def main2():
    vs = load_model("out/training_bs.model")
    visualize(vs)


def train_and_visualize():
    vs = VSpace1()
    def save_result():
        visualize(vs)

    try:
        vs.learn(on_kbd_interrupt=save_result)
    except KeyboardInterrupt:
        save_result()
        vs.model.save_params("out/training_bs.model")
        print 'OK interrupting learning'

    vs.model.save_params("out/training_bs.model")
    save_result()

    return vs


def load_model(file_name):
    vs = VSpace1()
    vs.model.load_params(file_name)

    return vs


def visualize(vs):
    # Do bootstrap for the confusion table.
    n_bs = 10
    widgets = [progressbar.Percentage(),
               ' ', progressbar.Bar(),
               ' ', progressbar.ETA(),
               ' ', progressbar.AdaptiveETA()]
    bs_progress = progressbar.ProgressBar(widgets=widgets).start()
    cts = []
    for bs_iter in bs_progress(range(n_bs)):
        n_dialogs = len(vs.training_dialogs)
        dataset = [random.choice(vs.training_dialogs) for _ in range(n_dialogs)]
        tracker = Tracker(vs.model)
        tracker.simulate(dataset)
        cts.append(tracker.out_data['confusion_tables'])


    ct = bootstrap.from_all_confusion_tables(cts)


    env = Environment(loader=FileSystemLoader('tpl'))
    env.globals.update(zip=zip)
    tpl = env.get_template('training.html')

    with open("out/training_bs.html", "w") as f_out:
        f_out.write(tpl.render(tracker=tracker.out_data, vspace=vs.out_data,
                bootstrap_ct=ct, model=vs.model))


if __name__ == '__main__':
    if sys.argv[1] == "run":
        main()
    elif sys.argv[1] == "vis":
        main2()
    else:
        print "Doing nothing."





# Log:
# Odstranuju b_slot, protoze je zatima si zbytecny.