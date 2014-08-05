import copy
import itertools
import time
import pprint


from jinja2 import Environment, FileSystemLoader

import matplotlib.pyplot as plt

import numpy as np; np.random.seed(0)

import pygit2

from sklearn.metrics import confusion_matrix

import theano
from theano import (function, pp, tensor as T)
from theano.printing import min_informative_str
from theano.tensor.shared_randomstreams import RandomStreams

from generator import DialogGenerator


def rand(*args):
    return np.random.rand(*args).astype(theano.config.floatX)


class VSpace1:
    dialog_cnt = 10
    lat_dims = 2
    proj_dims = 3
    learning_iters = 300
    learning_rate = 0.1
    rprop_plus = 1.2
    rprop_minus = 0.5

    def __init__(self):
        self.out_data = {}
        gen = DialogGenerator()
        self.training_dialogs = gen.generate_dialogs(self.dialog_cnt)
        self.values = {dai: ndx for dai, ndx in
                zip(gen.iterate_dais(), itertools.count())}

        self.slots = {slot: ndx for slot, ndx in
                zip(gen.iterate_slots(), itertools.count())}

        class Model:
            @classmethod
            def to_observation(cls, act):
                return cls.values[act]

            ontology = gen.ontology
            slots = self.slots
            values = self.values

            lat_dims = self.lat_dims
            proj_dims = self.proj_dims

            # Current state.
            s_old = T.vector(name='s')

            # Observation index into values.
            o = T.iscalar(name='o')

            # Index into slots.
            slot = T.iscalar(name='slot')


            # Transformation matrices in the update.
            U = theano.shared(value=rand(len(self.values),
                    lat_dims, lat_dims))

            # Translation vector in the update.
            u = theano.shared(value=rand(len(self.values),
                    lat_dims),)

            # Projection matrix for reading the state by hyperplane projection.
            P = theano.shared(value=rand(len(self.slots),
                    lat_dims, proj_dims))

            # Hyperplane translation vectors.
            b_value = theano.shared(value=rand(len(self.values),
                    proj_dims))

            params = [U, u, P, b_value]


            # New state.
            s_new = T.tensordot(U[o], s_old, [[0], [0]]) + u[o]
            f_s_new = function([s_old, o], s_new)

            # Projected state.
            proj_new = T.tensordot(P[slot], s_new, [[0], [0]])
            proj_old = T.tensordot(P[slot], s_old, [[0], [0]])
            f_proj_old = function([s_old, slot], proj_old)


            # Loss.
            loss = (proj_new - b_value[o]).norm(2)
            #loss += 0.1 * (U.norm(2) + u.norm(2) + P.norm(2) + b_value.norm(2))
            f_loss = function([s_old, o, slot], loss)

            # Loss grad.
            loss_grads = []
            shapes = []
            for param in params:
                shapes.append(param.shape.eval())
                loss_grads.append(
                        function([s_old, o, slot],
                                T.grad(loss, wrt=param)))

        self.model = Model

    def learn(self):
        learning_rate = self.learning_rate

        class rprop:
            g_rprops = []
            g_histories = []

        for shape in self.model.shapes:
            rprop.g_rprops.append(np.ones(shape, dtype=theano.config.floatX) * learning_rate)
            rprop.g_histories.append(np.zeros(shape, dtype=theano.config.floatX))

        self.out_data['losses'] = []
        for step in range(self.learning_iters):
            curr_loss = self.learning_iter(learning_rate=learning_rate, rprop=rprop)
            self.out_data['losses'].append(curr_loss)
            print curr_loss

        self.out_data['data'] = self.training_dialogs

    def learning_iter(self, learning_rate, rprop):
        # Prepare accumulators for gradient.
        accum_loss_grad = []
        for shape in self.model.shapes:
            accum_loss_grad.append(np.zeros(shape, dtype=theano.config.floatX))

        n_data = sum(len(dialog) for dialog in self.training_dialogs)

        # Compute the gradient over the whole training data.
        total_loss = 0.0
        for dialog in self.training_dialogs:
            s_old = np.asarray(np.zeros(self.model.lat_dims), dtype=theano.config.floatX)
            for act in dialog:
                o = self.model.to_observation(act)
                slot = self.slots[act.slot]

                total_loss += self.model.f_loss(s_old, o, slot)

                curr_loss_grads = [loss_grad(s_old, o, slot)
                                   for loss_grad in self.model.loss_grads]

                for loss_grad, accum in zip(curr_loss_grads, accum_loss_grad):
                    accum += 1.0 / n_data * loss_grad

                s_old = self.model.f_s_new(s_old, o)

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


class TrackerState:
    threshold = 0.1
    def __init__(self, scores, true_state):
        self.scores = scores

        self.true_state = copy.copy(true_state)
        self.best_vals = {}
        for slot, vals in self.scores.iteritems():
            score, val = min((score, val) for val, score in vals.iteritems() if score is not None)
            if score < self.threshold:
                self.best_vals[slot] = val
            else:
                self.best_vals[slot] = None

    def iter_confusion_entries(self):
        for slot in self.scores:
            y_true = self.true_state[slot]
            y_pred = self.best_vals[slot]
            yield slot, y_true, y_pred


class ConfusionTable:
    def __init__(self, rows, values):
        self.rows = rows
        self.values = values


class Tracker:
    def __init__(self, model):
        self.model = model

    def new_dialog(self):
        self.state = np.zeros(self.model.lat_dims, dtype=theano.config.floatX)

    def next(self, act):
        o = self.model.values[act]
        self.state = self.model.f_s_new(self.state, o)

    def decode(self, true_state):
        slot_proj_cache = {}
        slot_scores = {}
        for dai, value in self.model.values.iteritems():
            slot = self.model.slots[dai.slot]
            if not slot in slot_proj_cache:
                slot_proj_cache[slot] = self.model.f_proj_old(self.state, slot)
                slot_scores[dai.slot] = {}
                slot_scores[dai.slot][None] = None  # So that we have easier displaying.
            pos = slot_proj_cache[slot]
            slot_scores[dai.slot][dai.value] = np.linalg.norm(pos
                    - self.model.b_value.get_value()[value], 2)


        return TrackerState(slot_scores, true_state)



    def simulate(self, dialogs):
        self.out_data = {
            'simulation': []
        }

        ct_y_true = {slot: [] for slot in self.model.slots}
        ct_y_pred = {slot: [] for slot in self.model.slots}

        for dialog in dialogs:
            dialog_out = []
            # At the beginning of the dialog, the true state is that the user
            # wants nothing.
            true_state = {slot: None for slot in self.model.slots}

            self.new_dialog()

            for act in dialog:
                self.next(act)

                # Update the true state.
                true_state[act.slot] = act.value
                decoded_state = self.decode(true_state)

                dialog_out.append(
                        (act, decoded_state))

                for slot, y_true, y_pred in decoded_state.iter_confusion_entries():
                    ct_y_true[slot].append(unicode(y_true))
                    ct_y_pred[slot].append(unicode(y_pred))


            self.out_data['simulation'].append(dialog_out)

        # Build confusion tables.
        ct = {}
        for slot in self.model.slots:
            vals = ['None'] + self.model.ontology[slot]
            ct[slot] = ConfusionTable(
                    confusion_matrix(ct_y_true[slot], ct_y_pred[slot],
                            vals),
                    vals)

        self.out_data['confusion_tables'] = ct


def git_commit():
    # Commit code to git.
    repo = pygit2.Repository(".")
    index = repo.index
    index.read()
    index.add("vspace1.py")
    index.add("generator.py")
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

    env = Environment(loader=FileSystemLoader('tpl'))
    env.globals.update(zip=zip)
    tpl = env.get_template('training.html')

    out_data = {
        'losses': []
    }

    vs = VSpace1()
    vs.learn()

    tracker = Tracker(vs.model)
    tracker.simulate(vs.training_dialogs)

    with open("out/training.html", "w") as f_out:
        f_out.write(tpl.render(tracker=tracker.out_data, vspace=vs.out_data,
                training=out_data))

    git_commit()



# Removing something.

if __name__ == '__main__':
    main()




# Log:
# Odstranuju b_slot, protoze je zatima si zbytecny.