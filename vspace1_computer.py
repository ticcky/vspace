import theano

import numpy as np

from tracker import Tracker


def compute_gradient((model, dialog, n_data, )):
    print 'here'
    return 1
    tracker = Tracker(model)
    tracker.new_dialog()
    last_state = tracker.get_state()

    accum_loss_grad = []
    for shape in model.shapes:
        accum_loss_grad.append(np.zeros(shape, dtype=theano.config.floatX))

    total_loss = 0.0
    for act in dialog:
        act_ndx = model.acts[act]

        # Run tracker to get the new state and the true state.
        curr_state, true_state = tracker.next(act)

        # Compute the loss & gradient of the loss.
        val = [model.values[true_state[slot]] for slot in model.slots]

        total_loss += model.f_curr_slot_loss(curr_state, val)

        for param_loss_grad, accum in zip(model.loss_grads, accum_loss_grad):
            accum += 1.0 / n_data * param_loss_grad(last_state, act_ndx, val)

        last_state = curr_state

    return accum_loss_grad, total_loss