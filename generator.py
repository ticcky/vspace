import random
import numpy as np


class DialogCorpus(list):
    pass


class Dialog(list):
    pass


class Act:
    def __init__(self, act, slot, value):
        self.act = act
        self.slot = slot
        self.value = value

    def __unicode__(self):
        return "%s(%s=%s)" % (self.act, self.slot, self.value, )

    def __eq__(self, x):
        if unicode(x) == unicode(self):
            return True
        else:
            return False

    def __hash__(self):
        return hash(unicode(self))


class DialogGenerator:
    acts = ["inform"]  #, "reject", "confirm", "deny"]

    ontology = {
        'from': ['f_null', 'f_nm', 'f_prg', 'f_brno', 'f_cb'],
        'to': ['t_null', 't_nm', 't_prg', 't_brno', 't_cb'],
        'to2': ['t2_null', 't2_nm', 't2_prg', 't2_brno', 't2_cb'],
        'time': ['tm_null', '1', '2', '3'],
    }

    def __init__(self, really_random=False):
        if not really_random:
            random.seed(0)

    def iterate_nulls(self):
        for slot, vals in self.ontology.iteritems():
            yield Act("inform", slot, vals[0])


    def iterate_dais(self):
        for act in self.acts:
            for slot, vals in self.ontology.iteritems():
                for val in vals:
                    yield Act(act, slot, val)

    def iterate_slots(self):
        return iter(self.ontology.keys())

    def iterate_values(self):
        for slot, vals in self.ontology.iteritems():
            for val in vals:
                yield val

    def generate_dialog(self, mean_n_turns, mean_n_turn_acts):
        n_turns = np.random.poisson(mean_n_turns - 1) + 1

        dialog = Dialog()
        for null in self.iterate_nulls():
            dialog.append(null)

        for turn in range(n_turns):
            res = []
            done_slots = []
            for turn_act in range(np.random.poisson(mean_n_turn_acts) + 1):
                slot = None
                while slot is None or slot in done_slots:
                    act = random.choice(self.acts)
                    slot = random.choice(self.ontology.keys())
                    value = random.choice(self.ontology[slot][1:])
                    new_res = Act(act, slot, value)

                dialog.append(new_res)
                done_slots.append(slot)

                if len(done_slots) == len(self.ontology):
                    break


        return dialog

    def generate_dialogs(self, n_dialogs):
        res = DialogCorpus()
        for i in range(n_dialogs):
            res.append(self.generate_dialog(5, 1))
        return res


def main():
    for cnt in [10, 100]: #, 1000, 10000, 100000]:
        np.save("dialogs.%d.npy" % cnt, generate_dialogs(cnt))


if __name__ == '__main__':
    main()