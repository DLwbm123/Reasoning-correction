"""Synthetic semantics checks only; no medical observations or patients."""
import itertools
import unittest

from support_state import Atom, Literal as L, Record, Rule, Status, SupportState


class SupportSemantics(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d, self.u = [Atom("view-1", x) for x in "abcdu"]
        self.atoms = (self.a, self.b, self.c, self.d, self.u)
        self.rules = (
            Rule("ab-c", (L(self.a), L(self.b)), L(self.c)),
            Rule("u-c", (L(self.u),), L(self.c)),
            Rule("c-d", (L(self.c),), L(self.d)),
        )

    def state(self, records=()):
        return SupportState(self.atoms, self.rules, records, allowed_sources={"registered"})

    def rec(self, name, atom, positive=True):
        return Record(name, L(atom, positive), "registered")

    def test_alternative_support_survives_retraction_then_last_support_expires(self):
        s = self.state([self.rec("a", self.a), self.rec("b", self.b), self.rec("u", self.u)])
        changed = s.update(retract=["a"])
        self.assertEqual(s.values[self.c].status, Status.SUPPORTED)
        self.assertEqual(s.values[self.d].status, Status.SUPPORTED)
        self.assertIn(self.c, changed)  # lost one witness, still supported
        s.update(retract=["u"])
        self.assertEqual(s.values[self.c].status, Status.UNKNOWN)
        self.assertEqual(s.values[self.d].status, Status.UNKNOWN)
        self.assertEqual(s.values[self.b].status, Status.SUPPORTED)

    def test_conflict_does_not_propagate_as_positive_or_negative_truth(self):
        s = self.state([self.rec("a", self.a), self.rec("b", self.b)])
        s.update(add=[self.rec("not-a", self.a, False)])
        self.assertEqual(s.values[self.a].status, Status.CONFLICT)
        self.assertEqual(s.values[self.c].status, Status.UNKNOWN)
        s.update(retract=["a"])
        self.assertEqual(s.values[self.a].status, Status.REFUTED)
        self.assertEqual(s.values[self.c].status, Status.UNKNOWN)  # no contraposition

    def test_negative_rule_requires_explicit_clean_negative_support(self):
        rules = [Rule("not-a-not-c", (L(self.a, False),), L(self.c, False))]
        s = SupportState(self.atoms, rules, allowed_sources={"registered"})
        self.assertEqual(s.values[self.c].status, Status.UNKNOWN)
        s.update(add=[self.rec("not-a", self.a, False)])
        self.assertEqual(s.values[self.c].status, Status.REFUTED)

    def test_scope_and_source_and_atomic_failures(self):
        other = Atom("view-2", "a")
        s = SupportState([self.a, other], records=[self.rec("a", self.a)], allowed_sources={"registered"})
        self.assertEqual(s.values[other].status, Status.UNKNOWN)
        with self.assertRaises(ValueError):
            SupportState([self.a, other], [Rule("cross-view", (L(self.a),), L(other))],
                         allowed_sources={"registered"})
        old_records, old_values = dict(s.records), dict(s.values)
        for record in [Record("bad", L(other), "model-self-certification"), self.rec("a", other)]:
            with self.assertRaises(ValueError):
                s.update(retract=["a"], add=[record, record])
            self.assertEqual(s.records, old_records)
            self.assertEqual(s.values, old_values)

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            SupportState([self.a, self.b], [
                Rule("a-b", (L(self.a),), L(self.b)),
                Rule("b-a", (L(self.b),), L(self.a)),
            ], allowed_sources={"registered"})

    def test_incremental_matches_fresh_recompute_for_all_two_atom_transitions(self):
        # Four signed evidence bits => 16 evidence states; all 256 ordered pairs.
        # Exhaustive state equivalence is an engineering invariant, not a model gain.
        records = [self.rec("a+", self.a), self.rec("a-", self.a, False),
                   self.rec("b+", self.b), self.rec("b-", self.b, False)]
        subsets = [[r for r, bit in zip(records, bits) if bit]
                   for bits in itertools.product([False, True], repeat=4)]
        for old, new in itertools.product(subsets, repeat=2):
            s = self.state(old)
            s.update(retract=[r.id for r in old], add=new)
            self.assertEqual(s.values, self.state(new).values)


if __name__ == "__main__":
    unittest.main()
