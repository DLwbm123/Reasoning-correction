"""Small declared-support semantics prototype; not a clinical verifier or new ATMS.

Inputs are explicitly admitted signed records and declared sufficient-support rules.
Neither admission nor a source identifier establishes real-world truth. Rules form
a DAG; conflicting premises cannot license either sign of a downstream rule.
"""
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICT = "conflict"


@dataclass(frozen=True, order=True)
class Atom:
    scope: str  # exact image/view registration; registered ROI IDs belong in name
    name: str


@dataclass(frozen=True)
class Literal:
    atom: Atom
    positive: bool = True


@dataclass(frozen=True)
class Record:
    id: str
    literal: Literal
    source: str


@dataclass(frozen=True)
class Rule:
    id: str
    premises: tuple[Literal, ...]
    conclusion: Literal


@dataclass(frozen=True)
class Value:
    positive_supports: frozenset[str] = frozenset()
    negative_supports: frozenset[str] = frozenset()

    @property
    def status(self):
        if self.positive_supports and self.negative_supports:
            return Status.CONFLICT
        if self.positive_supports:
            return Status.SUPPORTED
        if self.negative_supports:
            return Status.REFUTED
        return Status.UNKNOWN


class SupportState:
    def __init__(self, atoms, rules=(), records=(), *, allowed_sources):
        self.atoms = frozenset(atoms)
        self.rules = tuple(rules)
        self.allowed_sources = frozenset(allowed_sources)
        self.children = {a: set() for a in self.atoms}
        self.by_head = {a: [] for a in self.atoms}
        ids = set()
        for rule in self.rules:
            if not rule.id or rule.id in ids or not rule.premises:
                raise ValueError("Rules need unique IDs and nonempty premises")
            ids.add(rule.id)
            self._check_literal(rule.conclusion)
            for literal in rule.premises:
                self._check_literal(literal)
                if literal.atom.scope != rule.conclusion.atom.scope:
                    raise ValueError("Cross-view support needs independent registration; unsupported here")
                self.children[literal.atom].add(rule.conclusion.atom)
            self.by_head[rule.conclusion.atom].append(rule)
        # Reject cycles, rather than assuming generated explanations are DAGs.
        self.order = []
        pending = set(self.atoms)
        while pending:
            ready = sorted(a for a in pending if not any(
                a in self.children[parent] for parent in pending))
            if not ready:
                raise ValueError("Cyclic support declarations are unsupported")
            self.order.extend(ready)
            pending.difference_update(ready)
        self.records = {}
        self.values = {a: Value() for a in self.atoms}
        self.update(add=records)

    def _check_literal(self, literal):
        if literal.atom not in self.atoms or type(literal.positive) is not bool:
            raise ValueError("Undeclared atom or invalid sign")

    def _evaluate(self, atom):
        positive, negative = set(), set()
        for record in self.records.values():
            if record.literal.atom == atom:
                (positive if record.literal.positive else negative).add("record:" + record.id)
        for rule in self.by_head[atom]:
            if all(self.values[p.atom].status == (
                Status.SUPPORTED if p.positive else Status.REFUTED
            ) for p in rule.premises):
                (positive if rule.conclusion.positive else negative).add("rule:" + rule.id)
        return Value(frozenset(positive), frozenset(negative))

    def update(self, *, retract=(), add=()):
        """Atomic explicit retraction/addition. No newest-record-wins policy.

        Returned nodes include changed support witnesses, even if status is stable.
        Source and record validation happens before mutating the state.
        """
        candidate = dict(self.records)
        changed_atoms = set()
        for rid in retract:
            if rid not in candidate:
                raise ValueError("Cannot retract an absent record")
            changed_atoms.add(candidate.pop(rid).literal.atom)
        for record in add:
            self._check_literal(record.literal)
            if (not record.id or record.id in candidate
                    or record.source not in self.allowed_sources):
                raise ValueError("Duplicate record ID or unadmitted source")
            candidate[record.id] = record
            changed_atoms.add(record.literal.atom)
        self.records = candidate
        dirty = set(changed_atoms)
        todo = list(changed_atoms)
        while todo:
            for child in self.children[todo.pop()]:
                if child not in dirty:
                    dirty.add(child)
                    todo.append(child)
        changed = set()
        for atom in self.order:
            if atom in dirty:
                value = self._evaluate(atom)
                if value != self.values[atom]:
                    changed.add(atom)
                    self.values[atom] = value
        return frozenset(changed)
