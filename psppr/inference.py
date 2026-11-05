"""Механизм вывода."""
from dataclasses import dataclass

@dataclass
class Result:
    known: set[str]
    steps: list[str]
    conflicts: list[str]

def infer(kb):
    known = {name for name, value in kb.facts.items() if value}
    false_facts = {name for name, value in kb.facts.items() if not value}
    steps, conflicts = [], []
    changed = True
    while changed:
        changed = False
        for rule in kb.rules:
            if rule.premises <= known and rule.conclusion not in known:
                known.add(rule.conclusion)
                steps.append(f"{rule.identifier}: {rule.source} => {rule.conclusion}")
                if rule.conclusion in false_facts:
                    conflicts.append(f"{rule.conclusion}: задано FALSE, но выведено TRUE ({rule.identifier})")
                changed = True
    return Result(known, steps, conflicts)
