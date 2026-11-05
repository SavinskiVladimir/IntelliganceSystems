"""Преобразование выражений в произвольной форме к КНФ."""
from dataclasses import dataclass
from parser import Expr, variables

@dataclass
class Answer:
    status: str
    model: dict[str, bool] | None = None
    true_model: dict[str, bool] | None = None
    false_model: dict[str, bool] | None = None

class Encoder:
    def __init__(self, names):
        self.names = {name: index + 1 for index, name in enumerate(sorted(names))}
        self.counter = len(self.names)
        self.clauses = []

    def fresh(self):
        self.counter += 1
        return self.counter

    def encode(self, expr):
        if expr.op == "ATOM":
            return self.names[expr.args[0]]
        if expr.op == "NOT":
            return -self.encode(expr.args[0])
        a, b = (self.encode(item) for item in expr.args)
        if expr.op == "IMPLIES":
            a = -a
            op = "OR"
        else:
            op = expr.op
        result = self.fresh()
        if op == "AND":
            self.clauses.extend([[-result, a], [-result, b], [result, -a, -b]])
        elif op == "OR":
            self.clauses.extend([[result, -a], [result, -b], [-result, a, b]])
        else:
            raise ValueError("Неподдерживаемый оператор")
        return result

def propagate(clauses, assigned):
    while True:
        changed = False
        for clause in clauses:
            remaining = [lit for lit in clause if abs(lit) not in assigned]
            satisfied = any((lit > 0) == assigned[abs(lit)] for lit in clause if abs(lit) in assigned)
            if satisfied:
                continue
            if not remaining:
                return False
            if len(remaining) == 1:
                lit = remaining[0]
                var, value = abs(lit), lit > 0
                if var in assigned and assigned[var] != value:
                    return False
                if var not in assigned:
                    assigned[var] = value
                    changed = True
        if not changed:
            return True

def dpll(clauses, assigned):
    assigned = assigned.copy()
    if not propagate(clauses, assigned):
        return None
    unsatisfied = [c for c in clauses if not any(abs(l) in assigned and assigned[abs(l)] == (l > 0) for l in c)]
    if not unsatisfied:
        return assigned
    clause = min(unsatisfied, key=lambda c: sum(abs(l) not in assigned for l in c))
    candidate = next(abs(lit) for lit in clause if abs(lit) not in assigned)
    for value in (True, False):
        result = dpll(clauses, {**assigned, candidate: value})
        if result is not None:
            return result
    return None

def solve(formulas, names=()):
    all_names = set(names)
    for formula in formulas:
        all_names.update(variables(formula))
    encoder = Encoder(all_names)
    for formula in formulas:
        encoder.clauses.append([encoder.encode(formula)])
    assignment = dpll(encoder.clauses, {})
    if assignment is None:
        return Answer("UNSAT")
    return Answer("SAT", {name: assignment.get(number, False)
                          for name, number in encoder.names.items()})

def entailment(formulas, goal, names=()):
    """Classify entailment and return witnesses for both goal truth values.

    Models satisfy all base formulas. No models are returned for an
    inconsistent knowledge base.
    """
    base = solve(formulas, names)
    if base.status == "UNSAT":
        return Answer("INCONSISTENT")
    not_goal = Expr("NOT", (goal,))
    counterexample = solve([*formulas, not_goal], names)
    if counterexample.status == "UNSAT":
        return Answer("ENTAILED", true_model=base.model)
    positive = solve([*formulas, goal], names)
    if positive.status == "UNSAT":
        return Answer("REFUTED", counterexample.model, false_model=counterexample.model)
    return Answer("UNDETERMINED", counterexample.model,
                  true_model=positive.model, false_model=counterexample.model)
