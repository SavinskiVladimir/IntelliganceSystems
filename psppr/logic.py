"""Определение булевых операций."""
from itertools import product
from parser import parse, variables

def evaluate(expr, values):
    if expr.op == "ATOM":
        return values[expr.args[0]]
    if expr.op == "NOT":
        return not evaluate(expr.args[0], values)
    a, b = expr.args
    if expr.op == "AND":
        return evaluate(a, values) and evaluate(b, values)
    if expr.op == "OR":
        return evaluate(a, values) or evaluate(b, values)
    if expr.op == "IMPLIES":
        return not evaluate(a, values) or evaluate(b, values)
    raise ValueError("Неизвестный оператор")

def truth_table(expr, limit=10):
    names = sorted(variables(expr))
    if len(names) > limit:
        raise ValueError("Максимум 10 переменных в таблице")
    for row in product((False, True), repeat=len(names)):
        assignment = dict(zip(names, row))
        yield assignment, evaluate(expr, assignment)

LAWS = {
    "de_morgan_and": ("!(a*b)", "!a+!b"),
    "de_morgan_or": ("!(a+b)", "!a*!b"),
    "double_negation": ("!!a", "a"),
    "commutative_and": ("a*b", "b*a"),
    "commutative_or": ("a+b", "b+a"),
    "associative_and": ("(a*b)*c", "a*(b*c)"),
    "associative_or": ("(a+b)+c", "a+(b+c)"),
    "distributive_and": ("a*(b+c)", "(a*b)+(a*c)"),
    "distributive_or": ("a+(b*c)", "(a+b)*(a+c)"),
    "idempotent_and": ("a*a", "a"),
    "idempotent_or": ("a+a", "a"),
    "absorption_and": ("a*(a+b)", "a"),
    "absorption_or": ("a+(a*b)", "a"),
    "implication": ("a->b", "!a+b"),
}

def verify_laws():
    results = {}
    for name, (left, right) in LAWS.items():
        a, b = parse(left), parse(right)
        names = sorted(variables(a) | variables(b))
        results[name] = all(evaluate(a, dict(zip(names, row))) == evaluate(b, dict(zip(names, row)))
                            for row in product((False, True), repeat=len(names)))
    return results

def format_expr(expr):
    if expr.op == "ATOM":
        return expr.args[0]
    if expr.op == "NOT":
        child = expr.args[0]
        return "!" + (format_expr(child) if child.op in ("ATOM", "NOT") else f"({format_expr(child)})")
    symbols = {"AND": "*", "OR": "+", "IMPLIES": "->"}
    return f"({format_expr(expr.args[0])}{symbols[expr.op]}{format_expr(expr.args[1])})"

def simplify(expr):
    """Упрощение логических выражений."""
    from parser import Expr
    steps = []
    def walk(node):
        if node.op == "ATOM":
            return node
        children = tuple(walk(part) for part in node.args)
        node = Expr(node.op, children)
        if node.op == "IMPLIES":
            steps.append("Устранение импликации")
            return walk(Expr("OR", (Expr("NOT", (children[0],)), children[1])))
        if node.op == "NOT":
            child = children[0]
            if child.op == "NOT":
                steps.append("Двойное отрицание")
                return walk(child.args[0])
            if child.op in ("AND", "OR"):
                steps.append("Закон де Моргана")
                op = "OR" if child.op == "AND" else "AND"
                return walk(Expr(op, tuple(Expr("NOT", (x,)) for x in child.args)))
        if node.op in ("AND", "OR"):
            a, b = children
            if a == b:
                steps.append("Идемпотентность")
                return a
            other = "OR" if node.op == "AND" else "AND"
            if b.op == other and a in b.args:
                steps.append("Поглощение")
                return a
            if a.op == other and b in a.args:
                steps.append("Поглощение")
                return b
        return node
    previous = None
    while expr != previous:
        previous, expr = expr, walk(expr)
    return expr, steps
