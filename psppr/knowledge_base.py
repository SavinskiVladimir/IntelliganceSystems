"""Классы для работы с базой правил."""
from dataclasses import dataclass
from lexer import LogicSyntaxError
from parser import parse, variables

@dataclass(frozen=True)
class Rule:
    identifier: str
    premises: frozenset[str]
    conclusion: str
    source: str

def get_premises(expr):
    if expr.op == "ATOM":
        return {expr.args[0]}
    if expr.op == "AND":
        return get_premises(expr.args[0]) | get_premises(expr.args[1])
    raise LogicSyntaxError("В посылке аксиомы Хорна разрешены только атомы и *")

class KnowledgeBase:
    def __init__(self):
        self.alphabet = {}
        self.rules = []
        self.axioms = []
        self.facts = {}

    def add_atom(self, name, description):
        if not name.isidentifier():
            raise LogicSyntaxError("Некорректный идентификатор")
        if name in self.alphabet:
            raise ValueError(f"Высказывание {name} уже объявлено")
        self.alphabet[name] = description

    def validate(self, expr):
        missing = variables(expr) - self.alphabet.keys()
        if missing:
            raise ValueError("Не объявлены: " + ", ".join(sorted(missing)))

    def add_rule(self, source):
        expr = parse(source)
        self.validate(expr)
        if expr.op != "IMPLIES" or expr.args[1].op != "ATOM":
            raise LogicSyntaxError("Аксиома должна иметь вид a*b -> c")
        rule = Rule(f"R{len(self.rules)+1}", frozenset(get_premises(expr.args[0])),
                    expr.args[1].args[0], source)
        self.rules.append(rule)
        self.axioms.append((rule.identifier, source, expr))
        return rule

    def add_axiom(self, source):
        expr = parse(source)
        self.validate(expr)
        if expr.op == "IMPLIES" and expr.args[1].op == "ATOM":
            try:
                return self.add_rule(source)
            except LogicSyntaxError:
                pass
        identifier = f"A{len(self.axioms) + 1}"
        self.axioms.append((identifier, source, expr))
        return identifier

    def formulas(self):
        from parser import Expr
        result = [expr for _, _, expr in self.axioms]
        for name, value in self.facts.items():
            expr = Expr("ATOM", (name,))
            result.append(expr if value else Expr("NOT", (expr,)))
        return result

    def set_fact(self, name, value):
        if name not in self.alphabet:
            raise ValueError(f"Неизвестное высказывание: {name}")
        self.facts[name] = value

    def reset_facts(self):
        self.facts.clear()
