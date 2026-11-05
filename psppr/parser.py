"""Парсер"""
from dataclasses import dataclass
from lexer import tokenize, LogicSyntaxError

@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple

class Parser:
    def __init__(self, source):
        self.tokens = tokenize(source)
        self.index = 0

    def accept(self, kind):
        if self.tokens[self.index].kind == kind:
            self.index += 1
            return True
        return False

    def expect(self, kind):
        token = self.tokens[self.index]
        if not self.accept(kind):
            raise LogicSyntaxError(f"Ожидался {kind}; найдено {token.value or 'конец ввода'} (позиция {token.position + 1})")
        return token

    def parse(self):
        result = self.implication()
        self.expect("EOF")
        return result

    def implication(self):
        left = self.disjunction()
        return Expr("IMPLIES", (left, self.implication())) if self.accept("IMPLIES") else left

    def disjunction(self):
        left = self.conjunction()
        while self.accept("OR"):
            left = Expr("OR", (left, self.conjunction()))
        return left

    def conjunction(self):
        left = self.negation()
        while self.accept("AND"):
            left = Expr("AND", (left, self.negation()))
        return left

    def negation(self):
        if self.accept("NOT"):
            return Expr("NOT", (self.negation(),))
        if self.accept("LPAREN"):
            result = self.implication()
            self.expect("RPAREN")
            return result
        return Expr("ATOM", (self.expect("IDENT").value,))

def parse(source):
    return Parser(source).parse()

def variables(expr):
    if expr.op == "ATOM":
        return {expr.args[0]}
    return set().union(*(variables(child) for child in expr.args))
