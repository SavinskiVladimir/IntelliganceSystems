"""Лексер"""
from dataclasses import dataclass

class LogicSyntaxError(ValueError):
    pass

@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int

def tokenize(source: str) -> list[Token]:
    tokens = []
    i = 0
    symbols = {'*': 'AND', '+': 'OR', '!': 'NOT', '(': 'LPAREN', ')': 'RPAREN'}
    while i < len(source):
        c = source[i]
        if c.isspace():
            i += 1
        elif source.startswith('->', i):
            tokens.append(Token('IMPLIES', '->', i))
            i += 2
        elif c in symbols:
            tokens.append(Token(symbols[c], c, i))
            i += 1
        elif c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            tokens.append(Token('IDENT', source[start:i], start))
        else:
            raise LogicSyntaxError(f'Недопустимый символ {c!r} в позиции {i + 1}')
    tokens.append(Token('EOF', '', len(source)))
    return tokens
