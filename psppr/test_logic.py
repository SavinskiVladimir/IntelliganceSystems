import unittest
from lexer import LogicSyntaxError
from parser import parse
from logic import evaluate, verify_laws, truth_table, simplify, format_expr
from knowledge_base import KnowledgeBase
from inference import infer
from main import make_demo
from solver import solve, entailment

class LogicTests(unittest.TestCase):
    def test_precedence(self):
        self.assertTrue(evaluate(parse("a+b*c"), {"a": True, "b": False, "c": False}))
        self.assertFalse(evaluate(parse("a*b->c"), {"a": True, "b": True, "c": False}))

    def test_truth_table(self):
        self.assertEqual(len(list(truth_table(parse("a+b")))), 4)

    def test_laws(self):
        self.assertTrue(all(verify_laws().values()))

    def test_invalid_rule(self):
        kb = KnowledgeBase()
        for name in "abc":
            kb.add_atom(name, name)
        with self.assertRaises(LogicSyntaxError):
            kb.add_rule("a+b -> c")

    def test_scenarios(self):
        first = infer(make_demo(1))
        self.assertIn("g", first.known)
        self.assertEqual(len(first.steps), 3)
        second = infer(make_demo(2))
        self.assertNotIn("g", second.known)
        third = infer(make_demo(3))
        self.assertTrue(third.conflicts)

    def test_arbitrary_theorems(self):
        kb = KnowledgeBase()
        for name in ("a", "b", "c"):
            kb.add_atom(name, name)
        kb.add_axiom("a -> (b+c)")
        kb.set_fact("a", True)
        kb.set_fact("b", False)
        self.assertEqual(entailment(kb.formulas(), parse("c"), kb.alphabet).status, "ENTAILED")
        self.assertEqual(entailment(kb.formulas(), parse("b"), kb.alphabet).status, "REFUTED")
        self.assertEqual(entailment(kb.formulas(), parse("a -> c"), kb.alphabet).status, "ENTAILED")

    def test_undetermined_and_conflict(self):
        kb = KnowledgeBase()
        kb.add_atom("a", "a")
        self.assertEqual(entailment(kb.formulas(), parse("a")).status, "UNDETERMINED")
        kb.add_axiom("a")
        kb.set_fact("a", False)
        self.assertEqual(solve(kb.formulas()).status, "UNSAT")
        self.assertEqual(entailment(kb.formulas(), parse("a")).status, "INCONSISTENT")

    def test_two_witness_models(self):
        kb = KnowledgeBase()
        for name in ("a", "b", "c"):
            kb.add_atom(name, name)
        kb.add_axiom("a -> (b+c)")
        kb.set_fact("a", True)
        result = entailment(kb.formulas(), parse("c"), kb.alphabet)
        self.assertEqual(result.status, "UNDETERMINED")
        self.assertFalse(evaluate(parse("c"), result.false_model))
        self.assertTrue(evaluate(parse("c"), result.true_model))
        for model in (result.false_model, result.true_model):
            self.assertTrue(all(evaluate(formula, model) for formula in kb.formulas()))

    def test_witnesses_for_definite_and_inconsistent_cases(self):
        kb = KnowledgeBase()
        kb.add_atom("a", "a")
        kb.set_fact("a", True)
        proved = entailment(kb.formulas(), parse("a"), kb.alphabet)
        self.assertEqual(proved.status, "ENTAILED")
        self.assertTrue(proved.true_model["a"])
        self.assertIsNone(proved.false_model)
        refuted = entailment(kb.formulas(), parse("!a"), kb.alphabet)
        self.assertEqual(refuted.status, "REFUTED")
        self.assertFalse(evaluate(parse("!a"), refuted.false_model))
        kb.add_axiom("!a")
        inconsistent = entailment(kb.formulas(), parse("a"), kb.alphabet)
        self.assertEqual(inconsistent.status, "INCONSISTENT")
        self.assertIsNone(inconsistent.true_model)
        self.assertIsNone(inconsistent.false_model)

    def test_simplification(self):
        expr, steps = simplify(parse("!(a*b)"))
        self.assertTrue(steps)
        self.assertEqual(format_expr(expr), "(!a+!b)")

    def test_sat_solver_matches_truth_table(self):
        from itertools import product
        from solver import solve
        from parser import Expr
        names = ("a", "b", "c")
        for bits in product((False, True), repeat=3):
            facts = [parse(name) if bit else Expr("NOT", (parse(name),))
                     for name, bit in zip(names, bits)]
            formula = parse("(a+b)*(!b->c)")
            answer = solve([*facts, formula])
            expected = evaluate(formula, dict(zip(names, bits)))
            self.assertEqual(answer.status == "SAT", expected)

if __name__ == "__main__":
    unittest.main()
