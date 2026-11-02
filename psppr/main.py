"""Console interface for a propositional-logic expert system."""
import re
from lexer import LogicSyntaxError
from parser import parse, variables
from logic import evaluate, truth_table, verify_laws, simplify, format_expr
from knowledge_base import KnowledgeBase
from inference import infer
from solver import solve, entailment

HELP = """
Команды:
  alp(id, описание)        объявить высказывание (например alp(a, много запросов))
  axi(a*b -> c)            добавить любую булеву аксиому
  fact(a, true|false)      указать исходный факт
  theorem(c)              проверить следование ЛЮБОЙ формулы из базы знаний
  check                   проверить совместимость базы знаний, показать модель
  simplify(!(a*b))        преобразовать формулу законами алгебры
  eval(a*!b)              вычислить формулу по ВСЕМ заданным значениям
  table(a -> b)           вывести таблицу истинности
  laws                    проверить законы булевой алгебры
  show_alphabet           вывести все высказывания
  show_axioms             вывести все аксиомы
  show_facts              вывести факты
  reset                   удалить факты, сохранить аксиомы
  demo 1|2|3              выполнить одну из трёх демонстраций
  help                    вывести эту справку
  exit                    завершить программу
Операторы: ! (НЕ), * (И), + (ИЛИ), -> (импликация), круглые скобки.
Приоритет: !, *, +, ->. Для произвольных теорем используется SAT-проверка.
Для правил Хорна дополнительно печатается пошаговый вывод.
UNKNOWN не означает FALSE. Несовместимая база помечается INCONSISTENT.
"""

def make_demo(case):
    kb = KnowledgeBase()
    descriptions = {
        "a": "высокая частота запросов",
        "b": "подозрительный User-Agent",
        "c": "повторные запросы с одного IP",
        "d": "критический ресурс",
        "e": "подозрительное поведение",
        "f": "активность бота",
        "g": "необходима блокировка",
    }
    for name, description in descriptions.items():
        kb.add_atom(name, description)
    for source in ("a*b -> e", "e*c -> f", "f*d -> g"):
        kb.add_rule(source)
    for name in "abc":
        kb.set_fact(name, True)
    kb.set_fact("d", case != 2)
    if case == 3:
        kb.set_fact("g", False)
    return kb

def print_proof(kb, goal):
    result = infer(kb)
    print("Начальные факты:", ", ".join(f"{k}={str(v).upper()}" for k, v in kb.facts.items()) or "нет")
    for i, step in enumerate(result.steps, 1):
        print(f"Шаг {i}: {step}")
    if result.conflicts:
        for conflict in result.conflicts:
            print("ПРОТИВОРЕЧИЕ:", conflict)
    query = parse(goal)
    answer = entailment(kb.formulas(), query, kb.alphabet)
    labels = {
        "ENTAILED": "TRUE (следует из базы знаний)",
        "REFUTED": "FALSE (опровержение следует из базы знаний)",
        "UNDETERMINED": "UNKNOWN (ни утверждение, ни отрицание не доказано)",
        "INCONSISTENT": "INCONSISTENT (база знаний противоречива)",
    }
    print(f"Теорема {goal}: {labels[answer.status]}")
    def show_model(title, model):
        print(title)
        for name, value in sorted(model.items()):
            description = kb.alphabet.get(name, "")
            detail = f" — {description}" if description else ""
            print(f"  {name} = {'TRUE' if value else 'FALSE'}{detail}")

    if answer.status == "ENTAILED":
        print("Объяснение: во всех допустимых моделях базы знаний теорема истинна.")
    elif answer.status == "REFUTED":
        print("Объяснение: во всех допустимых моделях теорема ложна.")
        show_model("Пример допустимой модели:", answer.false_model)
    elif answer.status == "UNDETERMINED":
        print("Объяснение: база допускает и истинность, и ложность теоремы.")
        show_model("Модель 1 — теорема FALSE:", answer.false_model)
        show_model("Модель 2 — теорема TRUE:", answer.true_model)
        print("Вывод: исходных сведений недостаточно для однозначного ответа.")
    else:
        print("Объяснение: ни одна интерпретация не удовлетворяет всем фактам и аксиомам.")
        print("Сначала устраните противоречие в базе знаний.")

def execute(kb, line):
    command = line.strip()
    if command in ("exit", "quit"):
        return False
    if command == "help":
        print(HELP)
    elif command == "show_alphabet":
        for key, value in kb.alphabet.items():
            print(f"{key}: {value}")
    elif command == "show_axioms":
        for identifier, source, _ in kb.axioms:
            print(f"{identifier}: {source}")
    elif command == "show_facts":
        for key, value in kb.facts.items():
            print(f"{key} = {value}")
    elif command == "laws":
        for name, passed in verify_laws().items():
            print(f"{name}: {'OK' if passed else 'FAILED'}")
    elif command == "check":
        answer = solve(kb.formulas(), kb.alphabet)
        print("База знаний:", "СОВМЕСТИМА" if answer.status == "SAT" else "ПРОТИВОРЕЧИВА")
        if answer.model:
            print("Допустимая модель:", ", ".join(f"{k}={int(v)}" for k, v in sorted(answer.model.items())))
    elif command == "reset":
        kb.reset_facts()
        print("Факты очищены")
    elif re.fullmatch(r"demo\s+[123]", command):
        number = int(command.split()[1])
        print(f"Сценарий {number}")
        print_proof(make_demo(number), "g")
    else:
        match = re.fullmatch(r"([a-z_]+)\s*\((.*)\)", command, re.DOTALL)
        if not match:
            raise LogicSyntaxError("Неизвестная команда. Введите help")
        name, args = match.groups()
        args = args.strip()
        if name == "alp":
            parts = args.split(",", 1)
            if len(parts) != 2 or not parts[1].strip():
                raise LogicSyntaxError("Формат: alp(id, описание)")
            kb.add_atom(parts[0].strip(), parts[1].strip())
            print("Высказывание добавлено")
        elif name == "axi":
            added = kb.add_axiom(args)
            print(f"Аксиома {added.identifier if hasattr(added, 'identifier') else added} добавлена")
        elif name == "fact":
            parts = [part.strip() for part in args.split(",")]
            if len(parts) != 2 or parts[1].lower() not in ("true", "false"):
                raise LogicSyntaxError("Формат: fact(id, true|false)")
            kb.set_fact(parts[0], parts[1].lower() == "true")
            print("Факт установлен")
        elif name == "theorem":
            kb.validate(parse(args))
            print_proof(kb, args)
        elif name == "simplify":
            expression, steps = simplify(parse(args))
            print("Результат:", format_expr(expression))
            print("Преобразования:", ", ".join(steps) or "не требуются")
        elif name == "eval":
            expr = parse(args)
            kb.validate(expr)
            missing = variables(expr) - kb.facts.keys()
            if missing:
                raise ValueError("Для eval задайте значения: " + ", ".join(sorted(missing)))
            print(evaluate(expr, kb.facts))
        elif name == "table":
            expr = parse(args)
            kb.validate(expr)
            names = sorted(variables(expr))
            print(" | ".join(names + [args]))
            for assignment, value in truth_table(expr):
                print(" | ".join(["1" if assignment[n] else "0" for n in names] + ["1" if value else "0"]))
        else:
            raise LogicSyntaxError("Неизвестная команда. Введите help")
    return True

def main():
    kb = KnowledgeBase()
    print("Логическая экспертная система. Введите help для справки.")
    while True:
        try:
            if not execute(kb, input("logic> ")):
                break
        except (LogicSyntaxError, ValueError, KeyError) as error:
            print("Ошибка:", error)
        except (EOFError, KeyboardInterrupt):
            print("\nВыход")
            break

if __name__ == "__main__":
    main()
