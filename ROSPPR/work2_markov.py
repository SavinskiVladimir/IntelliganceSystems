"""
Практическая работа №2 — интерактивная проверка моделей на Марковских цепях.

Новый зритель, которого нет в обучающих данных, вводит фильмы, которые
он посмотрел, в порядке просмотра. Скрипт применяет к его последовательности
уже обученные модели из ноутбука 02_markov_chains.ipynb:
  * дискретная цепь - состояние на шаге n (персональная матрица переходов,
    стянутая к общей);
  * непрерывная цепь - состояние в момент времени t;
  * скрытая Марковская модель - режимы просмотра (алгоритм Витерби)
    и вероятность последовательности (прямой алгоритм);
  * 10 рекомендованных фильмов по цепи на уровне фильмов (состояние - фильм).

Модели берутся из файла data/work2_models.npz, который сохраняет ноутбук.

Запуск из папки, где лежат ml_common.py и data/:
    python work2_markov.py           # интерактивный режим
    python work2_markov.py --demo    # готовый пример без ввода
"""

import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm

import ml_common as mc

MODELS_PATH = mc.DATA_DIR / "work2_models.npz"
TOP_N = 10
MIN_MOVIES = 3
STEPS = [1, 2, 3, 5, 10]
MOMENTS = [(10 / 1440, "через 10 мин"), (1 / 24, "через 1 ч"), (6 / 24, "через 6 ч"), (1, "через сутки")]


# ---------------------------------------------------------------------------
# Загрузка моделей и данных
# ---------------------------------------------------------------------------

def load_models():
    if not MODELS_PATH.exists():
        sys.exit(f"Не найден файл {MODELS_PATH}. Сначала выполните ноутбук 02_markov_chains.ipynb.")
    data = np.load(MODELS_PATH)
    return {key: data[key] for key in data.files}


def load_movies():
    data = mc.load_slice()
    ratings = pd.concat([data["train"], data["test"]], ignore_index=True)
    movies = data["movies"].set_index("movieId")
    movies["state"] = movies["genre_list"].apply(mc.genre_group)

    # Популярность (число положительных оценок) и вероятность фильма внутри своей группы
    liked = ratings[ratings["rating"] >= mc.RELEVANCE_THRESHOLD]
    popularity = liked["movieId"].value_counts().reindex(movies.index, fill_value=0) + 0.1
    group_total = popularity.groupby(movies["state"]).transform("sum")
    movies["popularity"] = popularity
    movies["p_in_group"] = popularity / group_total
    return movies, ratings


def item_transition_counts(ratings, movie_ids, window):
    # Сколько раз фильм f оценивали в пределах window оценок после фильма j (все зрители)
    seq = ratings.sort_values(["userId", "timestamp", "movieId"])
    users = seq["userId"].to_numpy()
    m = pd.Index(movie_ids).get_indexer(seq["movieId"])
    N = np.zeros((len(movie_ids), len(movie_ids)), dtype=np.float32)
    for d in range(1, window + 1):
        same = users[d:] == users[:-d]
        np.add.at(N, (m[:-d][same], m[d:][same]), 1)
    return N


# ---------------------------------------------------------------------------
# Ввод последовательности
# ---------------------------------------------------------------------------

def format_movie(movies, movie_id, states):
    info = movies.loc[movie_id]
    return f"{info['title']}  [{', '.join(info['genre_list'])}] -> {states[info['state']]}"


def collect_sequence(movies, states):
    print("\nВведите фильмы, которые вы смотрели, В ПОРЯДКЕ ПРОСМОТРА (от давних к недавним).")
    print("Названия в MovieLens — на английском (например: Inception, Joker, Frozen).")
    print("Команды: часть названия — поиск, «популярные» — популярные фильмы, пустая строка — закончить.\n")
    popular = movies["popularity"].sort_values(ascending=False).index
    sequence = []
    while True:
        query = input(f"[фильмов: {len(sequence)}] Поиск: ").strip()
        if query == "":
            if len(sequence) < MIN_MOVIES:
                answer = input(f"Введено меньше {MIN_MOVIES} фильмов, прогноз будет неточным. "
                               "Закончить? (д/н): ").strip().lower()
                if answer not in ("д", "y", "да", "yes"):
                    continue
            return sequence

        if query.lower() == "популярные":
            candidates = [m for m in popular if m not in sequence][:15]
        else:
            found = movies[movies["clean_title"].str.contains(query, case=False, regex=False)]
            candidates = [m for m in found["popularity"].sort_values(ascending=False).index[:10]
                          if m not in sequence]
        if not candidates:
            print("  Ничего не найдено (или все найденные фильмы уже введены).")
            continue
        for number, movie_id in enumerate(candidates, 1):
            print(f"  {number:2}. {format_movie(movies, movie_id, states)}")
        choice = input("Номер фильма (пустая строка — новый поиск): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            sequence.append(candidates[int(choice) - 1])


def ask_days():
    while True:
        text = input("Сколько дней назад вы смотрели последний фильм? (пустая строка — сегодня): ")
        text = text.strip().replace(",", ".")
        if text == "":
            return 0.0
        try:
            value = float(text)
            if value >= 0:
                return value
        except ValueError:
            pass
        print("  Введите неотрицательное число.")


def demo_sequence(movies):
    # Пример зрителя: смотрит в основном фантастику, иногда отвлекается на боевики
    by_state = {s: movies[movies["state"] == s]["popularity"].sort_values(ascending=False).index.tolist()
                for s in movies["state"].unique()}
    sci_fi, action = by_state.get(1, []), by_state.get(2, [])
    return sci_fi[:3] + action[:2] + sci_fi[3:7], 2.0


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------

def viterbi(obs, pi, A, B):
    log_A, log_B = np.log(A + 1e-300), np.log(B + 1e-300)
    delta = np.log(pi + 1e-300) + log_B[:, obs[0]]
    psi = np.zeros((len(obs), len(pi)), dtype=int)
    for t in range(1, len(obs)):
        candidates = delta[:, None] + log_A
        psi[t] = candidates.argmax(axis=0)
        delta = candidates.max(axis=0) + log_B[:, obs[t]]
    path = np.zeros(len(obs), dtype=int)
    path[-1] = delta.argmax()
    for t in range(len(obs) - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def forward(obs, pi, A, B):
    # Нормированный прямой алгоритм: log P(O) и распределение режима на последнем шаге
    alpha = pi * B[:, obs[0]]
    log_prob = np.log(alpha.sum())
    alpha /= alpha.sum()
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
        log_prob += np.log(alpha.sum())
        alpha /= alpha.sum()
    return log_prob, alpha


def analyse(sequence, days, models, movies, N_items):
    states = list(models["states"])
    n_states = len(states)
    obs = movies.loc[sequence, "state"].to_numpy()
    P, Q = models["P_ratings"], models["Q"]
    beta = float(models["beta"])

    # Персональная матрица переходов по введённой последовательности
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (obs[:-1], obs[1:]), 1)
    P_user = (counts + beta * P) / (counts.sum(axis=1, keepdims=True) + beta)
    start = np.eye(n_states)[obs[-1]]

    print("\nВаша последовательность состояний:")
    print("  " + " → ".join(states[s] for s in obs))
    print(f"Текущее состояние: «{states[obs[-1]]}»")

    print("\n=== Дискретная цепь: распределение группы на шаге n ===")
    table = pd.DataFrame([start @ np.linalg.matrix_power(P_user, n) for n in STEPS],
                         index=pd.Index(STEPS, name="шаг n"), columns=states)
    print(table.round(3).to_string())
    print("Стационарное (общая цепь): " + ", ".join(f"{s} {p:.2f}" for s, p in zip(states, models["pi_ratings"])))

    print("\n=== Непрерывная цепь: распределение группы через время t после последнего просмотра ===")
    moments = MOMENTS + [(days, f"сейчас ({days:g} дн.)")]
    table = pd.DataFrame([start @ expm(Q * t) for t, _ in moments],
                         index=pd.Index([label for _, label in moments], name="момент"), columns=states)
    print(table.round(3).to_string())

    print("\n=== Скрытая Марковская модель ===")
    names = list(models["hmm_names"])
    path = viterbi(obs, models["hmm_pi"], models["hmm_A"], models["hmm_B"])
    log_prob, current = forward(obs, models["hmm_pi"], models["hmm_A"], models["hmm_B"])
    for movie_id, hidden in zip(sequence, path):
        print(f"  {names[hidden]:<28} {movies.loc[movie_id, 'title']}")
    print(f"log P(последовательности) = {log_prob:.2f}, на один фильм {log_prob / len(obs):.3f} "
          f"(при случайном выборе из {n_states} групп было бы {np.log(1 / n_states):.3f})")
    print("Текущий режим: " + ", ".join(f"{n} {p:.2f}" for n, p in zip(names, current)))

    # Рекомендации: цепь на уровне фильмов. Вектор начальных состояний - равные доли последних
    # item_recent введённых фильмов, вероятность следующего фильма - pi0 @ P (строки P сглажены к популярности)
    alpha, recent = float(models["item_alpha"]), int(models["item_recent"])
    rows = movies.index.get_indexer(sequence[-recent:])
    pop = (movies["popularity"] / movies["popularity"].sum()).to_numpy()
    P_rows = (N_items[rows] + alpha * pop[None, :]) / (N_items[rows].sum(axis=1, keepdims=True) + alpha)
    scores = pd.Series(P_rows.mean(axis=0), index=movies.index).drop(index=sequence)
    print(f"\n=== Рекомендации (цепь на уровне фильмов, по последним {len(rows)} фильмам) ===")
    for number, movie_id in enumerate(scores.sort_values(ascending=False).index[:TOP_N], 1):
        column = movies.index.get_loc(movie_id)
        because = sequence[-recent:][int(np.argmax(N_items[rows, column]))]
        print(f"  {number:2}. {movies.loc[movie_id, 'title']}  [{', '.join(movies.loc[movie_id, 'genre_list'])}]")
        if N_items[rows, column].max() > 0:
            print(f"      вероятность {scores[movie_id]:.4f}; чаще всего смотрят после «{movies.loc[because, 'clean_title']}»")
        else:
            print(f"      вероятность {scores[movie_id]:.4f}; популярный фильм (переходов из введённых фильмов нет)")


def main():
    models = load_models()
    print("Загрузка данных...")
    movies, ratings = load_movies()
    N_items = item_transition_counts(ratings, movies.index.to_numpy(), int(models["item_window"]))
    states = list(models["states"])
    print(f"Фильмов: {len(movies)}, состояний цепи: {len(states)} ({', '.join(states)})")

    if "--demo" in sys.argv:
        sequence, days = demo_sequence(movies)
    else:
        sequence = collect_sequence(movies, states)
        if not sequence:
            print("Фильмов нет — анализировать нечего.")
            return
        days = ask_days()

    print("\nВведённые фильмы:")
    for movie_id in sequence:
        print(f"  {format_movie(movies, movie_id, states)}")
    analyse(sequence, days, models, movies, N_items)


if __name__ == "__main__":
    main()
