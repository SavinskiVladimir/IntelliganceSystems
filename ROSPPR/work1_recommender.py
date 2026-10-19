"""
Практическая работа №1 — интерактивная проверка рекомендательной системы.

Новый зритель, которого нет в обучающих данных, оценивает несколько фильмов
и получает рекомендации тремя методами из работы:
  * user-based - по похожим зрителям;
  * item-based - по фильмам, похожим на понравившиеся;
  * гибрид (суперпозиция) - взвешенная сумма двух методов.

Так проверяется обобщающая способность системы: может ли она работать
с пользователем, которого не видела при «обучении» (проблема холодного
старта пользователя).

Запуск из папки, где лежат ml_common.py и data/ml_slice:
    python work1_recommender.py           # интерактивный режим
    python work1_recommender.py --demo    # готовый пример без ввода
"""

import sys

import numpy as np
import pandas as pd

import ml_common as mc

# Параметры методов: мера и число соседей, выбранные в ноутбуке
# 01_memory_based.ipynb по валидационной выборке.
USER_MEASURE = "Косинус"
USER_K = 80
ITEM_MEASURE = "Косинус"
ITEM_K = 320
ALPHA = 0.5          # вес user-based в гибриде
PRED_K = 40          # число соседей для прогноза оценки
TOP_N = 10
MIN_RATINGS = 3      # сколько оценок желательно ввести


# ---------------------------------------------------------------------------
# Меры сходства (те же формулы, что в ноутбуке; строки - сравниваемые объекты)
# ---------------------------------------------------------------------------

def sim_pearson(R, M, min_common=2):
    N = M @ M.T
    Sx = R @ M.T
    Sxx = (R * R) @ M.T
    num = N * (R @ R.T) - Sx * Sx.T
    var = np.clip(N * Sxx - Sx ** 2, 0, None)
    den = np.sqrt(var * var.T)
    S = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
    S[N < min_common] = 0
    np.fill_diagonal(S, 0)
    return np.clip(S, -1, 1)


def sim_cosine(R, M):
    norms = np.sqrt((R * R).sum(axis=1))
    den = np.outer(norms, norms)
    S = np.divide(R @ R.T, den, out=np.zeros_like(den), where=den > 0)
    np.fill_diagonal(S, 0)
    return S


def sim_euclidean(R, M, min_common=1):
    N = M @ M.T
    Sxx = (R * R) @ M.T
    D2 = np.clip(Sxx + Sxx.T - 2 * (R @ R.T), 0, None)
    msd = np.divide(D2, N, out=np.zeros_like(D2), where=N > 0)
    S = np.where(N >= min_common, 1 / (1 + np.sqrt(msd)), 0)
    np.fill_diagonal(S, 0)
    return S


def sim_jaccard(R, M):
    inter = M @ M.T
    counts = M.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    S = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(S, 0)
    return S


def sim_ochiai(R, M):
    inter = M @ M.T
    counts = M.sum(axis=1)
    den = np.sqrt(np.outer(counts, counts))
    S = np.divide(inter, den, out=np.zeros_like(inter), where=den > 0)
    np.fill_diagonal(S, 0)
    return S


def sim_pearson_jaccard(R, M):
    return np.clip(sim_pearson(R, M), 0, None) * sim_jaccard(R, M)


MEASURES = {
    "Пирсон": sim_pearson,
    "Косинус": sim_cosine,
    "Евклид (L2)": sim_euclidean,
    "Жаккар": sim_jaccard,
    "Отиаи": sim_ochiai,
    "Пирсон × Жаккар": sim_pearson_jaccard,
}


def keep_top_k(S, k):
    S = np.where(S > 0, S, 0)
    if k >= S.shape[1]:
        return S
    top = np.argpartition(-S, k, axis=1)[:, :k]
    result = np.zeros_like(S)
    rows = np.arange(S.shape[0])[:, None]
    result[rows, top] = S[rows, top]
    return result


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_data():
    data = mc.load_slice()
    ratings = pd.concat([data["train"], data["test"]], ignore_index=True)
    movies = data["movies"].set_index("movieId")
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(movies.index.to_numpy())

    R = np.zeros((len(user_ids), len(movie_ids)))
    R[pd.Index(user_ids).get_indexer(ratings["userId"]),
      pd.Index(movie_ids).get_indexer(ratings["movieId"])] = ratings["rating"].to_numpy()
    M = (R > 0).astype(float)
    return R, M, movies, movie_ids


# ---------------------------------------------------------------------------
# Ввод оценок
# ---------------------------------------------------------------------------

def format_movie(movies, movie_id):
    info = movies.loc[movie_id]
    return f"{info['title']}  [{', '.join(info['genre_list'])}]"


def search_movies(movies, popularity, query, limit=10):
    found = movies[movies["clean_title"].str.contains(query, case=False, regex=False)]
    order = popularity.reindex(found.index).sort_values(ascending=False)
    return order.index[:limit].tolist()


def ask_rating(prompt):
    while True:
        text = input(prompt).strip().replace(",", ".")
        if text == "":
            return None
        try:
            value = float(text)
        except ValueError:
            print("  Введите число от 0.5 до 5 (шаг 0.5) или пустую строку для отмены.")
            continue
        if 0.5 <= value <= 5 and (value * 2).is_integer():
            return value
        print("  Оценка должна быть от 0.5 до 5 с шагом 0.5.")


def collect_ratings(movies, popularity):
    print("\nНазвания фильмов в MovieLens — на английском (например: Inception, Joker, Frozen).")
    print("Команды: часть названия — поиск, «популярные» — список популярных фильмов, пустая строка — закончить.\n")
    rated = {}
    while True:
        query = input(f"[оценено фильмов: {len(rated)}] Поиск: ").strip()
        if query == "":
            if len(rated) < MIN_RATINGS:
                answer = input(f"Оценено меньше {MIN_RATINGS} фильмов, рекомендации будут неточными. "
                               "Закончить? (д/н): ").strip().lower()
                if answer not in ("д", "y", "да", "yes"):
                    continue
            return rated

        if query.lower() == "популярные":
            candidates = [m for m in popularity.index if m not in rated][:15]
        else:
            candidates = [m for m in search_movies(movies, popularity, query) if m not in rated]
        if not candidates:
            print("  Ничего не найдено (или все найденные фильмы уже оценены).")
            continue

        for number, movie_id in enumerate(candidates, 1):
            print(f"  {number:2}. {format_movie(movies, movie_id)}")
        choice = input("Номер фильма (пустая строка — новый поиск): ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(candidates):
            continue
        movie_id = candidates[int(choice) - 1]
        rating = ask_rating(f"Ваша оценка фильму «{movies.loc[movie_id, 'clean_title']}» (0.5–5): ")
        if rating is not None:
            rated[movie_id] = rating


def demo_ratings(movies, popularity):
    # Пример зрителя: любит научную фантастику и не любит мелодрамы
    rated = {}
    sci_fi = [m for m in popularity.index if "Sci-Fi" in movies.loc[m, "genre_list"]][:6]
    romance = [m for m in popularity.index
               if "Romance" in movies.loc[m, "genre_list"] and "Sci-Fi" not in movies.loc[m, "genre_list"]][:3]
    for movie_id in sci_fi:
        rated[movie_id] = 5.0
    for movie_id in romance:
        rated[movie_id] = 1.5
    return rated


# ---------------------------------------------------------------------------
# Рекомендации для нового пользователя
# ---------------------------------------------------------------------------

def recommend_for_new_user(R, M, item_sims, rated, movie_ids, movies):
    col = pd.Index(movie_ids).get_indexer(list(rated))
    r_new = np.zeros(R.shape[1])
    r_new[col] = list(rated.values())
    m_new = (r_new > 0).astype(float)
    liked_new = ((r_new >= mc.RELEVANCE_THRESHOLD) & (m_new > 0)).astype(float)

    # user-based: новый пользователь добавляется последней строкой, и сходство
    # считается заново для всех пар (для ~2000 пользователей это доли секунды)
    R_ext = np.vstack([R, r_new])
    M_ext = np.vstack([M, m_new])
    S_user = MEASURES[USER_MEASURE](R_ext, M_ext)
    s_new = S_user[-1, :-1]                                  # сходство с остальными
    neighbours = keep_top_k(s_new[None, :], USER_K)[0]
    L = ((R >= mc.RELEVANCE_THRESHOLD) & (M > 0)).astype(float)
    user_scores = neighbours @ L
    neighbour_votes = (neighbours > 0) @ L                   # сколько соседям понравилось

    # item-based: сходство фильмов уже посчитано, нужен только вектор «понравилось»
    item_scores = liked_new @ item_sims
    contribution = item_sims * liked_new[:, None]            # вклад каждого понравившегося фильма

    def normalize(x):
        return x / x.max() if x.max() > 0 else x

    hybrid_scores = ALPHA * normalize(user_scores) + (1 - ALPHA) * normalize(item_scores)

    # Прогноз оценки (user-based) для пояснения рекомендаций
    user_mean = np.divide(R.sum(axis=1), M.sum(axis=1), out=np.zeros(R.shape[0]), where=M.sum(axis=1) > 0)
    mean_new = r_new[m_new > 0].mean()
    C = (R - user_mean[:, None]) * M

    def predict(item):
        w = np.where(M[:, item] > 0, np.clip(s_new, 0, None), 0)
        if np.count_nonzero(w) > PRED_K:
            w[w < np.sort(w)[-PRED_K]] = 0
        return mean_new + (w @ C[:, item]) / w.sum() if w.sum() > 0 else np.nan

    popularity = L.sum(axis=0)
    results = {}
    for name, scores in [("user-based", user_scores), ("item-based", item_scores), ("гибрид", hybrid_scores)]:
        scores = scores + 1e-6 * popularity / popularity.max()
        scores[m_new > 0] = -np.inf
        top = np.argsort(-scores)[:TOP_N]
        rows = []
        for item in top:
            reasons = []
            if name in ("user-based", "гибрид") and neighbour_votes[item] > 0:
                reasons.append(f"понравился {int(neighbour_votes[item])} похожим зрителям")
            if name in ("item-based", "гибрид") and item_scores[item] > 0:
                because = movie_ids[np.argmax(contribution[:, item])]
                reasons.append(f"похож на «{movies.loc[because, 'clean_title']}»")
            if not reasons:
                reasons.append("популярный фильм (нет данных для личной рекомендации)")
            rows.append((movie_ids[item], predict(item), "; ".join(reasons)))
        results[name] = rows
    return results, int((neighbours > 0).sum())


def print_results(results, n_neighbours, movies):
    print(f"\nПохожих зрителей (соседей) в расчёте: {n_neighbours} из максимума {USER_K}.")
    for name, rows in results.items():
        print(f"\n=== Рекомендации: {name} ===")
        for number, (movie_id, predicted, reason) in enumerate(rows, 1):
            pred_text = f"прогноз {np.clip(predicted, 0.5, 5):.1f}" if np.isfinite(predicted) else "прогноз —"
            print(f"  {number:2}. {format_movie(movies, movie_id)}")
            print(f"      {pred_text}; {reason}")


def main():
    print("Загрузка данных...")
    R, M, movies, movie_ids = load_data()
    L = ((R >= mc.RELEVANCE_THRESHOLD) & (M > 0)).astype(float)
    popularity = pd.Series(L.sum(axis=0), index=movie_ids).sort_values(ascending=False)

    print(f"Зрителей: {R.shape[0]}, фильмов: {R.shape[1]}, оценок: {int(M.sum()):,}")
    print(f"Методы: user-based ({USER_MEASURE}, k={USER_K}), item-based ({ITEM_MEASURE}, k={ITEM_K}), "
          f"гибрид (alpha={ALPHA})")
    item_sims = keep_top_k(MEASURES[ITEM_MEASURE](R.T, M.T), ITEM_K)

    if "--demo" in sys.argv:
        rated = demo_ratings(movies, popularity)
    else:
        rated = collect_ratings(movies, popularity)
    if not rated:
        print("Оценок нет — рекомендовать не по чему.")
        return

    print("\nВаши оценки:")
    for movie_id, rating in rated.items():
        print(f"  {rating:.1f}  {format_movie(movies, movie_id)}")

    results, n_neighbours = recommend_for_new_user(R, M, item_sims, rated, movie_ids, movies)
    print_results(results, n_neighbours, movies)


if __name__ == "__main__":
    main()
