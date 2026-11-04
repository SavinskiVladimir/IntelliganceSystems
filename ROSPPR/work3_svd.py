"""
Практическая работа №3 — проверка SVD, Funk SVD и SVD++ новым зрителем.

Сначала выполните 03_svd_clustering.ipynb. Он сохраняет data/work3_models.npz.
Демонстрация использует именно эти модели без переобучения на test.

Запуск из папки проекта:
    python work3_svd.py
    python work3_svd.py --demo
    python work3_svd.py --models другой/путь/work3_models.npz

Новый зритель оценивает фильмы или отмечает взаимодействие без оценки.
При желании отдельно задаёт контрольные оценки, которые не войдут в профиль.
После рекомендаций можно оценить знакомые фильмы из выдачи; отзывы сохраняются
в data/work3_feedback.csv. Пропуск не считается отрицательной оценкой.

Нужны только numpy, pandas и scipy. ml_common.py и исходные CSV для самой
демонстрации не требуются: каталог и факторы включены в файл моделей.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from scipy.special import logsumexp

TOP_N = 10
MIN_RATINGS = 3
FOLD_REG = 10.0  # регуляризация нового профиля; не lambda обучения исходной модели


# ---------------------------------------------------------------------------
# Загрузка: только числовые/строковые массивы, без pickle
# ---------------------------------------------------------------------------

def load_models(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Не найден {path}. Сначала выполните 03_svd_clustering.ipynb.")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if int(arrays.get("format_version", -1)) != 1:
        raise ValueError("Неизвестный формат файла моделей; выполните текущий ноутбук заново.")
    movies = pd.DataFrame({"movieId": arrays["movie_ids"], "title": arrays["titles"],
                           "genres": arrays["genres"], "popularity": arrays["popularity"]}).set_index("movieId")
    movies["genre_list"] = movies["genres"].str.split("|")
    if not movies.index.is_unique: raise ValueError("В каталоге повторяются movieId")
    models = {}
    for j in range(3):
        prefix = f"m{j}_"
        name = str(arrays[prefix+"name"])
        model = {key: arrays[prefix+key] for key in ["mu", "bi", "Q", "reg"]}
        model["kind"] = name
        if name == "SVD++": model["Y"] = arrays[prefix+"Y"]
        if len(model["Q"]) != len(movies) or not np.isfinite(model["Q"]).all():
            raise ValueError(f"Повреждены факторы {name}")
        models[name] = model
    return arrays, movies, models


# ---------------------------------------------------------------------------
# Новый зритель: регуляризованный fold-in при фиксированных факторах фильмов
# ---------------------------------------------------------------------------

def fold_in(model, movie_ids, ratings, seen, ridge=FOLD_REG):
    """Оценки ratings — {movieId: rating}; seen включает и неоценённые взаимодействия."""
    index = pd.Index(movie_ids)
    seen = set(seen) | set(ratings)
    history = index.get_indexer(sorted(seen))
    if (history < 0).any(): raise ValueError("В истории есть фильм вне каталога")
    Q = model["Q"]
    implicit = np.zeros(Q.shape[1])
    if model["kind"] == "SVD++" and len(history):
        implicit = model["Y"][history].sum(axis=0)/np.sqrt(len(history))
    bias, p = 0.0, np.zeros(Q.shape[1])
    if ratings:
        rows = index.get_indexer(list(ratings))
        design = np.column_stack([np.ones(len(rows)), Q[rows]])
        target = np.array(list(ratings.values()))-float(model["mu"])-model["bi"][rows]-Q[rows] @ implicit
        theta = np.linalg.solve(design.T @ design+ridge*np.eye(design.shape[1]), design.T @ target)
        bias, p = float(theta[0]), theta[1:]
    scores = float(model["mu"])+model["bi"]+bias+Q @ (p+implicit)
    return scores, p, bias


def transform_profiles(P, mean, scale):
    X = (P-mean)/scale
    return X/np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def gmm_posterior(X, weights, means, variances):
    d = X.shape[1]
    logp = np.empty((len(X), len(weights)))
    for c in range(len(weights)):
        logp[:, c] = (np.log(max(weights[c], 1e-300))-.5*(d*np.log(2*np.pi)
                      +np.log(variances[c]).sum()+((X-means[c])**2/variances[c]).sum(axis=1)))
    normalizer = logsumexp(logp, axis=1)
    return np.exp(logp-normalizer[:, None]), normalizer


# ---------------------------------------------------------------------------
# Поиск, ввод и валидация
# ---------------------------------------------------------------------------

def format_movie(movies, movie_id):
    row = movies.loc[movie_id]
    return f"{row['title']}  [{', '.join(row['genre_list'])}]"


def ask_rating(prompt, allow_seen=False):
    while True:
        text = input(prompt).strip().replace(",", ".")
        if text == "": return None
        if allow_seen and text.lower() in ("в", "видел", "seen"): return "seen"
        try:
            value = float(text)
        except ValueError:
            value = np.nan
        if np.isfinite(value) and .5 <= value <= 5 and (value*2).is_integer(): return value
        print("Введите оценку от 0.5 до 5 с шагом 0.5 или пропустите пустой строкой."
              + (" «в» — взаимодействие без оценки." if allow_seen else ""))


def choose_movie(movies, excluded):
    while True:
        query = input("Поиск (часть английского названия / популярные / Enter — закончить): ").strip()
        if not query: return None
        if query.lower() == "популярные":
            candidates = movies.sort_values("popularity", ascending=False, kind="stable")
        else:
            candidates = movies[movies["title"].str.contains(query, case=False, regex=False)]
            candidates = candidates.sort_values("popularity", ascending=False, kind="stable")
        candidates = candidates.loc[~candidates.index.isin(excluded)].head(12)
        if candidates.empty:
            print("Ничего не найдено или эти фильмы уже введены."); continue
        for number, movie_id in enumerate(candidates.index, 1):
            print(f"  {number:2}. {format_movie(movies, movie_id)}")
        text = input("Номер фильма (Enter — новый поиск): ").strip()
        if text.isdigit() and 1 <= int(text) <= len(candidates): return candidates.index[int(text)-1]


def collect_profile(movies):
    print("\nПРОФИЛЬ. Оцените знакомые фильмы. «в» — видел/взаимодействовал, но оценки нет.")
    print("SVD++ использует и такие взаимодействия; SVD и Funk SVD учат личный профиль по оценкам.")
    ratings, seen = {}, set()
    while True:
        movie_id = choose_movie(movies, seen)
        if movie_id is None: break
        value = ask_rating("Оценка (0.5–5) / «в» / Enter — отмена: ", allow_seen=True)
        if value is None: continue
        seen.add(movie_id)
        if value != "seen": ratings[movie_id] = value
        print(f"Оценок: {len(ratings)}; всего взаимодействий: {len(seen)}")
    if len(ratings) < MIN_RATINGS:
        print(f"Оценок меньше {MIN_RATINGS}: личные факторы определяются ненадёжно.")
    return ratings, seen


def collect_controls(movies, seen):
    print("\nКОНТРОЛЬ (необязательно). Введите другие знакомые фильмы и оценки.")
    print("Они НЕ войдут в профиль: сравним прогноз с вашими ответами. Enter — пропустить.")
    controls = {}
    while True:
        movie_id = choose_movie(movies, set(seen) | set(controls))
        if movie_id is None: return controls
        value = ask_rating("Контрольная оценка (0.5–5) / Enter — отмена: ")
        if value is not None: controls[movie_id] = value


def demo_profile(movies):
    # Это условные входные предпочтения для демонстрации, а не реальные отзывы.
    popular = movies.sort_values("popularity", ascending=False, kind="stable")
    sci = popular[popular["genre_list"].apply(lambda g: "Sci-Fi" in g)].index[:6]
    romance = popular[popular["genre_list"].apply(lambda g: "Romance" in g and "Sci-Fi" not in g)].index[:3]
    ratings = {movie_id: 5.0 for movie_id in sci}
    ratings.update({movie_id: 1.5 for movie_id in romance})
    if not ratings:
        ratings = {movie_id: 4.0 for movie_id in popular.index[:3]}
    return ratings, set(ratings)


# ---------------------------------------------------------------------------
# Сравнение моделей и объяснение
# ---------------------------------------------------------------------------

def predict_all(models, movies, ratings, seen):
    scores, profiles = {}, {}
    for name, model in models.items():
        scores[name], p, bias = fold_in(model, movies.index, ratings, seen)
        profiles[name] = p
    return scores, profiles


def check_controls(scores, movies, controls):
    if not controls: return
    rows = movies.index.get_indexer(list(controls))
    y = np.array(list(controls.values()))
    table = []
    print("\n=== Контрольные фильмы: эти оценки не участвовали в fold-in ===")
    for name, all_scores in scores.items():
        predicted = np.clip(all_scores[rows], .5, 5)
        table.append({"Метод": name, "Оценок": len(y), "MAE": np.mean(abs(y-predicted)),
                      "MSE": np.mean((y-predicted)**2), "RMSE": np.sqrt(np.mean((y-predicted)**2))})
        for movie_id, actual, pred in zip(controls, y, predicted):
            print(f"{name}: {movies.loc[movie_id, 'title']} — прогноз {pred:.2f}, ваша оценка {actual:.1f}")
    print(pd.DataFrame(table).round(4).to_string(index=False))
    print("Это проверка на выбранных вами фильмах, а не оценка всего каталога.")


def print_recommendations(scores, movies, models, profiles, seen, controls):
    results = {}
    excluded = set(seen) | set(controls)
    known = movies.index.isin(excluded)
    for name, all_scores in scores.items():
        filtered = np.where(known, -np.inf, all_scores)
        top = [j for j in np.argsort(-filtered, kind="stable")[:TOP_N] if np.isfinite(filtered[j])]
        results[name] = [(movies.index[j], float(np.clip(all_scores[j], .5, 5))) for j in top]
        print(f"\n=== {name}: рекомендации ===")
        for number, j in enumerate(top, 1):
            print(f"  {number:2}. {format_movie(movies, movies.index[j])}")
            print(f"      прогноз {np.clip(all_scores[j], .5, 5):.2f}")
            # У SVD однозначно нумеруем факторы. Названия смыслов не выдумываем.
            if name == "SVD":
                contribution = profiles[name]*models[name]["Q"][j]
                f = int(np.argmax(np.abs(contribution)))
                print(f"      наибольший по модулю личный вклад: фактор {f+1}, {contribution[f]:+.3f} балла")
    return results


def print_clusters(arrays, profile, n_ratings):
    if n_ratings < MIN_RATINGS:
        print("\nНедостаточно оценок для содержательного отнесения к группе зрителей."); return
    x = transform_profiles(profile[None, :], arrays["cluster_mean"], arrays["cluster_scale"])
    resp, _ = gmm_posterior(x, arrays["gmm_weights"], arrays["gmm_means"], arrays["gmm_variances"])
    print("\n=== Группы зрителей по SVD-профилю ===")
    print("EM, вероятности принадлежности: " + ", ".join(f"{c}: {p:.3f}" for c, p in enumerate(resp[0])))
    near = int(np.linalg.norm(arrays["hier_centers"]-x, axis=1).argmin())
    print(f"Ближайший центр иерархической группы: {near}.")
    print("Это приближение для нового зрителя: иерархия не имеет собственного predict и заново не перестраивается.")


def collect_feedback(results, movies, path):
    print("\n=== Оценки рекомендаций ===")
    print("Оцените только знакомые фильмы. Неизвестный фильм пропустите (Enter).")
    answer = input("Собрать отзывы? (д/н): ").strip().lower()
    if answer not in ("д", "да", "y", "yes"): return
    unique = list(dict.fromkeys(movie_id for rows in results.values() for movie_id, _ in rows))
    ratings = {}
    for movie_id in unique:
        value = ask_rating(f"{format_movie(movies, movie_id)}\nВаша оценка / Enter — пропуск: ")
        if value is not None: ratings[movie_id] = value
    if not ratings:
        print("Оценок нет — отзывы не записаны."); return
    session = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    records = [{"session": session, "timestamp": now, "method": name, "movieId": movie_id,
                "prediction": prediction, "rating": ratings[movie_id]}
               for name, rows in results.items() for movie_id, prediction in rows if movie_id in ratings]
    frame = pd.DataFrame(records)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
    print("Отзывы сохранены:", path)
    for name, part in frame.groupby("method"):
        print(f"{name}: {len(part)} оценок, MAE={abs(part['rating']-part['prediction']).mean():.3f}, "
              f"понравились {(part['rating'] >= 4).mean():.1%}")
    print("Показатели рассчитаны только по оценённым рекомендациям. Пропуски не стали отрицательными отзывами.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=Path("data/work3_models.npz"))
    parser.add_argument("--feedback", type=Path, default=Path("data/work3_feedback.csv"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    try:
        arrays, movies, models = load_models(args.models)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, str(error)+"\n")
    print(f"Фильмов: {len(movies)}. Методы: {', '.join(models)}.")
    if args.demo:
        print("ДЕМО: условный любитель фантастики, не любящий мелодрамы; оценки заданы сценарием.")
        ratings, seen = demo_profile(movies)
        controls = {}
    else:
        ratings, seen = collect_profile(movies)
        if not seen:
            print("История пуста — персональные рекомендации не рассчитаны."); return
        controls = collect_controls(movies, seen)
    print("\nВаш профиль:")
    for movie_id in sorted(seen):
        label = f"{ratings[movie_id]:.1f}" if movie_id in ratings else "взаимодействие без оценки"
        print(f"  {label} — {format_movie(movies, movie_id)}")
    scores, profiles = predict_all(models, movies, ratings, seen)
    check_controls(scores, movies, controls)
    results = print_recommendations(scores, movies, models, profiles, seen, controls)
    print_clusters(arrays, profiles["SVD"], len(ratings))
    if not args.demo: collect_feedback(results, movies, args.feedback)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nВвод завершён.")
