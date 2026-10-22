"""
Общий модуль для всех практических работ по РОПСППР.

Предметная область: онлайн-кинотеатр.
Данные: MovieLens 32M (GroupLens, 2024), из которого формируется
компактный «свежий» срез: оценки последних лет по фильмам последних лет.

Модуль отвечает за:
  * чтение исходного архива ml-32m.zip (без распаковки на диск);
  * разбор описаний фильмов (год выпуска, список жанров, one-hot жанров);
  * формирование среза по заданным порогам;
  * временное разбиение оценок на обучающую и тестовую части;
  * общие метрики качества (MAE, RMSE, Precision@K, Recall@K, NDCG@K);
  * сохранение и загрузку подготовленного среза.

Все практические работы импортируют функции отсюда, чтобы данные,
разбиение и метрики были одинаковыми и результаты можно было сравнивать.
"""

import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

# Пути по умолчанию (относительно папки с ноутбуками)
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-32m.zip"
SLICE_DIR = DATA_DIR / "ml_slice"

# Папка внутри архива
ZIP_FOLDER = "ml-32m"

# Жанры MovieLens (в том порядке, в котором они описаны в README датасета)
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Порог «понравившегося» фильма для метрик топ-N рекомендаций
RELEVANCE_THRESHOLD = 4.0

# Жанровые группы - укрупнённые жанры (состояния Марковской цепи в работе №2).
# Порядок словаря задаёт приоритет: фильм относится к первой группе, в которую
# попадает хотя бы один из его жанров. Сначала идут более «узкие» группы,
# в конце - самая общая (драма), она же - группа по умолчанию.
GENRE_GROUPS = {
    "Семейное":   ["Animation", "Children"],
    "Фантастика": ["Sci-Fi", "Fantasy"],
    "Боевик":     ["Action", "Adventure", "War", "Western"],
    "Триллер":    ["Thriller", "Horror", "Crime", "Mystery", "Film-Noir"],
    "Комедия":    ["Comedy", "Musical"],
    "Драма":      ["Drama", "Romance", "Documentary"],
}
GROUP_NAMES = list(GENRE_GROUPS)


# ---------------------------------------------------------------------------
# Чтение исходного архива
# ---------------------------------------------------------------------------

def _to_timestamp(date):
    """Переводит дату (строку или datetime) в unix-время в секундах."""
    if date is None:
        return None
    return int(pd.Timestamp(date).timestamp())


def read_ratings(zip_path=ZIP_PATH, start_date=None, end_date=None,
                 chunksize=2_000_000):
    """
    Читает ratings.csv из архива частями и оставляет только оценки
    из интервала [start_date, end_date).

    Файл содержит 32 млн строк, поэтому он читается кусками по chunksize
    строк: в память попадают только нужные оценки.
    """
    start_ts = _to_timestamp(start_date)
    end_ts = _to_timestamp(end_date)
    dtypes = {"userId": "int32", "movieId": "int32",
              "rating": "float32", "timestamp": "int64"}

    parts = []
    with ZipFile(zip_path) as zf:
        with zf.open(f"{ZIP_FOLDER}/ratings.csv") as f:
            for chunk in pd.read_csv(f, dtype=dtypes, chunksize=chunksize):
                mask = np.ones(len(chunk), dtype=bool)
                if start_ts is not None:
                    mask &= chunk["timestamp"].to_numpy() >= start_ts
                if end_ts is not None:
                    mask &= chunk["timestamp"].to_numpy() < end_ts
                if mask.any():
                    parts.append(chunk[mask])

    ratings = pd.concat(parts, ignore_index=True)
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return ratings


def read_movies(zip_path=ZIP_PATH):
    """Читает movies.csv и сразу разбирает его (см. parse_movies)."""
    with ZipFile(zip_path) as zf:
        with zf.open(f"{ZIP_FOLDER}/movies.csv") as f:
            movies = pd.read_csv(f)
    return parse_movies(movies)


def read_links(zip_path=ZIP_PATH):
    """Читает links.csv: соответствие movieId -> imdbId, tmdbId."""
    with ZipFile(zip_path) as zf:
        with zf.open(f"{ZIP_FOLDER}/links.csv") as f:
            links = pd.read_csv(f, dtype={"imdbId": "string"})
    links["tmdbId"] = links["tmdbId"].astype("Int64")
    return links


def read_tags(zip_path=ZIP_PATH, start_date=None, end_date=None):
    """Читает tags.csv (пользовательские теги) с фильтром по времени."""
    with ZipFile(zip_path) as zf:
        with zf.open(f"{ZIP_FOLDER}/tags.csv") as f:
            tags = pd.read_csv(f)
    if start_date is not None:
        tags = tags[tags["timestamp"] >= _to_timestamp(start_date)]
    if end_date is not None:
        tags = tags[tags["timestamp"] < _to_timestamp(end_date)]
    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
    return tags.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Разбор описаний фильмов
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def parse_movies(movies):
    """
    Приводит movies.csv к удобному виду:
      * year       - год выпуска, извлечённый из названия «Title (1999)»;
      * clean_title - название без года;
      * genre_list - список жанров вместо строки «Action|Comedy»;
      * по одной 0/1-колонке на каждый жанр из GENRES (one-hot).
    """
    movies = movies.copy()
    years = movies["title"].str.extract(_YEAR_RE)[0]
    movies["year"] = pd.to_numeric(years, errors="coerce").astype("Int64")
    movies["clean_title"] = movies["title"].str.replace(_YEAR_RE, "", regex=True).str.strip()

    def split_genres(value):
        if not isinstance(value, str) or value == "(no genres listed)":
            return []
        return value.split("|")

    movies["genre_list"] = movies["genres"].apply(split_genres)
    for genre in GENRES:
        movies[genre] = movies["genre_list"].apply(lambda lst, g=genre: int(g in lst))
    return movies


# ---------------------------------------------------------------------------
# Формирование среза
# ---------------------------------------------------------------------------

def build_slice(ratings, movies, min_movie_year=2010, n_movies=3000,
                n_users=2000, min_user_ratings=20, min_movie_ratings=10,
                random_state=42, verbose=True):
    """
    Формирует компактный срез данных.

    Шаги:
      1. Оставляем фильмы, вышедшие не раньше min_movie_year.
      2. Из них берём n_movies самых оцениваемых за выбранный период:
         редкие фильмы почти не имеют общих оценок с другими, и
         коллаборативные методы на них ничего не дают.
      3. Пользователей, поставивших этим фильмам не меньше
         min_user_ratings оценок, отбираем СЛУЧАЙНО (а не самых
         активных), чтобы срез был похож на реальную аудиторию,
         а не только на «супер-зрителей» с тысячами оценок.
      4. Итеративно удаляем фильмы с числом оценок < min_movie_ratings
         и пользователей с числом оценок < min_user_ratings, пока
         оба условия не будут выполнены одновременно (так называемое
         k-core разрежение): удаление фильмов уменьшает число оценок
         у пользователей и наоборот, поэтому одного прохода мало.
    """
    log = print if verbose else (lambda *a, **k: None)

    # 1. Свежие фильмы
    fresh_ids = movies.loc[movies["year"] >= min_movie_year, "movieId"]
    data = ratings[ratings["movieId"].isin(fresh_ids)]
    log(f"Оценок свежих фильмов (год >= {min_movie_year}): {len(data):,}")

    # 2. Самые оцениваемые фильмы
    movie_counts = data["movieId"].value_counts()
    top_movies = movie_counts.head(n_movies).index
    data = data[data["movieId"].isin(top_movies)]
    log(f"После выбора {len(top_movies)} популярных фильмов: {len(data):,} оценок")

    # 3. Случайная выборка достаточно активных пользователей
    user_counts = data["userId"].value_counts()
    eligible = user_counts[user_counts >= min_user_ratings].index
    rng = np.random.default_rng(random_state)
    if len(eligible) > n_users:
        chosen = rng.choice(np.sort(eligible.to_numpy()), size=n_users, replace=False)
    else:
        chosen = eligible.to_numpy()
    data = data[data["userId"].isin(chosen)]
    log(f"Подходящих пользователей: {len(eligible):,}, выбрано: {len(chosen):,}")

    # 4. Итеративное k-core разрежение
    iteration = 0
    while True:
        iteration += 1
        m_cnt = data["movieId"].value_counts()
        u_cnt = data["userId"].value_counts()
        bad_movies = m_cnt[m_cnt < min_movie_ratings].index
        bad_users = u_cnt[u_cnt < min_user_ratings].index
        if len(bad_movies) == 0 and len(bad_users) == 0:
            break
        data = data[~data["movieId"].isin(bad_movies) & ~data["userId"].isin(bad_users)]
    log(f"k-core разрежение завершено за {iteration} итераций")

    data = data.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    movies_slice = movies[movies["movieId"].isin(data["movieId"].unique())].reset_index(drop=True)
    return data, movies_slice


def describe_slice(ratings):
    """Короткая сводка по срезу: размеры, плотность, период."""
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    density = len(ratings) / (n_users * n_movies)
    print(f"Пользователей: {n_users:,}")
    print(f"Фильмов:       {n_movies:,}")
    print(f"Оценок:        {len(ratings):,}")
    print(f"Плотность матрицы: {density:.2%}")
    print(f"Оценок на пользователя: медиана {ratings.groupby('userId').size().median():.0f}")
    print(f"Оценок на фильм:        медиана {ratings.groupby('movieId').size().median():.0f}")
    print(f"Период: {ratings['datetime'].min():%Y-%m-%d} — {ratings['datetime'].max():%Y-%m-%d}")


# ---------------------------------------------------------------------------
# Жанровые группы
# ---------------------------------------------------------------------------

def genre_group(genre_list):
    """
    Номер жанровой группы фильма (индекс в GROUP_NAMES).
    Фильм с несколькими жанрами относится к первой по приоритету группе;
    фильм без подходящих жанров (например, только IMAX) - к последней.
    """
    genres = set(genre_list)
    for index, members in enumerate(GENRE_GROUPS.values()):
        if genres.intersection(members):
            return index
    return len(GROUP_NAMES) - 1


# ---------------------------------------------------------------------------
# Сессии оценивания
# ---------------------------------------------------------------------------

def assign_sessions(ratings, gap_seconds=3600):
    """
    Делит оценки каждого пользователя на сессии: новая сессия начинается,
    если с предыдущей оценки прошло больше gap_seconds секунд.

    В MovieLens пользователи обычно оценивают фильмы пачками: за один
    заход на сайт - десятки оценок с интервалом в секунды. Поэтому
    единицей «поведения во времени» разумнее считать сессию, а не
    отдельную оценку. Добавляет колонку session (номер сессии
    пользователя, начиная с 0).
    """
    data = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    gap = data.groupby("userId")["timestamp"].diff()
    is_new = gap.isna() | (gap > gap_seconds)
    data["session"] = is_new.astype(int).groupby(data["userId"]).cumsum() - 1
    return data


# ---------------------------------------------------------------------------
# Временное разбиение
# ---------------------------------------------------------------------------

def temporal_split(ratings, test_ratio=0.2):
    """
    Для каждого пользователя последние test_ratio его оценок по времени
    уходят в тест, остальные - в обучение.

    Так моделируется реальная работа рекомендательной системы:
    модель знает только прошлое пользователя и должна угадать,
    что он оценит дальше. Тестовые оценки играют роль реакции
    пользователя на рекомендации.

    При одинаковом времени оценки упорядочиваются по movieId,
    чтобы разбиение было воспроизводимым.
    """
    data = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    sizes = data.groupby("userId")["movieId"].transform("size")
    position = data.groupby("userId").cumcount()
    n_test = np.maximum(1, np.round(sizes * test_ratio)).astype(int)
    is_test = position >= (sizes - n_test)
    train = data[~is_test].reset_index(drop=True)
    test = data[is_test].reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# Матрица оценок
# ---------------------------------------------------------------------------

def rating_matrix(ratings, users=None, movies=None):
    """
    Строит матрицу «пользователь x фильм» (DataFrame), где отсутствующие
    оценки - NaN. Через users / movies можно зафиксировать порядок строк
    и столбцов (например, взять все фильмы среза, а не только из train).
    """
    matrix = ratings.pivot_table(index="userId", columns="movieId",
                                 values="rating", aggfunc="last")
    if users is not None:
        matrix = matrix.reindex(index=users)
    if movies is not None:
        matrix = matrix.reindex(columns=movies)
    return matrix


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    """Средняя абсолютная ошибка прогноза оценки."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Корень из средней квадратичной ошибки прогноза оценки."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ranking_metrics(recommendations, test, k=10, threshold=RELEVANCE_THRESHOLD):
    """
    Метрики качества топ-N рекомендаций.

    recommendations - словарь {userId: [movieId, ...]} в порядке убывания
                      уверенности системы;
    test            - тестовые оценки (userId, movieId, rating).

    Релевантными считаются тестовые фильмы с оценкой >= threshold
    (по умолчанию 4.0 из 5 - «понравился»).

      Precision@K - доля релевантных среди K рекомендованных;
      Recall@K    - доля найденных релевантных среди всех релевантных
                    фильмов пользователя в тесте;
      NDCG@K      - учитывает позицию: релевантный фильм на 1-м месте
                    ценнее, чем на 10-м.

    Усредняется по пользователям, у которых в тесте есть хотя бы один
    релевантный фильм.
    """
    relevant = (test[test["rating"] >= threshold]
                .groupby("userId")["movieId"].apply(set).to_dict())
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    precisions, recalls, ndcgs = [], [], []
    for user, rel in relevant.items():
        recs = list(recommendations.get(user, []))[:k]
        hits = np.array([movie in rel for movie in recs], dtype=float)
        precisions.append(hits.sum() / k)
        recalls.append(hits.sum() / len(rel))
        dcg = float((hits * discounts[:len(hits)]).sum())
        idcg = float(discounts[:min(len(rel), k)].sum())
        ndcgs.append(dcg / idcg)

    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "users": len(relevant),
    }


# ---------------------------------------------------------------------------
# Сохранение и загрузка подготовленного среза
# ---------------------------------------------------------------------------

def save_slice(slice_dir=SLICE_DIR, **frames):
    """
    Сохраняет таблицы среза в CSV (например, train=..., test=..., movies=...).
    Списки жанров записываются строкой через «|».
    """
    slice_dir = Path(slice_dir)
    slice_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame = frame.copy()
        if "genre_list" in frame.columns:
            frame["genre_list"] = frame["genre_list"].apply("|".join)
        if "datetime" in frame.columns:
            frame = frame.drop(columns="datetime")
        frame.to_csv(slice_dir / f"{name}.csv", index=False)
        print(f"Сохранено: {slice_dir / f'{name}.csv'} ({len(frame):,} строк)")


def load_slice(slice_dir=SLICE_DIR):
    """
    Загружает подготовленный срез. Возвращает словарь таблиц:
    train, test, movies, links, tags (если файлы есть).
    """
    slice_dir = Path(slice_dir)
    result = {}
    for path in sorted(slice_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if "genre_list" in frame.columns:
            frame["genre_list"] = frame["genre_list"].fillna("").apply(
                lambda s: s.split("|") if s else [])
        if "year" in frame.columns:
            frame["year"] = frame["year"].astype("Int64")
        if "tmdbId" in frame.columns:
            frame["tmdbId"] = frame["tmdbId"].astype("Int64")
        if "timestamp" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="s")
        result[path.stem] = frame
    return result
