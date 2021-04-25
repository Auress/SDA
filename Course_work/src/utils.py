import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted


def show_feature_importances(feature_names: list,
                             feature_importances: list,
                             get_top: int = None):
    """
    Отображение важности признаков.

    Parameters
    ----------
    feature_names: list
        Список признаков.

    feature_importances: list
        Список значений важности признаков.

    get_top: int
        Сколько из топовых признаков вернуть.

    Returns
    -------
    result: list
        Список топовых признокав.
    """

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    plt.figure(figsize=(20, len(feature_importances) * 0.355))

    sns.barplot(feature_importances['importance'], feature_importances['feature'])

    plt.xlabel('Importance')
    plt.title('Importance of features')
    plt.show()

    if get_top is not None:
        return feature_importances['feature'][:get_top].tolist()


def do_train_test_split(data: pd.DataFrame,
                        target: list,
                        features: list,
                        random_state: int  = 42,
                        train_size: float = 0.5,  # разделение train и valid/test
                        valid_size: float = 0.5):  # разделение valid и test

    """
    Разделение на train, valid и test датасеты.

    Parameters
    ----------
    data: pd.DataFrame
        Матрица признаков.

    target: list
        Значения целевого признака.

    features: list
        Список признаков для новых датасетов.

    random_state: int
        random state.

    train_size: float
        Размер части train при разделении на train и valid/test.

    valid_size: float
        Размер части valid при разделении на valid и test.

    Returns
    -------
    x_train, x_valid, x_test, y_train, y_valid, y_test: pd.DataFrame
        Матрицы признаков после разделения.
    """

    x_train, x_valid = train_test_split(
        data.drop(target, axis=1), train_size=train_size, shuffle=True, random_state=random_state
    )
    y_train, y_valid = train_test_split(
        data[target], train_size=train_size, shuffle=True, random_state=random_state
    )

    x_train = x_train[features]
    x_valid = x_valid[features]

    x_valid, x_test = train_test_split(
        x_valid, train_size=valid_size, shuffle=True, random_state=random_state
    )
    y_valid, y_test = train_test_split(
        y_valid, train_size=valid_size, shuffle=True, random_state=random_state
    )

    print("x_train.shape = {} rows, {} cols".format(*x_train.shape))
    print("x_valid.shape = {} rows, {} cols".format(*x_valid.shape))
    print("x_test.shape = {} rows, {} cols".format(*x_test.shape))

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def make_cross_validation(X: pd.DataFrame,
                          y: pd.Series,
                          estimator: object,
                          params: dict,
                          metric: callable,
                          cv_strategy):
    """
    Кросс-валидация.

    Parameters
    ----------
    X: pd.DataFrame
        Матрица признаков.

    y: pd.Series
        Вектор целевой переменной.

    estimator: callable
        Объект модели для обучения.

    paprams: dict
        Параметры модели

    metric: callable
        Метрика для оценки качества решения.
        Ожидается, что на вход будет передана функция,
        которая принимает 2 аргумента: y_true, y_pred.

    cv_strategy: cross-validation generator
        Объект для описания стратегии кросс-валидации.
        Ожидается, что на вход будет передан объект типа
        KFold или StratifiedKFold.

    Returns
    -------
    oof_score: float
        Значение метрики качества на OOF-прогнозах.

    fold_train_scores: List[float]
        Значение метрики качества на каждом обучающем датасете кросс-валидации.

    fold_valid_scores: List[float]
        Значение метрики качества на каждом валидационном датасете кросс-валидации.

    oof_predictions: np.array
        Прогнозы на OOF.

    """
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])
    X = X.reindex()
    y = y.reindex()

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = copy.deepcopy(estimator)

        try:
            model.fit(x_train, y_train,
                      eval_set=(x_valid, y_valid),
                      **params)
        except ValueError: # XGBoost хочет такой синтаксис
            model.fit(x_train, y_train,
                      eval_set=[(x_train, y_train), (x_valid, y_valid)],
                      **params)

        y_train_pred = model.predict_proba(x_train)[:, 1]
        y_valid_pred = model.predict_proba(x_valid)[:, 1]

        fold_train_scores.append(metric(y_train, y_train_pred))
        fold_valid_scores.append(metric(y_valid, y_valid_pred))
        oof_predictions[valid_idx] = y_valid_pred

        msg = (
            f"Fold: {fold_number + 1}, train-observations = {len(train_idx)}, "
            f"valid-observations = {len(valid_idx)}\n"
            f"train-score = {round(fold_train_scores[fold_number], 4)}, "
            f"valid-score = {round(fold_valid_scores[fold_number], 4)}"
        )
        print(msg)
        print("=" * 69)
        estimators.append(model)

    oof_score = metric(y, oof_predictions)
    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions


def create_numerical_aggs(data: pd.DataFrame,
                          groupby_id: str,
                          aggs: dict,
                          prefix: Optional[str] = None,
                          suffix: Optional[str] = None,
                          ) -> pd.DataFrame:
    """
    Построение агрегаций для числовых признаков.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Выборка для построения агрегаций.

    groupby_id: str
        Название ключа, по которому нужно произвести группировку.

    aggs: dict
        Словарь с названием признака и списка функций.
        Ключ словаря - название признака, который используется для
        вычисления агрегаций, значение словаря - список с названием
        функций для вычисления агрегаций.

    prefix: str, optional, default = None
        Префикс для названия признаков.
        Опциональный параметр, по умолчанию, не используется.

    suffix: str, optional, default = None
        Суффикс для названия признаков.
        Опциональный параметр, по умолчанию, не используется.

    Returns
    -------
    stats: pandas.core.frame.DataFrame
        Выборка с рассчитанными агрегациями.

    """
    if not prefix:
        prefix = ""
    if not suffix:
        suffix = ""

    data_grouped = data.groupby(groupby_id)
    stats = data_grouped.agg(aggs)
    stats.columns = [f"{prefix}{feature}_{stat}{suffix}".upper() for feature, stat in stats]
    stats = stats.reset_index()

    return stats


class MeanTargetEncoding(BaseEstimator, TransformerMixin):
    """
       Таргет энкодинг категориальных признаков.
    """

    def __init__(self, alpha: float = 0, folds: int = 5, random_state = 2177):
        self.folds = folds
        self.alpha = alpha
        self.random_state = random_state
        self.features = None
        self.cv = None

    def fit(self, X, y):
        self.features = {}
        self.cv = KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        global_mean = np.mean(y)

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        for fold_number, (train_idx, valid_idx) in enumerate(self.cv.split(X, y), start=1):
            x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

            data = pd.DataFrame({'feature': x_train, 'target': y_train})
            data = data.groupby(['feature'])['target'].agg([np.mean, np.size])
            data = data.reset_index()
            score = data['mean'] * data['size'] + global_mean * self.alpha
            score = score / (data['size'] + self.alpha)

            self.features[f'fold_{fold_number}'] = {key: value for key, value in zip(data['feature'], score)}

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, 'features')

        X.reset_index(drop=True, inplace=True)

        x_transformed = X.copy(deep=True)

        for fold_number, (train_idx, valid_idx) in enumerate(self.cv.split(X, y), start=1):
            x_transformed.loc[valid_idx] = x_transformed.loc[valid_idx].map(self.features[f'fold_{fold_number}'])

        return x_transformed.astype('float')

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        x_transformed = X.copy(deep=True)

        for fold_number, (train_idx, valid_idx) in enumerate(self.cv.split(X, y), start=1):
            x_transformed.loc[valid_idx] = x_transformed.loc[valid_idx].map(self.features[f'fold_{fold_number}'])

        return x_transformed.astype('float')
