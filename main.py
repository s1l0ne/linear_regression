import logging as log
from datetime import datetime

import pandas as pd

from config import DataSet, get_config, setup_logging
from models.linear_regression import linear_regression
from utils.charts import make_charts, make_charts_compare
from utils.data import get_data, preprocess_data, train_test_split
from utils.evaluate_model import evaluate_model

setup_logging(datetime.now().strftime("%d-%m-%Y_%H%M%S.log"))
config = get_config(DataSet.house_prices)


if __name__ == '__main__':
    log.info('Инициализация: запуск пайплайна обработки данных')

    log.info('Этап 1: чтение данных из CSV')
    train_data, test_data, answers_data = get_data(config)

    features = config['features']
    target = config['target']

    train_data = train_data[features + [target]]
    test_data = pd.merge(test_data[['Id'] + features], answers_data[['Id', target]], on='Id')
    data = pd.concat([train_data, test_data], ignore_index=True)
    train_data, test_data = train_test_split(data.dropna())
    log.info(f'Данные успешно загружены: {data.shape[0]} записей')

    log.info('Этап 2: построение визуализаций (до предобработки)')
    make_charts(train_data, features, target, 'before')
    make_charts_compare(train_data, test_data, features, target, 'before/compare')
    log.info('Графики успешно сохранены в директории charts/before')

    log.info('Этап 3: предобработка данных')
    train_data = preprocess_data(train_data)
    test_data = test_data.dropna()
    log.info('Предобработка завершена')

    log.info('Этап 4: построение визуализаций (после предобработки)')
    make_charts(train_data, features, target, 'after')
    make_charts_compare(train_data, test_data, features, target, 'after/compare')
    log.info('Графики успешно сохранены в директории charts/after')

    log.info('Этап 5: инициализация и обучение модели линейной регрессии (linear regression)')
    X = train_data[features].to_numpy()
    y = train_data[target].to_numpy()

    predict = linear_regression(X, y)
    log.info('Обучение завершено успешно')

    log.info('Этап 6: расчёт метрик MAE, MSE, RMSE и R2-score')
    mae, mse, rmse, r2 = evaluate_model(X, y, predict)
    log.info(f'Метрики успешно расчитаны: MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}, R2-score = {r2:.2f}')
    log.info('Нормализованные метрики: MAE')

    log.info('Этап 7: расчёт метрик MAE, MSE, RMSE и R2-score на незнакомых данных')
    X_test = test_data[features].to_numpy()
    y_test = test_data[target].to_numpy()

    mae, mse, rmse, r2 = evaluate_model(X_test, y_test, predict)
    log.info(f'Метрики для незнакомых данных успешно расчитаны: MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f},'
             f' R2-score = {r2:.2f}')

    log.info('Завершение работы: пайплайн выполнен без критических ошибок')
