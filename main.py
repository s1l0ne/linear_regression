from config import get_config, DataSet, setup_logging
from utils.charts import make_charts
from utils.data import get_data, preprocess_data
from models.linear_regression import linear_regression
import logging as log
from datetime import datetime


setup_logging(datetime.now().strftime("%d-%m-%Y_%H%M%S.log"))
config = get_config(DataSet.house_prices)


if __name__ == '__main__':
    log.info('Запуск')

    log.info('Считывание данных')
    train_data, test_data, answers_data = get_data(config)

    features = config['features']
    target = config['target']

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]
    log.info('Данные считаны')

    log.info('Создание чартов before')
    make_charts(train_data, features, target, 'before')
    log.info('Чарты before созданы')

    log.info('Начало обработки данных')
    train_data = preprocess_data(train_data)
    log.info('Данные обработаны')

    log.info('Создание чартов after')
    make_charts(train_data, features, target, 'after')
    log.info('Чарты after созданы')

    log.info('Начало обучения модели')
    X = train_data[features].to_numpy()
    y = train_data[target].to_numpy()

    predict = linear_regression(X, y)
    log.info('Модель обучена')

    log.info('Заверщение работы программы')