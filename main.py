from config import get_config, DataSet, setup_logging
from utils.charts import make_charts
from utils.data import get_data, preprocess_data
from models.linear_regression import linear_regression
import logging as log
from datetime import datetime


setup_logging(datetime.now().strftime("%d-%m-%Y_%H%M%S.log"))
config = get_config(DataSet.house_prices)


if __name__ == '__main__':
    log.info('Инициализация: запуск пайплайна обработки данных')

    log.info('Этап 1: чтение данных из CSV')
    train_data, test_data, answers_data = get_data(config)

    features = config['features']
    target = config['target']

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]
    log.info(f'Данные успешно загружены: {train_data.shape[0]} записей в train')

    log.info('Этап 2: построение визуализаций (до предобработки)')
    make_charts(train_data, features, target, 'before')
    log.info('Графики успешно сохранены в директории charts/before')

    log.info('Этап 3: предобработка данных')
    train_data = preprocess_data(train_data)
    log.info('Предобработка завершена')

    log.info('Этап 4: построение визуализаций (после предобработки)')
    make_charts(train_data, features, target, 'after')
    log.info('Графики успешно сохранены в директории charts/after')

    log.info('Этап 5: инициализация и обучение модели линейной регрессии (linear regression)')
    X = train_data[features].to_numpy()
    y = train_data[target].to_numpy()

    predict = linear_regression(X, y)
    log.info('Обучение завершено успешно')

    log.info('Заверщение работы: пайплайн выполнен без критических ошибок')