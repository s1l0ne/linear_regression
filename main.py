from config import get_config, DataSet
from utils.charts import make_charts
from utils.data import get_data

config = get_config(DataSet.house_prices)


if __name__ == '__main__':
    train_data, test_data, answers_data = get_data(config)

    features = config['features']
    target = config['target']

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    make_charts(train_data, features, target, 'before')
