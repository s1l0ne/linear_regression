from config import get_config, DataSet
from utils.charts import make_charts, chart_linear_reg
from utils.data import get_data, preprocess_data
from models.linear_regression import linear_regression


config = get_config(DataSet.house_prices)


if __name__ == '__main__':
    train_data, test_data, answers_data = get_data(config)

    features = config['features']
    target = config['target']

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    make_charts(train_data, features, target, 'before')

    train_data = preprocess_data(train_data)

    make_charts(train_data, features, target, 'after')

    X = train_data[features].to_numpy()
    y = train_data[target].to_numpy()

    predict = linear_regression(X, y)

    chart_linear_reg(train_data, features, target, 'regression', predict)
