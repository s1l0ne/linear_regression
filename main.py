import pandas as pd


class DataSet:
    house_prices = 0


DATA_SET = DataSet.house_prices

PATHS = (
    ('house_prices/train.csv', 'house_prices/test.csv', 'house_prices/sample_submission.csv'),
         )


def get_data() -> map:
    return map(pd.read_csv, map(lambda x: 'data/' + x, PATHS[DATA_SET]))


if __name__ == '__main__':
    train_data, test_data, answers_data = get_data()

    print(train_data.head())

    features = ['GrLivArea', 'GarageCars', 'FullBath']
    target = 'SalePrice'

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    print(train_data.head())
    print(test_data.head())
