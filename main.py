import pandas as pd
from enum import IntEnum


class DataSet(IntEnum):
    house_prices = 0


DATA_SET = DataSet.house_prices

PATHS = (
    ('house_prices/train.csv', 'house_prices/test.csv', 'house_prices/sample_submission.csv'),
         )


def get_data() -> tuple:
    return tuple(pd.read_csv(f'data/{path}') for path in PATHS[DATA_SET])


if __name__ == '__main__':
    train_data, test_data, answers_data = get_data()

    print(train_data.head())

    features = ['GrLivArea', 'GarageCars', 'FullBath']
    target = 'SalePrice'

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    print(train_data.head())
    print(test_data.head())
