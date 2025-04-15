import pandas as pd
from enum import IntEnum


class DataSet(IntEnum):
    house_prices = 0


DATA_SET = DataSet.house_prices

PATHS = {
    DataSet.house_prices: ('house_prices/train.csv', 'house_prices/test.csv', 'house_prices/sample_submission.csv'),
        }


def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, test_path, answers_path = PATHS[DATA_SET]

    return (
        pd.read_csv(f'data/{train_path}'),
        pd.read_csv(f'data/{test_path}'),
        pd.read_csv(f'data/{answers_path}')
    )


if __name__ == '__main__':
    train_data, test_data, answers_data = get_data()

    print(train_data.head())

    features = ['GrLivArea', 'GarageCars', 'FullBath']
    target = 'SalePrice'

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    print(train_data.head())
    print(test_data.head())
