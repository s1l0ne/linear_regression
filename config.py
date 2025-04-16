from enum import IntEnum


class DataSet(IntEnum):
    house_prices = 0


DATASET_CONFIG = {
    DataSet.house_prices: {
        'paths': (
            'house_prices/train.csv',
            'house_prices/test.csv',
            'house_prices/sample_submission.csv'
        ),
        'features': ['GrLivArea'],
        'target': 'SalePrice'
    }
}
# , 'GarageCars', 'FullBath'


def get_config(dataset: DataSet) -> dict:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Конфигурация для {dataset} не найдена.")
    return DATASET_CONFIG[dataset]
