import logging
import logging as log
import os
import sys
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
        'features': ['GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr',
                     'TotRmsAbvGrd', 'GarageArea'],
        'target': 'SalePrice'
    }
}


def get_config(dataset: DataSet) -> dict:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Конфигурация для {dataset} не найдена.")
    return DATASET_CONFIG[dataset]


def setup_logging(filename: str = None):
    os.makedirs('logs', exist_ok=True)

    path = f'logs/{filename}'

    handlers = [log.StreamHandler(sys.stderr)]
    if filename:
        handlers.append(logging.FileHandler(path, mode='w'))

    logging.basicConfig(
        level=log.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        handlers=handlers
    )
