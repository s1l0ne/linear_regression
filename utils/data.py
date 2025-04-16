import pandas as pd


def get_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, test_path, answers_path = config['paths']

    return (
        pd.read_csv(f'data/{train_path}'),
        pd.read_csv(f'data/{test_path}'),
        pd.read_csv(f'data/{answers_path}')
    )