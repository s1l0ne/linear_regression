import numpy as np
import pandas as pd


def get_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, test_path, answers_path = config['paths']

    return (
        pd.read_csv(f'data/{train_path}'),
        pd.read_csv(f'data/{test_path}'),
        pd.read_csv(f'data/{answers_path}')
    )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()

    for column in df.select_dtypes(include=['number']).columns:
        mean = df[column].mean()
        sigma = df[column].std()

        df = df[(df[column] >= mean - sigma * 3) & (df[column] <= mean + sigma * 3)]

    return df


def train_test_split(df, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(df))
    split = int(len(df) * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()