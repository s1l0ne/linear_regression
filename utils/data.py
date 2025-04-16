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