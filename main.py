import pandas as pd
import matplotlib.pyplot as plt
import os
from config import get_config, DataSet


config = get_config(DataSet.house_prices)


def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, test_path, answers_path = config['paths']

    return (
        pd.read_csv(f'data/{train_path}'),
        pd.read_csv(f'data/{test_path}'),
        pd.read_csv(f'data/{answers_path}')
    )


def make_charts(df: pd.DataFrame, x_names: list[str], y_name: str, folder: str) -> None:
    folder_path = f"charts/{folder}"
    os.makedirs(folder_path, exist_ok=True)

    for feature in x_names:
        img = df[[feature, y_name]].plot.scatter(x=feature, y=y_name)

        img.set_title(f"{feature} vs {y_name}")
        img.set_xlabel(feature)
        img.set_ylabel(y_name)

        plt.savefig(f"{folder_path}/{feature}.png", dpi=300, bbox_inches='tight')

        plt.close()


if __name__ == '__main__':
    train_data, test_data, answers_data = get_data()

    features = config['features']
    target = config['target']

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    make_charts(train_data, features, target, 'before')
