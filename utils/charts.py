import os
import pandas as pd
import matplotlib.pyplot as plt


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