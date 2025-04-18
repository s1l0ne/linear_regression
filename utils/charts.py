import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_charts(df: pd.DataFrame, x_names: list[str], y_name: str, folder: str) -> None:
    folder_path = f"charts/{folder}"
    os.makedirs(folder_path, exist_ok=True)

    for feature in x_names:
        img = df[[feature, y_name]].plot.scatter(x=feature, y=y_name, s=1)

        img.set_title(f"{feature} vs {y_name}")
        img.set_xlabel(feature)
        img.set_ylabel(y_name)

        plt.savefig(f"{folder_path}/{feature}.png", dpi=300, bbox_inches='tight')

        plt.close()


def make_charts_compare(df1: pd.DataFrame, df2: pd.DataFrame, x_names: list[str], y_name: str, folder: str) -> None:
    folder_path = f"charts/{folder}"
    os.makedirs(folder_path, exist_ok=True)

    for feature in x_names:
        img = df1[[feature, y_name]].plot.scatter(x=feature, y=y_name, s=1, c='blue')
        df2[[feature, y_name]].plot.scatter(x=feature, y=y_name, s=1, c='red', label='df2', ax=img)

        img.set_title(f"{feature} vs {y_name}")
        img.set_xlabel(feature)
        img.set_ylabel(y_name)

        plt.savefig(f"{folder_path}/{feature}.png", dpi=300, bbox_inches='tight')

        plt.close()