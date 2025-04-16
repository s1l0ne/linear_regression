import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def chart_linear_reg(df: pd.DataFrame, x_names: list[str], y_name: str, folder: str, func) -> None:
    folder_path = f"charts/{folder}"
    os.makedirs(folder_path, exist_ok=True)

    for feature in x_names:
        img = df[[feature, y_name]].plot.scatter(x=feature, y=y_name)

        x_vals = np.linspace(df[feature].min(), df[feature].max(), 500).reshape(-1, 1)
        y_vals = func(x_vals)

        img.plot(x_vals, y_vals, color='red', label='Model prediction')

        img.set_title(f"{feature} vs {y_name}")
        img.set_xlabel(feature)
        img.set_ylabel(y_name)

        plt.savefig(f"{folder_path}/{feature}.png", dpi=300, bbox_inches='tight')

        plt.close()
