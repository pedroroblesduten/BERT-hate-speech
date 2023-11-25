import torch
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm
import time

def plot_losses(train_loss, val_loss, path, model_name):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train loss", color="tab:blue", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation loss", color="tab:orange", linewidth=2)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(
        f"{model_name}: training and validation loss", fontsize=16, fontweight="bold"
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    # Grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Spines (border) color
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    # Tick width
    ax.tick_params(width=0.5)

    # Background color
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()

    save_path = os.path.join(path, model_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
