"""
Bar plot of final normalized performance for each method in a folder of .pkl trajectories,
sorted by performance and colored.

Usage:
    python plot_folder.py /path/to/folder          
"""

import os
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt


def load_all(folder):
    data = {}
    mins = None
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.pkl'):
            continue
        label = fn[:-12]
        with open(os.path.join(folder, fn), 'rb') as f:
            trajs = pickle.load(f)
        outputs = np.array([env['outputs'] for env in trajs])
        if mins is None:
            mins = np.array([env['minimum'] for env in trajs])
        data[label] = outputs

    return data, mins


def normalize_data(data, mins, baseline_label=None, init_steps=10):
    if baseline_label is None:
        baseline_label = next(iter(data))
    base = data[baseline_label]
    n_runs = base.shape[0]

    medians = np.median(base[:, :init_steps], axis=1)
    # mins = np.full(n_runs, np.inf)
    for arr in data.values():
        mins = np.minimum(mins, np.nanmin(arr, axis=1))
    denom = medians - mins
    zero = denom == 0
    denom[zero] = 1.0
    norm = {}
    for label, arr in data.items():
        filled_arr = np.where(np.isnan(arr), np.inf, arr)
        best = np.minimum.accumulate(filled_arr, axis=1)
        n = (medians[:, None] - best) / denom[:, None]
        
        n[zero, :] = 1.0
        norm[label] = n
    return norm


def plot_final_bar_sorted(normed, save_path=None):
    labels = list(normed.keys())
    finals = np.array([normed[l][:, -1] for l in labels])  
    means = finals.mean(axis=1)
    order = np.argsort(-means)
    sorted_labels = [labels[i] for i in order]
    sorted_means = means[order]

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(sorted_labels)))

    x = np.arange(len(sorted_labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, sorted_means, capsize=5, color=colors)

    bot = np.min(sorted_means)
    top = np.max(sorted_means)
    pad = (top - bot) * 0.1
    plt.ylim(bot - pad, top + pad)
    plt.xticks(x, sorted_labels, rotation=45, ha='right')
    plt.ylabel('Mean Normalised Performance')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    print(f"Saved bar plot to {save_path}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder', help="Folder with .pkl trajectory files")
    args = p.parse_args()

    data, mins = load_all(args.folder)
    if not data:
        raise RuntimeError(f"No .pkl files found in {args.folder!r}")

    normed = normalize_data(data, mins)
    plot_final_bar_sorted(normed, f'{args.folder}/barplot.png')


if __name__ == '__main__':
    main()
