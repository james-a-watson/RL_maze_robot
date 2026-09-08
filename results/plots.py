"""Regenerate the result figures from the saved run data.

Every training run stored its per-episode step count (``Ts``) and return
(``Rs``) as JSON under ``saved_run_data/``. This script turns that data into
the figures shown in the README, so the results are reproducible without a
robot or a GPU:

    python results/plots.py

Figures are written next to this file as PNGs.
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "saved_run_data")


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def steps_and_returns(runs, title, outfile, highlight=None):
    """Two panels: steps-to-goal and return per episode, one line per run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, run in runs.items():
        name = label.replace("_", " ")
        alpha = 1.0 if (highlight is None or highlight in label) else 0.25
        ax1.plot(run["Ts"], label=name, alpha=alpha)
        ax2.plot(run["Rs"], label=name, alpha=alpha)
    ax1.set(title="Steps to goal", xlabel="Episode", ylabel="Steps")
    ax2.set(title="Return per episode", xlabel="Episode", ylabel=r"$\Sigma$ reward")
    for ax in (ax1, ax2):
        _tidy(ax)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(HERE, outfile)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def validation_summary(outfile):
    """Success vs. crash over the 100-episode greedy evaluation."""
    v = load("validation_results_100.json")
    goal, wall = v["goal"], v["wall"]
    other = 100 - goal - wall
    fig, ax = plt.subplots(figsize=(7, 2.4))
    left = 0
    for value, colour, label in [(goal, "#2e7d32", "reached goal"),
                                 (wall, "#c62828", "hit wall"),
                                 (other, "#9e9e9e", "timed out")]:
        if value <= 0:
            continue
        ax.barh(0, value, left=left, color=colour, label="%s (%d)" % (label, value))
        left += value
    ax.set(xlim=(0, 100), yticks=[])
    ax.set_title("Greedy-policy evaluation: %d/100 reached the goal" % goal,
                 fontweight="bold", pad=12)
    ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path = os.path.join(HERE, outfile)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def main():
    steps_and_returns(load("hyperparam_tuning.json"),
                      r"Sarsa($\lambda$): learning-rate ($\alpha$) and trace ($\lambda$) sweep, 50 episodes",
                      "sarsa_alpha_lambda.png")
    steps_and_returns(load("hyperparam_tuning_tilings.json"),
                      "Sarsa($\\lambda$): number of tilings, 20 episodes",
                      "sarsa_tilings.png")
    steps_and_returns(load("hyperparam_tuning_250.json"),
                      r"Sarsa($\lambda$): extended run to 250 episodes",
                      "sarsa_extended.png",
                      highlight="0.3, λ=0.3")
    steps_and_returns(load("actor_critic.json"),
                      "Actor-Critic: feature-width (nF) sweep, 20 episodes",
                      "actor_critic.png")
    validation_summary("validation.png")


if __name__ == "__main__":
    main()
