import matplotlib.pyplot as plt
import numpy as np
from utils import format_axis, set_fig_size

STATE_COLORS = {1: "lightgreen", 2: "lightblue", 3: "salmon", 4: "maroon"}


def plot_basic_profiles(V, path_to_fig=None):
    fig, axis = plt.subplots(1, 1, figsize=set_fig_size(435, fraction=0.9))
    axis.plot(V)

    if path_to_fig is not None:
        fig.savefig(path_to_fig, transparent=True, bbox_inches="tight")

    return fig, axis


def plot_coefficients(U, path_to_fig=None):
    fig, axis = plt.subplots(1, 1, figsize=set_fig_size(435, fraction=0.9))
    axis.hist(U.ravel(), bins="auto")

    if path_to_fig is not None:
        fig.savefig(path_to_fig, transparent=True, bbox_inches="tight")

    return fig, axis


def plot_profile(m, axis, label=None):
    axis.plot(m, label=label, color="orange")

    return axis


def plot_observations(x, axis, missing=0, label=None):
    # unique_vaules = np.unique(x[x != missing])
    # for value in unique_vaules:

    #    x_masked = x.copy()
    #    x_masked[x != value] = missing
    #    x_masked[x_masked == missing] = float(np.nan)

    #    axis.plot(x_masked, marker="o", linestyle="", label=label,
    #              color=STATE_COLORS[value], markersize=4)

    x_masked = x.copy()
    x_masked[x_masked == missing] = float(np.nan)
    axis.plot(
        x_masked,
        marker="o",
        linestyle="",
        label=label,
        color="k",
        markersize=3,
        alpha=0.7,
    )  # , markeredgecolor="white")

    return axis


def plot_profiles_and_observations(X, M, path_to_fig=None, n_states=4):
    fig, axes = plt.subplots(2, 2, figsize=set_fig_size(435, fraction=0.9))

    for n, axis in enumerate(axes.ravel()):
        axis = plot_profile(M[n], axis, label="Latent profile")
        axis = plot_observations(X[n], axis, label="Measurements")

        format_axis(
            axis,
            fig,
            arrowed_spines=True,
            ylim=(-0.05, n_states + 0.5),
            xlim=(0, X.shape[1]),
            xlabel="Time",
        )

    # fig.legend()
    fig.tight_layout()

    if path_to_fig is not None:
        fig.savefig(path_to_fig, transparent=True, bbox_inches="tight")

    return fig, axis
