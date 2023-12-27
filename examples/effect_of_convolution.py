"""
Simple example
"""
# third party
import matplotlib.pyplot as plt

# local
from lmc.utils import finite_difference_matrix, laplacian_kernel_matrix

# from plotting import plot_profiles_and_observations
from utils import format_axis, make_gif, set_fig_size


def plot_regularisation(
    reg_profile, fig=None, axis=None, axis_label=None, path_to_fig=None
):
    if fig is None and axis is None:
        fig, axis = plt.subplots(1, 1, figsize=set_fig_size(435, fraction=0.9))

    axis.plot(reg_profile, label=axis_label, marker="o", linestyle="-")

    format_axis(
        axis,
        fig,
        arrowed_spines=True,
        ylim=(-1.05, 1.05),
        xlim=(0, len(reg_profile)),
        xlabel="Time points",
        ylabel="Penalization",
    )

    if axis_label is not None:
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.25, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
        )

    fig.tight_layout()

    if path_to_fig is not None:
        fig.savefig(path_to_fig, transparent=True, bbox_inches="tight")

    return fig, axis


def plot_regularisation_series(reg_profiles, axis_label=None, path_to_dir=None):
    for n, reg_profile in enumerate(reg_profiles):
        fig, axis = plt.subplots(1, 1, figsize=set_fig_size(435, fraction=0.9))
        axis.plot(reg_profile, label=axis_label, marker="o", linestyle="-")

        format_axis(
            axis,
            fig,
            arrowed_spines=True,
            ylim=(-1.05, 1.05),
            xlim=(0, len(reg_profile)),
            xlabel="Time points",
            ylabel="Penalization",
        )

        fig.tight_layout()

        if path_to_dir is not None:
            fig.savefig(
                f"{path_to_dir}/frame{n}.jpg", transparent=True, bbox_inches="tight"
            )

    return fig, axis


def main():
    n_timepoints = 10
    K = finite_difference_matrix(n_timepoints)
    D = laplacian_kernel_matrix(n_timepoints)
    R = K @ D
    # fig, axis = plot_regularisation(K[4], axis_label="LMC")
    # plot_regularisation(
    #    R[4],
    #    fig=fig,
    #    axis=axis,
    #    axis_label="CMC",
    #    path_to_fig="./figures/effect_conv.pdf",
    # )

    n_timepoints = 20
    K = finite_difference_matrix(n_timepoints)
    D = laplacian_kernel_matrix(n_timepoints)
    R = K @ D

    plot_regularisation_series(
        K,
        path_to_dir="./figures/reg",
    )
    make_gif("./figures/reg/reg.gif", "./figures/reg")

    plot_regularisation_series(R, path_to_dir="./figures/convreg")
    make_gif("./figures/convreg/convreg.gif", "./figures/convreg")


if __name__ == "__main__":
    main()
