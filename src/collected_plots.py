from protocol_from_cluster_state import (
    run_encoded,
    run_modularized,
    run_alternative_modularized,
    run_simulated_direct,
    run_encoded_simulated_distance,
    run_direct,
    run_encoded_simulated_fully_corrected,
    run_concatenated,
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import noisy_graph_states as nsf
import noisy_graph_states.libs.graph as gt
import noisy_graph_states.libs.matrix as mat
import os

# color-blind friendly color schmees from https://sronpersonalpages.nl/~pault/
HIGH_CONTRAST_COLORS = (
    "#004488",
    "#DDAA33",
    "#BB5566",
    "#000000",
)  # blue, yellow, red, black
MEDIUM_CONTRAST_COLORS = (
    "#6699CC",
    "#004488",
    "#EECC66",
    "#994455",
    "#997700",
    "#EE99AA",
    "#000000",
)  # light_blue dark_blue light_yellow dark_red dark_yellow light_red black
BRIGHT_COLORS = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
)  # blue red green yellow cyan purple grey black


def fidelity(target_ket, rho):
    return mat.H(target_ket) @ rho @ target_ket


def plot_encoded_transport_poster(path_prefix="."):
    # first plot for the poster, shows encoded vs. non-encoded transport
    # for 5-qubit code embedding variant 1
    distances = np.arange(6, 128 + 2, 2, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    ps = [0.99, 0.995, 0.997, 0.999]

    for p, color in zip(ps, colors):
        y3 = [run_modularized(dist, p) for dist in distances]
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y3,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1-p)*100:.1f}\\%",
        )
        plt.plot(distances, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_distance_poster.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    epsilons = np.logspace(-2, -4, num=40)
    ps = 1 - epsilons
    distances = [16, 32, 64]
    for dist, color in zip(distances, colors):
        y2 = [run_modularized(dist, p) for p in ps]
        y4 = [run_simulated_direct(dist, p) for p in ps]
        plt.plot(epsilons, y2, ls="solid", color=color, label=f"{dist=}")
        plt.plot(epsilons, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xscale("log")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_error_poster.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_chained_poster(output_path="encoded_chained_poster.pdf"):
    # second plot for the poster, shows splitting distance in different number of segments
    # in an encoded way vs. non-encoded transport
    # for 5-qubit code embedding variant 2 (variant 2 is necessary for this)
    distances = np.arange(24, 200, 6, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    ps = [0.998, 0.999, 0.9995, 0.9999]
    for p, color in zip(ps, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for dist in distances]
        y2 = []
        y3 = []
        for dist in distances:
            map2 = run_alternative_modularized((dist // 2) - 5, p, return_map=True)
            state2 = nsf.State(gt.bipartite_graph, [map2] * 2)
            rho2 = nsf.noisy_bp_dm(state2, [0, 1])
            fid2 = np.real_if_close(fidelity(gt.bell_pair_ket, rho2))[0, 0]
            y2.append(fid2)
            map3 = run_alternative_modularized((dist // 3) - 5, p, return_map=True)
            state3 = nsf.State(gt.bipartite_graph, [map3] * 3)
            rho3 = nsf.noisy_bp_dm(state3, [0, 1])
            fid3 = np.real_if_close(fidelity(gt.bell_pair_ket, rho3))[0, 0]
            y3.append(fid3)
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y1,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1-p)*100:.2f}\\%",
        )
        plt.plot(distances, y2, ls="dashed", color=color)
        plt.plot(distances, y3, ls="dashdot", color=color)
        plt.plot(distances, y4, ls="dotted", color=color)
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(output_path, bbox_inches="tight")
    # plt.show()
    plt.cla()


def cheese_crab(output_path="cheese_crab.pdf"):
    # plot the full protocol visualisation up to the final decoding measurements
    run_encoded(6, 0.99, save_strategy_plot_path=output_path)


def plot_encoded_transport_variant_1(path_prefix="."):
    distances = np.arange(6, 128 + 2, 2, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    ps = [0.99, 0.995, 0.997, 0.999]

    for p, color in zip(ps, colors):
        y3 = [run_modularized(dist, p) for dist in distances]
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y3,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1 - p) * 100:.1f}\\%",
        )
        plt.plot(distances, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_distance_variant1.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    epsilons = np.logspace(-2, -4, num=40)
    ps = 1 - epsilons
    distances = [16, 32, 64]
    for dist, color in zip(distances, colors):
        y2 = [run_modularized(dist, p) for p in ps]
        y4 = [run_simulated_direct(dist, p) for p in ps]
        plt.plot(epsilons, y2, ls="solid", color=color, label=f"{dist=}")
        plt.plot(epsilons, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xscale("log")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_error_variant1.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_encoded_transport_variant_2(path_prefix="."):
    distances = np.arange(6, 128 + 2, 2, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    ps = [0.99, 0.995, 0.997, 0.999]

    for p, color in zip(ps, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for dist in distances]
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y1,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1 - p) * 100:.1f}\\%",
        )
        plt.plot(distances, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_distance_variant2.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    epsilons = np.logspace(-2, -4, num=40)
    ps = 1 - epsilons
    distances = [16, 32, 64]
    for dist, color in zip(distances, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for p in ps]
        y4 = [run_simulated_direct(dist, p) for p in ps]
        plt.plot(epsilons, y1, ls="solid", color=color, label=f"{dist=}")
        plt.plot(epsilons, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xscale("log")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_error_variant2.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_encoded_transport_both_variants(path_prefix="."):
    distances = np.arange(6, 128 + 1, 1, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    ps = [0.99, 0.995, 0.997, 0.999]

    for p, color in zip(ps, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for dist in distances]
        y3 = [run_modularized(dist, p) for dist in distances]
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y3,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1 - p) * 100:.1f}\\%",
        )
        plt.plot(distances, y1, ls="dashed", color=color)
        plt.plot(distances, y4, ls="dotted", color=color)
        # if p == 0.997:
        #     print(distances)
        #     print(np.array(y3)-np.array(y4))

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_distance_both_variants.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    epsilons = np.logspace(-2, -4, num=40)
    ps = 1 - epsilons
    distances = [16, 32, 64, 128]
    for dist, color in zip(distances, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for p in ps]
        y2 = [run_modularized(dist, p) for p in ps]
        y4 = [run_simulated_direct(dist, p) for p in ps]
        plt.plot(epsilons, y2, ls="solid", color=color, label=f"{dist=}")
        plt.plot(epsilons, y1, ls="dashed", color=color)
        plt.plot(epsilons, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.ylim(0.6, 1.025)
    plt.xscale("log")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.savefig(
        os.path.join(path_prefix, "encoded_vs_direct_by_error_both_variants.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_uncorrectable(output_path="uncorrectable.pdf"):  # do we need this?
    # we have identified which initial errors lead to uncorrectable errors
    # we verify this by looking at very small error rates and see that our prediction
    # of the uncorrectable errors matches what we get from the full protocol
    # distances = [6, 8, 10, 12, 14, 16]
    distances = [8, 32, 128]
    epsilons = np.logspace(-1.7, -6.1, num=80)

    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    for dist, color in zip(distances, colors):
        res_simulated = [
            run_encoded_simulated_distance(dist, 1 - epsilon) for epsilon in epsilons
        ]
        res_direct = [run_simulated_direct(dist, 1 - epsilon) for epsilon in epsilons]

        plt.plot(
            epsilons,
            [1 - x for x in res_simulated],
            c=color,
            label=f"{dist=}",
        )
        plt.plot(epsilons, [1 - x for x in res_direct], ls="dotted", c=color)
    res_corrected = [
        run_encoded_simulated_fully_corrected(1 - epsilon) for epsilon in epsilons
    ]
    plt.plot(
        epsilons,
        [1 - x for x in res_corrected],
        ls="dashed",
        c="gray",
        label="uncorrectable",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(8e-7, 1.5e-2)
    plt.grid()
    plt.legend()
    plt.xlabel(r"error per qubit $\epsilon$")
    plt.ylabel(r"Bell pair infidelity $1-F$")
    plt.savefig(output_path, bbox_inches="tight")
    # plt.show()
    plt.cla()


def plot_chained(path_prefix="."):
    distances = np.arange(36, 350, 6, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    ps = [0.998, 0.999, 0.9995, 0.9999]
    for p, color in zip(ps, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for dist in distances]
        y2 = []
        y3 = []
        for dist in distances:
            map2 = run_alternative_modularized((dist // 2) - 5, p, return_map=True)
            state2 = nsf.State(gt.bipartite_graph, [map2] * 2)
            rho2 = nsf.noisy_bp_dm(state2, [0, 1])
            fid2 = np.real_if_close(fidelity(gt.bell_pair_ket, rho2))[0, 0]
            y2.append(fid2)
            map3 = run_alternative_modularized((dist // 3) - 5, p, return_map=True)
            state3 = nsf.State(gt.bipartite_graph, [map3] * 3)
            rho3 = nsf.noisy_bp_dm(state3, [0, 1])
            fid3 = np.real_if_close(fidelity(gt.bell_pair_ket, rho3))[0, 0]
            y3.append(fid3)
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y1,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1 - p) * 100:.2f}\\%",
        )
        plt.plot(distances, y2, ls="dashed", color=color)
        plt.plot(distances, y3, ls="dashdot", color=color)
        plt.plot(distances, y4, ls="dotted", color=color)
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_chained_by_distance.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    epsilons = np.logspace(-2.6, -4.3, num=80)
    ps = 1 - epsilons
    distances = [36, 72, 144, 288]
    for dist, color in zip(distances, colors):
        y1 = [run_alternative_modularized(dist - 5, p) for p in ps]
        y2 = []
        y3 = []
        for p in ps:
            map2 = run_alternative_modularized((dist // 2) - 5, p, return_map=True)
            state2 = nsf.State(gt.bipartite_graph, [map2] * 2)
            rho2 = nsf.noisy_bp_dm(state2, [0, 1])
            fid2 = np.real_if_close(fidelity(gt.bell_pair_ket, rho2))[0, 0]
            y2.append(fid2)
            map3 = run_alternative_modularized((dist // 3) - 5, p, return_map=True)
            state3 = nsf.State(gt.bipartite_graph, [map3] * 3)
            rho3 = nsf.noisy_bp_dm(state3, [0, 1])
            fid3 = np.real_if_close(fidelity(gt.bell_pair_ket, rho3))[0, 0]
            y3.append(fid3)
        y4 = [run_simulated_direct(dist, p) for p in ps]
        plt.plot(
            epsilons,
            y1,
            ls="solid",
            color=color,
            label=f"{dist=}",
        )
        plt.plot(epsilons, y2, ls="dashed", color=color)
        plt.plot(epsilons, y3, ls="dashdot", color=color)
        plt.plot(epsilons, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.ylim(0.58, 1.02)
    plt.xscale("log")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.xlim(9e-5, 2e-3)
    plt.savefig(
        os.path.join(path_prefix, "encoded_chained_by_error.pdf"), bbox_inches="tight"
    )
    # plt.show()
    plt.cla()


def plot_concatenated(path_prefix="."):
    colors = MEDIUM_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    diagonal_distance = 24
    epsilons = np.logspace(0, -3.2, num=80)
    ps = 1 - epsilons
    concatenation_levels = [0, 1, 2, 3, 4, 5, 6]
    for cl, color in zip(concatenation_levels, colors):
        fids = [
            run_concatenated(
                diagonal_distance, noise_parameter=p, concatenation_levels=cl
            )
            for p in ps
        ]
        plt.plot(epsilons, fids, color=color, label=cl)
    plt.plot(
        epsilons,
        [run_simulated_direct(diagonal_distance, p) for p in ps],
        ls="dotted",
        color="gray",
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.xlim(8e-4, 1)
    plt.xscale("log")
    plt.savefig(
        os.path.join(path_prefix, "encoded_concatenated_by_error.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    distances = np.arange(12, 200, 6, dtype=int)
    p = 0.995
    concatenation_levels = [0, 1, 2, 3, 4, 5, 6]
    for cl, color in zip(concatenation_levels, colors):
        fids = [
            run_concatenated(
                diagonal_distance, noise_parameter=p, concatenation_levels=cl
            )
            for diagonal_distance in distances
        ]
        plt.plot(distances, fids, color=color, label=cl)
    plt.plot(
        distances,
        [run_simulated_direct(diagonal_distance, p) for diagonal_distance in distances],
        ls="dotted",
        color="gray",
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_concatenated_by_distance.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_concatenated_chained(path_prefix="."):
    # here we have four possible variables: error parameter, distance, number of segments, concatenation levels
    colors = MEDIUM_CONTRAST_COLORS
    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    diagonal_distance = 36 * 6
    epsilons = np.logspace(0, -3.2, num=80)
    ps = 1 - epsilons
    concatenation_levels = [0, 1, 2, 3, 4, 5, 6]
    for cl, color in zip(concatenation_levels, colors):
        print(cl)
        y1 = [
            run_concatenated(
                diagonal_distance, noise_parameter=p, concatenation_levels=cl
            )
            for p in ps
        ]
        y2 = []
        y3 = []
        for p in ps:
            map2 = run_concatenated(
                diagonal_distance // 2, p, concatenation_levels=cl, return_map=True
            )
            state2 = nsf.State(gt.bipartite_graph, [map2] * 2)
            rho2 = nsf.noisy_bp_dm(state2, [0, 1])
            fid2 = np.real_if_close(fidelity(gt.bell_pair_ket, rho2))[0, 0]
            y2.append(fid2)
            map3 = run_concatenated(
                diagonal_distance // 3, p, concatenation_levels=cl, return_map=True
            )
            state3 = nsf.State(gt.bipartite_graph, [map3] * 3)
            rho3 = nsf.noisy_bp_dm(state3, [0, 1])
            fid3 = np.real_if_close(fidelity(gt.bell_pair_ket, rho3))[0, 0]
            y3.append(fid3)
        plt.plot(epsilons, y1, color=color, label=cl)
        plt.plot(epsilons, y2, color=color, ls="dashed")
        plt.plot(epsilons, y3, color=color, ls="dashdot")
    plt.plot(
        epsilons,
        [run_simulated_direct(diagonal_distance, p) for p in ps],
        ls="dotted",
        color="gray",
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.xlim(8e-4, 1)
    plt.xscale("log")
    plt.savefig(
        os.path.join(path_prefix, "encoded_chained_concatenated_by_error.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    distances = np.arange(36, 300, 6, dtype=int)
    p = 0.995
    concatenation_levels = [0, 1, 2, 3, 4, 5, 6]
    for cl, color in zip(concatenation_levels, colors):
        print(cl)
        y1 = [
            run_concatenated(
                diagonal_distance, noise_parameter=p, concatenation_levels=cl
            )
            for diagonal_distance in distances
        ]
        y2 = []
        y3 = []
        for diagonal_distance in distances:
            map2 = run_concatenated(
                diagonal_distance // 2, p, concatenation_levels=cl, return_map=True
            )
            state2 = nsf.State(gt.bipartite_graph, [map2] * 2)
            rho2 = nsf.noisy_bp_dm(state2, [0, 1])
            fid2 = np.real_if_close(fidelity(gt.bell_pair_ket, rho2))[0, 0]
            y2.append(fid2)
            map3 = run_concatenated(
                diagonal_distance // 3, p, concatenation_levels=cl, return_map=True
            )
            state3 = nsf.State(gt.bipartite_graph, [map3] * 3)
            rho3 = nsf.noisy_bp_dm(state3, [0, 1])
            fid3 = np.real_if_close(fidelity(gt.bell_pair_ket, rho3))[0, 0]
            y3.append(fid3)
        plt.plot(distances, y1, color=color, label=cl)
        plt.plot(distances, y2, color=color, ls="dashed")
        plt.plot(distances, y3, color=color, ls="dashdot")
    plt.plot(
        distances,
        [run_simulated_direct(diagonal_distance, p) for diagonal_distance in distances],
        ls="dotted",
        color="gray",
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "encoded_chained_concatenated_by_distance.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_reassigning_error_syndromes(path_prefix="."):
    distances = np.arange(6, 128 + 2 + 4, 1, dtype=int)
    colors = HIGH_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
    ps = [0.99, 0.995, 0.997, 0.999]

    for p, color in zip(ps, colors):
        y1 = [run_modularized(dist, p) for dist in distances]
        y2 = [
            run_modularized(dist, p, correction_strategy="standard")
            for dist in distances
        ]
        y4 = [run_simulated_direct(dist, p) for dist in distances]
        plt.plot(
            distances,
            y1,
            ls="solid",
            color=color,
            label=f"$\\varepsilon$={(1 - p) * 100:.1f}\\%",
        )
        plt.plot(distances, y2, ls="dashed", color=color)
        plt.plot(distances, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.xlim(0, 128 + 2)
    plt.savefig(
        os.path.join(
            path_prefix, "reassigning_error_syndromes_by_distance_variant1.pdf"
        ),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()

    epsilons = np.logspace(-2, -4, num=40)
    ps = 1 - epsilons
    distances = [16, 32, 64]
    for dist, color in zip(distances, colors):
        y1 = [run_modularized(dist, p) for p in ps]
        y2 = [run_modularized(dist, p, correction_strategy="standard") for p in ps]
        y4 = [run_simulated_direct(dist, p) for p in ps]
        plt.plot(epsilons, y1, ls="solid", color=color, label=f"{dist=}")
        plt.plot(epsilons, y2, ls="dashed", color=color)
        plt.plot(epsilons, y4, ls="dotted", color=color)

    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xscale("log")
    plt.xlabel(r"error parameter $\varepsilon$")
    plt.savefig(
        os.path.join(path_prefix, "reassigning_error_syndromes_by_error_variant1.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


def plot_concatenation_strategies(path_prefix="."):
    colors = MEDIUM_CONTRAST_COLORS

    mpl.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    # diagonal_distance = 24
    # epsilons = np.logspace(0, -3.2, num=80)
    # ps = 1 - epsilons
    # concatenation_levels = [0, 1, 2, 3, 4, 5, 6]
    # for cl, color in zip(concatenation_levels, colors):
    #     fids = [
    #         run_concatenated(
    #             diagonal_distance, noise_parameter=p, concatenation_levels=cl
    #         )
    #         for p in ps
    #     ]
    #     plt.plot(epsilons, fids, color=color, label=cl)
    # plt.plot(
    #     epsilons,
    #     [run_simulated_direct(diagonal_distance, p) for p in ps],
    #     ls="dotted",
    #     color="gray",
    # )
    # plt.grid()
    # plt.legend()
    # plt.ylabel("Fidelity")
    # plt.xlabel(r"error parameter $\varepsilon$")
    # plt.xlim(8e-4, 1)
    # plt.xscale("log")
    # plt.savefig(
    #     os.path.join(path_prefix, "encoded_concatenated_by_error.pdf"),
    #     bbox_inches="tight",
    # )
    # # plt.show()
    # plt.cla()

    distances = np.arange(12, 200, 6, dtype=int)
    p = 0.995
    concatenation_levels = [0, 1, 2, 3, 4, 5, 6]
    for cl, color in zip(concatenation_levels, colors):
        y1 = [
            run_concatenated(
                diagonal_distance, noise_parameter=p, concatenation_levels=cl
            )
            for diagonal_distance in distances
        ]
        plt.plot(distances, y1, color=color, label=cl)

        y2 = [
            run_concatenated(
                diagonal_distance,
                noise_parameter=p,
                concatenation_levels=cl,
                correction_strategy_outer="standard",
            )
            for diagonal_distance in distances
        ]
        plt.plot(distances, y2, ls="dashed", color=color)

        y3 = [
            run_concatenated(
                diagonal_distance,
                noise_parameter=p,
                concatenation_levels=cl,
                correction_strategy_inner="standard",
            )
            for diagonal_distance in distances
        ]
        plt.plot(distances, y3, ls="dashdot", color=color)

        y4 = [
            run_concatenated(
                diagonal_distance,
                noise_parameter=p,
                concatenation_levels=cl,
                correction_strategy_inner="standard",
                correction_strategy_outer="standard",
            )
            for diagonal_distance in distances
        ]
        plt.plot(distances, y4, ls=(0, (5, 10)), color=color)

    plt.plot(
        distances,
        [run_simulated_direct(diagonal_distance, p) for diagonal_distance in distances],
        ls="dotted",
        color="gray",
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Fidelity")
    plt.xlabel("diagonal distance")
    plt.savefig(
        os.path.join(path_prefix, "concatenated_strategy_comparison.pdf"),
        bbox_inches="tight",
    )
    # plt.show()
    plt.cla()


if __name__ == "__main__":
    plots_directory = "plots"
    os.makedirs(plots_directory, exist_ok=True)

    ## pick which ones to plot
    # plot_encoded_transport_poster(path_prefix=plots_directory)
    # plot_chained_poster(os.path.join(plots_directory, "encoded_chained_poster.pdf"))

    # cheese_crab(os.path.join(plots_directory, "cheese_crab.pdf"))
    # plot_encoded_transport_variant_1(plots_directory)
    # plot_encoded_transport_variant_2(plots_directory)
    plot_encoded_transport_both_variants(plots_directory)
    # plot_chained(plots_directory)
    # plot_uncorrectable(os.path.join(plots_directory, "uncorrectable.pdf"))
    plot_concatenated(path_prefix=plots_directory)
    plot_concatenated_chained(path_prefix=plots_directory)
    # plot_reassigning_error_syndromes(plots_directory)
    # plot_concatenation_strategies(plots_directory)

    pass
