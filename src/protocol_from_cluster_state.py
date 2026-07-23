from collections import defaultdict

import networkx as nx
import noisy_graph_states as nsf
import noisy_graph_states.libs.graph as gt
import noisy_graph_states.libs.matrix as mat
from noisy_graph_states.tools.patterns import pattern_to_sequence
import numpy as np
from matplotlib import pyplot as plt

import graph
import visualize
from visualize import draw_on_grid

from two_d_cluster_tools import (
    DIRECTION_UP,
    DIRECTION_RIGHT,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    get_idx_by_direction,
    get_idx_by_directions,
    idx_sequence_by_directions,
)
from five_qubit_cluster_ring_correction import (
    get_correction_rules_cluster_ring_xz_optimized,
    get_correction_rules_cluster_ring,
)

NUM_CODE_QUBITS = 5


def fidelity(target_ket, rho):
    return mat.H(target_ket) @ rho @ target_ket


def mirror_directions(directions):
    new_directions = []
    for direction in directions:
        if direction == DIRECTION_UP:
            new_directions.append(DIRECTION_DOWN)
        elif direction == DIRECTION_RIGHT:
            new_directions.append(DIRECTION_LEFT)
        elif direction == DIRECTION_DOWN:
            new_directions.append(DIRECTION_UP)
        elif direction == DIRECTION_LEFT:
            new_directions.append(DIRECTION_RIGHT)
        else:
            raise ValueError(f"{direction=} is not recognised.")
    return new_directions


def perform_correction(
    state,
    correction_mapping_dict,
    start_idx,
    target_idx,
    correction_strategy="xz_optimized",
):
    # state is expected to be the state after the first code qubit has been measured

    syndrome_indices = list(correction_mapping_dict.values())
    reduced_maps = nsf.reduce_maps(state, [start_idx, target_idx] + syndrome_indices)
    compiled_maps = nsf.compile_maps(*reduced_maps)
    # print(compiled_maps)
    # print(1 - np.sum(compiled_maps.weights))

    if correction_strategy == "xz_optimized":
        correction_rules = get_correction_rules_cluster_ring_xz_optimized(
            idx_dict=correction_mapping_dict, input_idx=start_idx, output_idx=target_idx
        )
    elif correction_strategy == "standard":
        correction_rules = get_correction_rules_cluster_ring(
            idx_dict=correction_mapping_dict, input_idx=start_idx, output_idx=target_idx
        )

    updated_weights = defaultdict(float)
    for weight, noise in zip(compiled_maps.weights, compiled_maps.noises):
        new_noise = list(noise)
        filtered_noise = list(
            filter(lambda noise_part: noise_part in syndrome_indices, noise)
        )
        for combination, to_flip in correction_rules.items():
            if all(
                [noise_part in combination for noise_part in filtered_noise]
            ) and all([combination_part in noise for combination_part in combination]):
                # print("aaaaah")
                new_noise = list(nsf.add_or_remove(to_flip, tuple(new_noise)))
        for idx in syndrome_indices:
            try:
                new_noise.remove(idx)
            except ValueError:
                pass
        new_noise = tuple(new_noise)
        # print(weight, noise, "mapped to", new_noise)
        updated_weights[new_noise] += weight

    weights = []
    noises = []
    for noise, weight in updated_weights.items():
        noises.append(noise)
        weights.append(weight)
    new_map = nsf.Map(weights, noises)
    new_state = nsf.State(state.graph, maps=[new_map])
    for idx in syndrome_indices:
        new_state = nsf.x_measurement(new_state, idx)
    return new_state


def get_code_directions(mirrored=False):
    if mirrored:
        UP = DIRECTION_LEFT
        DOWN = DIRECTION_RIGHT
        RIGHT = DIRECTION_DOWN
        LEFT = DIRECTION_UP
    else:
        UP = DIRECTION_UP
        DOWN = DIRECTION_DOWN
        RIGHT = DIRECTION_RIGHT
        LEFT = DIRECTION_LEFT

    setup_ys = [[UP], [DOWN, LEFT], [DOWN, RIGHT]]
    setup_zs = [
        [UP, UP],
        [DOWN, LEFT, LEFT],
        [DOWN, LEFT, DOWN],
        [DOWN, RIGHT, RIGHT],
        [DOWN, RIGHT, DOWN],
    ]

    path1_start = [LEFT]
    path1 = [LEFT, LEFT, UP, UP, UP, UP, RIGHT, UP, RIGHT]
    path2_start = [LEFT, UP]
    path2 = [UP, UP, RIGHT, UP, RIGHT]
    path3_start = [RIGHT, UP]
    path3 = [UP, RIGHT, UP]
    path4_start = [RIGHT]
    path4 = [RIGHT, RIGHT, UP]
    path5_start = [DOWN]
    path5 = [DOWN, DOWN, RIGHT, RIGHT, RIGHT, UP, RIGHT, UP, RIGHT]
    path_cutouts = [
        # from path1
        [LEFT] * 4,
        [LEFT] * 4 + [UP],
        [LEFT] * 4 + [UP] * 2,
        [LEFT] * 4 + [UP] * 3,
        [LEFT] * 4 + [UP] * 4,
        [DOWN, LEFT, LEFT, LEFT],
        # from path2
        [LEFT, LEFT, UP],
        [LEFT, LEFT, UP, UP],
        [LEFT, LEFT, UP, UP, UP],
        # from path4
        [DOWN, RIGHT, DOWN, RIGHT],
        [DOWN, RIGHT, RIGHT, RIGHT],
        # from path5
        [DOWN, DOWN, DOWN, LEFT],
        [DOWN] * 4,
        [DOWN] * 4 + [RIGHT],
        [DOWN] * 4 + [RIGHT] * 2,
        [DOWN] * 4 + [RIGHT] * 3,
    ]

    return {
        "setup_ys": setup_ys,
        "setup_zs": setup_zs,
        "path1_start": path1_start,
        "path1": path1,
        "path2_start": path2_start,
        "path2": path2,
        "path3_start": path3_start,
        "path3": path3,
        "path4_start": path4_start,
        "path4": path4,
        "path5_start": path5_start,
        "path5": path5,
        "path_cutouts": path_cutouts,
    }


def get_alternative_code_directions(mirrored=False):
    # this one has the input/output qubit outside
    if mirrored:
        UP = DIRECTION_LEFT
        DOWN = DIRECTION_RIGHT
        RIGHT = DIRECTION_DOWN
        LEFT = DIRECTION_UP
    else:
        UP = DIRECTION_UP
        DOWN = DIRECTION_DOWN
        RIGHT = DIRECTION_RIGHT
        LEFT = DIRECTION_LEFT

    setup_ys = [
        [UP, UP, LEFT, LEFT],
        [UP, UP, LEFT, LEFT, LEFT],
        [UP, LEFT, LEFT, LEFT],
        [RIGHT],
        [LEFT, LEFT],
        [LEFT, LEFT, LEFT, DOWN],
        [DOWN, DOWN, LEFT, LEFT],
        [DOWN, DOWN, RIGHT],
        [LEFT, DOWN],
        [LEFT, DOWN, DOWN],
        [LEFT],
        [UP],
        [UP, UP],
    ]
    setup_zs = [
        [DOWN],
        [UP, LEFT],
        [UP, LEFT, LEFT],
        [RIGHT, RIGHT],
        [UP, UP, RIGHT],
        [UP, LEFT, LEFT, LEFT, LEFT],
        [UP, UP, LEFT, LEFT, LEFT, LEFT],
        [UP, UP, UP],
        [UP, UP, UP, LEFT, LEFT],
        [UP, UP, UP, LEFT, LEFT, LEFT],
        [DOWN, DOWN, RIGHT, RIGHT],
        [DOWN, DOWN, DOWN, RIGHT],
        [DOWN, DOWN, DOWN, LEFT],
        [DOWN, DOWN, DOWN, LEFT, LEFT],
        [DOWN, LEFT, LEFT, LEFT, LEFT],
        [DOWN, LEFT, LEFT],
    ]

    path1_start = [LEFT, LEFT, LEFT]
    path1 = [LEFT, LEFT, UP, UP, UP, RIGHT, UP, RIGHT]
    path2_start = [LEFT, UP, UP]
    path2 = [UP, UP, RIGHT]
    path3_start = [RIGHT, UP]
    path3 = [RIGHT, UP]
    path4_start = [DOWN, RIGHT]
    path4 = [RIGHT, RIGHT, UP]
    path5_start = [DOWN, DOWN]
    path5 = [DOWN, DOWN, RIGHT, RIGHT, RIGHT, UP, RIGHT]
    path_cutouts = [
        # from path1
        [LEFT] * 6,
        [LEFT] * 6 + [UP],
        [LEFT] * 6 + [UP] * 2,
        [LEFT] * 6 + [UP] * 3,
        [DOWN] + [LEFT] * 5,
        # from path2
        [UP] * 4 + [LEFT] * 2,
        # from path4
        # not strictly necessary
        [DOWN] * 3 + [RIGHT] * 2,
        [DOWN] * 2 + [RIGHT] * 3,
        # from path5
        [DOWN] * 4 + [LEFT],
        [DOWN] * 5,
        [DOWN] * 5 + [RIGHT],
        [DOWN] * 5 + [RIGHT] * 2,
        [DOWN] * 5 + [RIGHT] * 3,
    ]

    return {
        "setup_ys": setup_ys,
        "setup_zs": setup_zs,
        "path1_start": path1_start,
        "path1": path1,
        "path2_start": path2_start,
        "path2": path2,
        "path3_start": path3_start,
        "path3": path3,
        "path4_start": path4_start,
        "path4": path4,
        "path5_start": path5_start,
        "path5": path5,
        "path_cutouts": path_cutouts,
    }


def reverse_directions(path):
    new_path = []
    for direction in path[::-1]:
        if direction == DIRECTION_UP:
            new_path.append(DIRECTION_DOWN)
        elif direction == DIRECTION_DOWN:
            new_path.append(DIRECTION_UP)
        elif direction == DIRECTION_LEFT:
            new_path.append(DIRECTION_RIGHT)
        elif direction == DIRECTION_RIGHT:
            new_path.append(DIRECTION_LEFT)
    return new_path


def run_encoded(diagonal_distance, noise_parameter, save_strategy_plot_path=None):
    distance = diagonal_distance
    num_rows = distance + 11
    num_cols = distance + 11
    shape = (num_rows, num_cols)
    start_graph = graph.construct_2d_cluster_graph(shape)

    start_idx = 5 * num_cols + 5  # 105
    target_idx = get_idx_by_directions(
        start_idx, [DIRECTION_UP, DIRECTION_RIGHT] * distance, shape
    )
    start_dirs = get_code_directions()
    target_dirs = get_code_directions(mirrored=True)

    seq = []
    # first setup both sides
    seq += [
        ("y", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_zs"]
    ]
    seq += [
        ("y", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_zs"]
    ]
    # cut out all other zs
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["path_cutouts"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["path_cutouts"]
    ]

    # construct the paths
    path1 = (
        start_dirs["path1"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path1"])[:-1]
    )
    # print(target_dirs["path1"])
    path1_start = get_idx_by_directions(start_idx, start_dirs["path1_start"], shape)
    path1_ids = idx_sequence_by_directions(
        start_idx=path1_start, directions=path1, shape=shape
    )

    path2 = (
        start_dirs["path2"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 5)
        + reverse_directions(target_dirs["path2"])[:-1]
    )
    path2_start = get_idx_by_directions(start_idx, start_dirs["path2_start"], shape)
    path2_ids = idx_sequence_by_directions(
        start_idx=path2_start, directions=path2, shape=shape
    )

    path3 = (
        start_dirs["path3"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 5)
        + reverse_directions(target_dirs["path3"])[:-1]
    )
    path3_start = get_idx_by_directions(start_idx, start_dirs["path3_start"], shape)
    path3_ids = idx_sequence_by_directions(
        start_idx=path3_start, directions=path3, shape=shape
    )

    path4 = (
        start_dirs["path4"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 4)
        + reverse_directions(target_dirs["path4"])[:-1]
    )
    path4_start = get_idx_by_directions(start_idx, start_dirs["path4_start"], shape)
    path4_ids = idx_sequence_by_directions(
        start_idx=path4_start, directions=path4, shape=shape
    )

    path5 = (
        start_dirs["path5"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path5"])[:-1]
    )
    path5_start = get_idx_by_directions(start_idx, start_dirs["path5_start"], shape)
    path5_ids = idx_sequence_by_directions(
        start_idx=path5_start, directions=path5, shape=shape
    )

    for path in [path1_ids, path2_ids, path3_ids, path4_ids, path5_ids]:
        seq += [("x", idx, start_idx) for idx in path]

    if save_strategy_plot_path is not None:
        strat = nsf.Strategy(start_graph, seq)
        visualize.visualize_2d_cluster_end_state(
            strat, shape, show=False, save=save_strategy_plot_path
        )
    # from time import time
    # start_time = time()
    # strat.populate_cache()
    # print(f"{shape[0]*shape[1]} Nodes. With {distance=}. Propagating all noises took {time()-start_time:.2f} seconds.")
    # strat.save()

    # do the final measurement
    seq += [
        (
            "x",
            get_idx_by_directions(target_idx, target_dirs["path1_start"], shape),
            start_idx,
        )
    ]

    correction_mapping_dict = {
        2: get_idx_by_directions(target_idx, target_dirs["path2_start"], shape),
        3: get_idx_by_directions(target_idx, target_dirs["path3_start"], shape),
        4: get_idx_by_directions(target_idx, target_dirs["path4_start"], shape),
        5: get_idx_by_directions(target_idx, target_dirs["path5_start"], shape),
    }

    strat = nsf.Strategy(start_graph, seq)
    # if save_strategy_plot_path is not None:
    #     visualize.visualize_2d_cluster_end_state(strat, shape, show=False, save=save_strategy_plot_path)
    # from time import time
    # start_time = time()
    # strat.populate_cache()
    # print(f"{shape[0]*shape[1]} Nodes. With {distance=}. Propagating all noises took {time()-start_time:.2f} seconds.")
    # strat.save()

    p = noise_parameter
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.State(start_graph, maps=[])
    state = nsf.pauli_noise(
        state,
        indices=range(num_rows * num_cols),
        coefficients=coefficients,
    )
    state = strat(state)
    strat.save()

    state = perform_correction(state, correction_mapping_dict, start_idx, target_idx)

    output_rho = nsf.noisy_bp_dm(
        state,
        target_indices=[start_idx, target_idx],
    )
    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    return fid


def run_encoded_simulated_distance(diagonal_distance, noise_parameter):
    distance = 6
    additional_distance = diagonal_distance - distance
    num_rows = distance + 11
    num_cols = distance + 11
    shape = (num_rows, num_cols)
    start_graph = graph.construct_2d_cluster_graph(shape)

    start_idx = 5 * num_cols + 5
    target_idx = get_idx_by_directions(
        start_idx, [DIRECTION_UP, DIRECTION_RIGHT] * distance, shape
    )
    start_dirs = get_code_directions()
    target_dirs = get_code_directions(mirrored=True)

    seq = []
    # first setup both sides
    seq += [
        ("y", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_zs"]
    ]
    seq += [
        ("y", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_zs"]
    ]
    # cut out all other zs
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["path_cutouts"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["path_cutouts"]
    ]

    # construct the paths
    path1 = (
        start_dirs["path1"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path1"])[:-1]
    )
    # print(target_dirs["path1"])
    path1_start = get_idx_by_directions(start_idx, start_dirs["path1_start"], shape)
    path1_ids = idx_sequence_by_directions(
        start_idx=path1_start, directions=path1, shape=shape
    )

    path2 = (
        start_dirs["path2"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 5)
        + reverse_directions(target_dirs["path2"])[:-1]
    )
    path2_start = get_idx_by_directions(start_idx, start_dirs["path2_start"], shape)
    path2_ids = idx_sequence_by_directions(
        start_idx=path2_start, directions=path2, shape=shape
    )

    path3 = (
        start_dirs["path3"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 5)
        + reverse_directions(target_dirs["path3"])[:-1]
    )
    path3_start = get_idx_by_directions(start_idx, start_dirs["path3_start"], shape)
    path3_ids = idx_sequence_by_directions(
        start_idx=path3_start, directions=path3, shape=shape
    )

    path4 = (
        start_dirs["path4"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 4)
        + reverse_directions(target_dirs["path4"])[:-1]
    )
    path4_start = get_idx_by_directions(start_idx, start_dirs["path4_start"], shape)
    path4_ids = idx_sequence_by_directions(
        start_idx=path4_start, directions=path4, shape=shape
    )

    path5 = (
        start_dirs["path5"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path5"])[:-1]
    )
    path5_start = get_idx_by_directions(start_idx, start_dirs["path5_start"], shape)
    path5_ids = idx_sequence_by_directions(
        start_idx=path5_start, directions=path5, shape=shape
    )

    for path in [path1_ids, path2_ids, path3_ids, path4_ids, path5_ids]:
        seq += [("x", idx, start_idx) for idx in path]

    # aux_strat = nsf.Strategy(start_graph, seq)
    # weight_vector = aux_strat.get_weight_vector_expression()
    #
    # interest_set = [90,198,215,214,199,181,180]
    # interest_set = list(sorted(interest_set))
    # new_weight_vector = defaultdict(list)
    #
    # for k, v in weight_vector.items():
    #     new_key = tuple()
    #     for idx in interest_set:
    #         if idx in k:
    #             new_key += (idx,)
    #     if new_key:
    #         new_weight_vector[new_key] += v
    #
    # for k, v in new_weight_vector.items():
    #     print(k, v)
    #     if "x_58" in v:
    #         print("aaaaaah")
    # aux_strat.save()
    # visualize.visualize_2d_cluster_end_state(aux_strat, shape)

    # do the final measurement
    seq += [
        (
            "x",
            get_idx_by_directions(target_idx, target_dirs["path1_start"], shape),
            start_idx,
        )
    ]

    correction_mapping_dict = {
        2: get_idx_by_directions(target_idx, target_dirs["path2_start"], shape),
        3: get_idx_by_directions(target_idx, target_dirs["path3_start"], shape),
        4: get_idx_by_directions(target_idx, target_dirs["path4_start"], shape),
        5: get_idx_by_directions(target_idx, target_dirs["path5_start"], shape),
    }

    strat = nsf.Strategy(start_graph, seq)

    p = noise_parameter
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.State(start_graph, maps=[])
    state = nsf.pauli_noise(
        state, indices=range(num_rows * num_cols), coefficients=coefficients
    )

    # additional noise from simulated distance
    q = p**additional_distance
    additional_coefficients = [q + (1 - q) / 4, (1 - q) / 4, (1 - q) / 4, (1 - q) / 4]
    additional_indices = [
        get_idx_by_directions(path1_start, start_dirs["path1"] + [DIRECTION_UP], shape),
        get_idx_by_directions(
            path1_start, start_dirs["path1"] + [DIRECTION_UP, DIRECTION_RIGHT], shape
        ),
        get_idx_by_directions(path2_start, start_dirs["path2"] + [DIRECTION_UP], shape),
        get_idx_by_directions(
            path2_start, start_dirs["path2"] + [DIRECTION_UP, DIRECTION_RIGHT], shape
        ),
        get_idx_by_directions(
            path3_start, start_dirs["path3"] + [DIRECTION_RIGHT], shape
        ),
        get_idx_by_directions(
            path3_start, start_dirs["path3"] + [DIRECTION_RIGHT, DIRECTION_UP], shape
        ),
        get_idx_by_directions(
            path4_start, start_dirs["path4"] + [DIRECTION_RIGHT], shape
        ),
        get_idx_by_directions(
            path4_start, start_dirs["path4"] + [DIRECTION_RIGHT, DIRECTION_UP], shape
        ),
        get_idx_by_directions(path5_start, start_dirs["path5"] + [DIRECTION_UP], shape),
        get_idx_by_directions(
            path5_start, start_dirs["path5"] + [DIRECTION_UP, DIRECTION_RIGHT], shape
        ),
    ]
    state = nsf.pauli_noise(
        state, indices=additional_indices, coefficients=additional_coefficients
    )

    state = strat(state)
    strat.save()

    state = perform_correction(state, correction_mapping_dict, start_idx, target_idx)

    output_rho = nsf.noisy_bp_dm(
        state,
        target_indices=[start_idx, target_idx],
    )
    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    return fid


# try to find out which noises cannot be suppressed
def run_encoded_simulated_fully_corrected(noise_parameter, noise=None):
    distance = 6
    # additional_distance = diagonal_distance - distance
    num_rows = distance + 11
    num_cols = distance + 11
    shape = (num_rows, num_cols)
    start_graph = graph.construct_2d_cluster_graph(shape)

    start_idx = 5 * num_cols + 5
    target_idx = get_idx_by_directions(
        start_idx, [DIRECTION_UP, DIRECTION_RIGHT] * distance, shape
    )
    start_dirs = get_code_directions()
    target_dirs = get_code_directions(mirrored=True)

    seq = []
    # first setup both sides
    seq += [
        ("y", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_zs"]
    ]
    seq += [
        ("y", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_zs"]
    ]
    # cut out all other zs
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["path_cutouts"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["path_cutouts"]
    ]

    # construct the paths
    path1 = (
        start_dirs["path1"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path1"])[:-1]
    )
    # print(target_dirs["path1"])
    path1_start = get_idx_by_directions(start_idx, start_dirs["path1_start"], shape)
    path1_ids = idx_sequence_by_directions(
        start_idx=path1_start, directions=path1, shape=shape
    )

    path2 = (
        start_dirs["path2"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 5)
        + reverse_directions(target_dirs["path2"])[:-1]
    )
    path2_start = get_idx_by_directions(start_idx, start_dirs["path2_start"], shape)
    path2_ids = idx_sequence_by_directions(
        start_idx=path2_start, directions=path2, shape=shape
    )

    path3 = (
        start_dirs["path3"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 5)
        + reverse_directions(target_dirs["path3"])[:-1]
    )
    path3_start = get_idx_by_directions(start_idx, start_dirs["path3_start"], shape)
    path3_ids = idx_sequence_by_directions(
        start_idx=path3_start, directions=path3, shape=shape
    )

    path4 = (
        start_dirs["path4"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 4)
        + reverse_directions(target_dirs["path4"])[:-1]
    )
    path4_start = get_idx_by_directions(start_idx, start_dirs["path4_start"], shape)
    path4_ids = idx_sequence_by_directions(
        start_idx=path4_start, directions=path4, shape=shape
    )

    path5 = (
        start_dirs["path5"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path5"])[:-1]
    )
    path5_start = get_idx_by_directions(start_idx, start_dirs["path5_start"], shape)
    path5_ids = idx_sequence_by_directions(
        start_idx=path5_start, directions=path5, shape=shape
    )

    for path in [path1_ids, path2_ids, path3_ids, path4_ids, path5_ids]:
        seq += [("x", idx, start_idx) for idx in path]

    # do the final measurement
    seq += [
        (
            "x",
            get_idx_by_directions(target_idx, target_dirs["path1_start"], shape),
            start_idx,
        )
    ]

    correction_mapping_dict = {
        2: get_idx_by_directions(target_idx, target_dirs["path2_start"], shape),
        3: get_idx_by_directions(target_idx, target_dirs["path3_start"], shape),
        4: get_idx_by_directions(target_idx, target_dirs["path4_start"], shape),
        5: get_idx_by_directions(target_idx, target_dirs["path5_start"], shape),
    }

    strat = nsf.Strategy(start_graph, seq)

    p = noise_parameter
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.State(start_graph, maps=[])
    affected_indices = [
        start_idx,
        target_idx,
        # the ys
        get_idx_by_directions(start_idx, [DIRECTION_UP], shape),
        # get_idx_by_directions(start_idx, [DIRECTION_DOWN, DIRECTION_LEFT], shape),
        # get_idx_by_directions(start_idx, [DIRECTION_DOWN, DIRECTION_RIGHT], shape),
        get_idx_by_directions(target_idx, [DIRECTION_LEFT], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_DOWN, DIRECTION_RIGHT], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_UP, DIRECTION_RIGHT], shape),
        # the z-neighbours of the ys
        # get_idx_by_directions(start_idx, [DIRECTION_UP, DIRECTION_UP], shape),
        get_idx_by_directions(
            start_idx, [DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_LEFT], shape
        ),
        get_idx_by_directions(
            start_idx, [DIRECTION_DOWN, DIRECTION_RIGHT, DIRECTION_RIGHT], shape
        ),
        get_idx_by_directions(
            start_idx, [DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_DOWN], shape
        ),
        # get_idx_by_directions(start_idx, [DIRECTION_DOWN, DIRECTION_RIGHT, DIRECTION_DOWN], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_LEFT, DIRECTION_LEFT], shape),
        get_idx_by_directions(
            target_idx, [DIRECTION_RIGHT, DIRECTION_UP, DIRECTION_UP], shape
        ),
        get_idx_by_directions(
            target_idx, [DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_DOWN], shape
        ),
        get_idx_by_directions(
            target_idx, [DIRECTION_RIGHT, DIRECTION_UP, DIRECTION_RIGHT], shape
        ),
        # get_idx_by_directions(target_idx, [DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_RIGHT], shape),
        # the encoding and decoding qubits  TURNS OUT THESE ARE ACTUALLY NOT CONTRIBUTING
        # get_idx_by_directions(start_idx, [DIRECTION_UP, DIRECTION_LEFT], shape),
        # get_idx_by_directions(start_idx, [DIRECTION_UP, DIRECTION_RIGHT], shape),
        # get_idx_by_directions(start_idx, [DIRECTION_LEFT], shape),
        # get_idx_by_directions(start_idx, [DIRECTION_RIGHT], shape),
        # get_idx_by_directions(start_idx, [DIRECTION_DOWN], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_LEFT, DIRECTION_UP], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_LEFT, DIRECTION_DOWN], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_UP], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_DOWN], shape),
        # get_idx_by_directions(target_idx, [DIRECTION_RIGHT], shape),
        # the two special annoying ones that are correlated on an even and odd
        get_idx_by_directions(
            start_idx,
            [DIRECTION_LEFT, DIRECTION_LEFT, DIRECTION_UP, DIRECTION_UP],
            shape,
        ),
        get_idx_by_directions(
            target_idx,
            [DIRECTION_UP, DIRECTION_UP, DIRECTION_LEFT, DIRECTION_LEFT],
            shape,
        ),
    ]
    # affected_indices = []
    state = nsf.pauli_noise(state, indices=affected_indices, coefficients=coefficients)
    # if noise is not None:
    #     noise_idx = noise["idx"]
    #     if noise["type"] == "x":
    #         state = nsf.x_noise(state, [noise_idx], epsilon=0.1)
    #     elif noise["type"] == "y":
    #         state = nsf.y_noise(state, [noise_idx], epsilon=0.1)
    #     elif noise["type"] == "z":
    #         state = nsf.z_noise(state, [noise_idx], epsilon=0.1)

    state = strat(state)

    # print("aaaah")
    # weight_vector = strat.get_weight_vector_expression()
    # strat.save()
    # relevant_nodes = [90, 214, 215, 199, 180, 181, 198]
    # decoding_nodes = [180, 181, 199, 214, 215]
    # for idx, val in weight_vector.items():
    #     # if any(i in relevant_nodes for i in idx):
    #     # # check if there are any weights with 3 components
    #     # count = 0
    #     # for node in decoding_nodes:
    #     #     if node in idx:
    #     #         count += 1
    #     # # if count >= 3:
    #     # if count == 2:
    #     # if 90 in idx and 198 not in idx:
    #     if 90 in idx and 198 in idx:
    #     if any(i in relevant_nodes for i in idx) and 90 not in idx and 198 not in idx:
    #         print(idx, val)

    # error correction rewrites maps
    # visualize.visualize_2d_cluster_end_state(strat, shape)
    # quit()

    syndrome_indices = list(correction_mapping_dict.values())
    reduced_maps = nsf.reduce_maps(state, [start_idx, target_idx] + syndrome_indices)
    compiled_maps = nsf.compile_maps(*reduced_maps)
    # print(syndrome_indices, start_idx, target_idx)
    #
    # sorted_map = sorted(
    #     list(zip(compiled_maps.weights, compiled_maps.noises)), reverse=True
    # )
    # print(1 - np.sum(compiled_maps.weights))
    # for weight, noise in sorted_map:
    #     print(weight, noise)
    # print("===========")
    # # error correction rewrites maps

    correction_rules = get_correction_rules_cluster_ring_xz_optimized(
        idx_dict=correction_mapping_dict, input_idx=start_idx, output_idx=target_idx
    )

    updated_weights = defaultdict(float)
    for weight, noise in zip(compiled_maps.weights, compiled_maps.noises):
        new_noise = list(noise)
        filtered_noise = list(
            filter(lambda noise_part: noise_part in syndrome_indices, noise)
        )
        for combination, to_flip in correction_rules.items():
            if all(
                [noise_part in combination for noise_part in filtered_noise]
            ) and all([combination_part in noise for combination_part in combination]):
                # print("aaaaah")
                new_noise = list(nsf.add_or_remove(to_flip, tuple(new_noise)))
        for idx in syndrome_indices:
            try:
                new_noise.remove(idx)
            except ValueError:
                pass
        new_noise = tuple(new_noise)
        # print(weight, noise, "mapped to", new_noise)
        updated_weights[new_noise] += weight

    weights = []
    noises = []
    for noise, weight in updated_weights.items():
        noises.append(noise)
        weights.append(weight)
    new_map = nsf.Map(weights, noises)
    state.maps = [new_map]
    for idx in syndrome_indices:
        state = nsf.x_measurement(state, idx)

    output_rho = nsf.noisy_bp_dm(
        state,
        target_indices=[start_idx, target_idx],
    )
    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    return fid


def run_direct(diagonal_distance, noise_parameter, save_strategy_plot_path=None):
    distance = diagonal_distance
    num_rows = distance + 11
    num_cols = distance + 11
    shape = (num_rows, num_cols)
    start_graph = graph.construct_2d_cluster_graph(shape)

    start_idx = 5 * num_cols + 5  # 105
    directions_to_target = [DIRECTION_RIGHT, DIRECTION_UP] * distance
    target_idx = get_idx_by_directions(start_idx, directions_to_target, shape)

    path = idx_sequence_by_directions(
        start_idx, directions_to_target[:-1], shape, start_inclusive=False
    )
    cutouts_start = [
        [DIRECTION_LEFT],
        [DIRECTION_DOWN],
        [DIRECTION_DOWN, DIRECTION_RIGHT],
    ]
    cutouts_target = [
        [DIRECTION_UP],
        [DIRECTION_RIGHT],
        [DIRECTION_RIGHT, DIRECTION_DOWN],
    ]
    cutouts = [
        get_idx_by_directions(start_idx, directions, shape)
        for directions in cutouts_start
    ] + [
        get_idx_by_directions(target_idx, directions, shape)
        for directions in cutouts_target
    ]

    seq = []
    seq += [("z", idx) for idx in cutouts]
    seq += [("x", idx, start_idx) for idx in path]

    strat = nsf.Strategy(start_graph, seq)
    if save_strategy_plot_path is not None:
        visualize.visualize_2d_cluster_end_state(
            strat, shape, show=False, save=save_strategy_plot_path
        )

    p = noise_parameter
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.State(start_graph, maps=[])
    state = nsf.pauli_noise(
        state,
        indices=range(num_rows * num_cols),
        coefficients=coefficients,
    )
    state = strat(state)
    strat.save()
    # weight_vector = strat.get_weight_vector_expression()
    #
    # interest_set = [start_idx, target_idx]
    # print(interest_set)
    # interest_set = list(sorted(interest_set))
    # new_weight_vector = defaultdict(list)
    #
    # for k, v in weight_vector.items():
    #     new_key = tuple()
    #     for idx in interest_set:
    #         if idx in k:
    #             new_key += (idx,)
    #     if new_key:
    #         new_weight_vector[new_key] += v
    #
    # for k, v in new_weight_vector.items():
    #     print(k, len(v), v)
    #
    # visualize.visualize_2d_cluster_end_state(strat, shape)
    # quit()

    output_rho = nsf.noisy_bp_dm(
        state,
        target_indices=[start_idx, target_idx],
    )

    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    return fid


def run_simulated_direct(diagonal_distance, noise_parameter):
    distance = diagonal_distance
    graph = nx.Graph([(0, 1)])
    state = nsf.State(graph, maps=[])

    p = noise_parameter
    # odd pattern: distance + 2 of the z measurements
    noise_map = nsf.Map(weights=[(1 - p ** (distance + 2)) / 2], noises=[(0,)])
    state = noise_map(state)
    # even pattern: (distance - 1) + 4 of the z-measurements
    noise_map = nsf.Map(weights=[(1 - p ** (distance + 3)) / 2], noises=[(1,)])
    state = noise_map(state)

    pauli_weights = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.pauli_noise(state, indices=[0, 1], coefficients=pauli_weights)

    output_rho = nsf.noisy_bp_dm(
        state,
        target_indices=[0, 1],
    )

    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    return fid


def run_modularized(
    diagonal_distance, noise_parameter, correction_strategy="xz_optimized"
):
    base_transport_length = (diagonal_distance - 2) * 2
    path_1_length = base_transport_length + 2 * 7
    path_2_length = base_transport_length + 2 * 2
    path_3_length = base_transport_length
    path_4_length = base_transport_length + 2 * 1
    path_5_length = base_transport_length + 2 * 7

    input_idx = 0
    output_idx = 6
    graph = nx.Graph(
        [(input_idx, i) for i in range(1, 6)] + [(i, output_idx) for i in range(1, 6)]
    )
    state = nsf.State(graph, maps=[])
    p = noise_parameter

    def apply_pattern(state, idx, parity_type, amount, p):
        if amount == 0:
            return state
        q = p**amount
        weight = (1 - q) / 2
        if parity_type == "odd":
            new_state = nsf.z_noise(state, [idx], epsilon=weight)
        elif parity_type == "even":
            neighboring_indices = [((idx - 1) + 1) % 5 + 1, ((idx - 1) - 1) % 5 + 1]
            noise_pattern = (input_idx,) + tuple(neighboring_indices)
            noise_map = nsf.Map(weights=[weight], noises=[noise_pattern])
            new_state = noise_map(state)
        else:
            raise ValueError(f"Unknown parity type {parity_type}. Must be odd or even.")
        return new_state

    # # for transport only
    for path_idx, path_length in zip(
        range(1, 6),
        [path_1_length, path_2_length, path_3_length, path_4_length, path_5_length],
    ):
        state = apply_pattern(
            state, idx=path_idx, parity_type="odd", amount=path_length // 2, p=p
        )
        state = apply_pattern(
            state, idx=path_idx, parity_type="even", amount=path_length // 2, p=p
        )

    ## noises from encoding/decoding
    # first, the ones that effectively just add to the length
    extra_amounts = {
        1: {"odd": 4 * 2, "even": 2 * 2},
        2: {"odd": 2 * 2, "even": 0 * 2},
        3: {"odd": 0 * 2, "even": 0 * 2},
        4: {"odd": 1 * 2, "even": 0 * 2},
        5: {"odd": 3 * 2, "even": 2 * 2},
    }
    for path_idx in range(1, 6):
        state = apply_pattern(
            state,
            idx=path_idx,
            parity_type="odd",
            amount=extra_amounts[path_idx]["odd"],
            p=p,
        )
        state = apply_pattern(
            state,
            idx=path_idx,
            parity_type="even",
            amount=extra_amounts[path_idx]["even"],
            p=p,
        )
    # from y measurements
    q = p**2
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(1, 5)])
    state = noise_map(state)
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(4, 5)])
    state = noise_map(state)
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 2, 3)])
    state = noise_map(state)
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(2, 3, output_idx)])
    state = noise_map(state)
    #
    # from z measurements
    q = p**2
    # 55
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(input_idx, 4, 5)])
    state = noise_map(state)
    # 57 has same pattern as 74y
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(4, 5)])
    state = noise_map(state)
    # 71
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(input_idx, 1, 2)])
    state = noise_map(state)
    # 75
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(input_idx, 3, 4)])
    state = noise_map(state)
    # 122
    noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(input_idx, 3)])
    state = noise_map(state)
    # 124 !
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(3, 4)])
    state = noise_map(state)
    # 196 !!
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 3, 4, output_idx)])
    state = noise_map(state)

    # decoding qubits
    pauli_weights = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.pauli_noise(state, indices=[1, 2, 3, 4, 5], coefficients=pauli_weights)

    # input/output
    pauli_weights = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.pauli_noise(
        state, indices=[input_idx, output_idx], coefficients=pauli_weights
    )

    state = nsf.x_measurement(state, index=1, b0=input_idx)
    correction_mapping_dict = {2: 2, 3: 3, 4: 4, 5: 5}
    new_state = perform_correction(
        state,
        correction_mapping_dict,
        input_idx,
        output_idx,
        correction_strategy=correction_strategy,
    )

    output_rho = nsf.noisy_bp_dm(
        new_state,
        target_indices=[input_idx, output_idx],
    )

    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    # print(fid)
    return fid


def run_alternative_encoded(diagonal_distance, noise_parameter):
    distance = diagonal_distance
    num_rows = distance + 14
    num_cols = distance + 14
    shape = (num_rows, num_cols)
    start_graph = graph.construct_2d_cluster_graph(shape)

    start_idx = 6 * num_cols + 7
    input_idx = start_idx - 2 * num_cols - 3
    target_idx = get_idx_by_directions(
        start_idx, [DIRECTION_UP, DIRECTION_RIGHT] * distance, shape
    )
    output_idx = target_idx + 3 * num_cols + 2
    start_dirs = get_alternative_code_directions()
    target_dirs = get_alternative_code_directions(mirrored=True)

    seq = []
    # first setup both sides
    seq += [
        ("y", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["setup_zs"]
    ]
    seq += [
        ("y", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_ys"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["setup_zs"]
    ]
    # cut out all other zs
    seq += [
        ("z", get_idx_by_directions(start_idx, dirs, shape))
        for dirs in start_dirs["path_cutouts"]
    ]
    seq += [
        ("z", get_idx_by_directions(target_idx, dirs, shape))
        for dirs in target_dirs["path_cutouts"]
    ]

    # strat = nsf.Strategy(start_graph, seq)
    # visualize.visualize_2d_cluster_end_state(strat, shape)

    # construct the paths
    path1 = (
        start_dirs["path1"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 1)
        + reverse_directions(target_dirs["path1"])[:-1]
    )
    # print(target_dirs["path1"])
    path1_start = get_idx_by_directions(start_idx, start_dirs["path1_start"], shape)
    path1_ids = idx_sequence_by_directions(
        start_idx=path1_start, directions=path1, shape=shape
    )

    path2 = (
        start_dirs["path2"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 4)
        + reverse_directions(target_dirs["path2"])[:-1]
    )
    path2_start = get_idx_by_directions(start_idx, start_dirs["path2_start"], shape)
    path2_ids = idx_sequence_by_directions(
        start_idx=path2_start, directions=path2, shape=shape
    )

    path3 = (
        start_dirs["path3"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 4)
        + reverse_directions(target_dirs["path3"])[:-1]
    )
    path3_start = get_idx_by_directions(start_idx, start_dirs["path3_start"], shape)
    path3_ids = idx_sequence_by_directions(
        start_idx=path3_start, directions=path3, shape=shape
    )

    path4 = (
        start_dirs["path4"]
        + [DIRECTION_RIGHT, DIRECTION_UP] * (distance - 3)
        + reverse_directions(target_dirs["path4"])[:-1]
    )
    path4_start = get_idx_by_directions(start_idx, start_dirs["path4_start"], shape)
    path4_ids = idx_sequence_by_directions(
        start_idx=path4_start, directions=path4, shape=shape
    )

    path5 = (
        start_dirs["path5"]
        + [DIRECTION_UP, DIRECTION_RIGHT] * (distance - 1)
        + reverse_directions(target_dirs["path5"])[:-1]
    )
    path5_start = get_idx_by_directions(start_idx, start_dirs["path5_start"], shape)
    path5_ids = idx_sequence_by_directions(
        start_idx=path5_start, directions=path5, shape=shape
    )
    # print(path1_ids)
    # print(path2_ids)
    # print(path3_ids)
    # print(path4_ids)
    # print(path5_ids)

    # move input and output out of the way
    seq += [("x", start_idx, input_idx)]
    seq += [
        ("z", get_idx_by_direction(input_idx, direction, shape))
        for direction in [DIRECTION_DOWN, DIRECTION_LEFT]
    ]
    seq += [("x", target_idx, output_idx)]
    seq += [
        ("z", get_idx_by_direction(output_idx, direction, shape))
        for direction in [DIRECTION_UP, DIRECTION_RIGHT]
    ]

    for path in [path1_ids, path2_ids, path3_ids, path4_ids, path5_ids]:
        seq += [("x", idx, input_idx) for idx in path]

    # strat = nsf.Strategy(start_graph, seq)
    # visualize.visualize_2d_cluster_end_state(strat, shape)

    # aux_strat = nsf.Strategy(start_graph, seq)
    # weight_vector = aux_strat.get_weight_vector_expression()
    #
    # interest_set = [84,315,313,271,232,234,255]
    # interest_set = list(sorted(interest_set))
    # new_weight_vector = defaultdict(list)
    #
    # for k, v in weight_vector.items():
    #     new_key = tuple()
    #     for idx in interest_set:
    #         if idx in k:
    #             new_key += (idx,)
    #     if new_key:
    #         new_weight_vector[new_key] += v
    #
    # for k, v in new_weight_vector.items():
    #     print(k, v)
    #     if "x_58" in v:
    #         print("aaaaaah")
    # aux_strat.save()
    # visualize.visualize_2d_cluster_end_state(aux_strat, shape)

    # do the final measurement
    seq += [
        (
            "x",
            get_idx_by_directions(target_idx, target_dirs["path1_start"], shape),
            input_idx,
        )
    ]

    correction_mapping_dict = {
        2: get_idx_by_directions(target_idx, target_dirs["path2_start"], shape),
        3: get_idx_by_directions(target_idx, target_dirs["path3_start"], shape),
        4: get_idx_by_directions(target_idx, target_dirs["path4_start"], shape),
        5: get_idx_by_directions(target_idx, target_dirs["path5_start"], shape),
    }

    strat = nsf.Strategy(start_graph, seq)
    # from time import time
    # start_time = time()
    # strat.populate_cache()
    # print(f"{shape[0]*shape[1]} Nodes. With {distance=}. Propagating all noises took {time()-start_time:.2f} seconds.")
    # strat.save()

    p = noise_parameter
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.State(start_graph, maps=[])
    state = nsf.pauli_noise(
        state,
        indices=range(num_rows * num_cols),
        coefficients=coefficients,
    )

    state = strat(state)
    strat.save()
    # visualize.visualize_2d_cluster_end_state(strat, shape)

    state = perform_correction(state, correction_mapping_dict, input_idx, output_idx)

    output_rho = nsf.noisy_bp_dm(
        state,
        target_indices=[input_idx, output_idx],
    )
    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]
    # print(fid)
    # visualize.visualize_2d_cluster_end_state(strat, shape)
    return fid


def run_alternative_modularized(
    diagonal_distance,
    noise_parameter,
    return_map=False,
    correction_strategy="xz_optimized",
):
    base_transport_length = (diagonal_distance - 2) * 2
    path_1_length = base_transport_length + 2 * 9
    path_2_length = base_transport_length + 2 * 1
    path_3_length = base_transport_length
    path_4_length = base_transport_length + 2 * 2
    path_5_length = base_transport_length + 2 * 8

    input_idx = 0
    output_idx = 6
    graph = nx.Graph(
        [(input_idx, i) for i in range(1, 6)] + [(i, output_idx) for i in range(1, 6)]
    )
    state = nsf.State(graph, maps=[])
    p = noise_parameter

    def apply_pattern(state, idx, parity_type, amount, p):
        if amount == 0:
            return state
        q = p**amount
        weight = (1 - q) / 2
        if parity_type == "odd":
            new_state = nsf.z_noise(state, [idx], epsilon=weight)
        elif parity_type == "even":
            neighboring_indices = [((idx - 1) + 1) % 5 + 1, ((idx - 1) - 1) % 5 + 1]
            noise_pattern = (input_idx,) + tuple(neighboring_indices)
            noise_map = nsf.Map(weights=[weight], noises=[noise_pattern])
            new_state = noise_map(state)
        else:
            raise ValueError(f"Unknown parity type {parity_type}. Must be odd or even.")
        return new_state

    # for transport only
    for path_idx, path_length in zip(
        range(1, 6),
        [path_1_length, path_2_length, path_3_length, path_4_length, path_5_length],
    ):
        state = apply_pattern(
            state, idx=path_idx, parity_type="odd", amount=path_length // 2, p=p
        )
        state = apply_pattern(
            state, idx=path_idx, parity_type="even", amount=path_length // 2, p=p
        )

    ## noises from encoding/decoding
    # first, the ones that effectively just add to the length
    extra_amounts = {
        1: {"odd": 6 * 2, "even": 2 * 2},
        2: {"odd": 2 * 2, "even": 0 * 2},
        3: {"odd": 0 * 2, "even": 0 * 2},
        4: {"odd": 1 * 2 + 1, "even": 0 * 2},
        5: {"odd": 3 * 2, "even": 2 * 2},
    }
    for path_idx in range(1, 6):
        state = apply_pattern(
            state,
            idx=path_idx,
            parity_type="odd",
            amount=extra_amounts[path_idx]["odd"],
            p=p,
        )
        state = apply_pattern(
            state,
            idx=path_idx,
            parity_type="even",
            amount=extra_amounts[path_idx]["even"],
            p=p,
        )
    # from other measurements
    # q = p ** 2
    # 85, 65 ++
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(input_idx, 2, 3, 4, 5)])
    state = noise_map(state)
    # ++ 295, 296
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(2, 3, 4, 5, output_idx)])
    state = noise_map(state)
    # 86 ++
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 1)])
    state = noise_map(state)
    # ++ 275
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(1, output_idx)])
    state = noise_map(state)
    # 88, 68 -- 235, 236
    noise_map = nsf.Map(weights=[(1 - p**4) / 2], noises=[(4, 5)])
    state = noise_map(state)
    # 89, 103, 128 -- 215, 334 ++
    noise_map = nsf.Map(weights=[(1 - p**5) / 2], noises=[(input_idx, 3, 4)])
    state = noise_map(state)
    # ++ 233
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(3, 4, output_idx)])
    state = noise_map(state)
    # 104 -- 314
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(2, 3, 4, 5)])
    state = noise_map(state)
    # 106 -- 274
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(1, 2, 3, 4)])
    state = noise_map(state)
    # 107 ++
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 1, 2, 3, 5)])
    state = noise_map(state)
    # ++ 254
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(1, 2, 3, 5, output_idx)])
    state = noise_map(state)
    # 125, 145 ++
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(input_idx, 1, 2, 3, 4)])
    state = noise_map(state)
    # ++ 292, 293
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(1, 2, 3, 4, output_idx)])
    state = noise_map(state)
    # 126, 146 ++
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(input_idx, 2, 3, 4)])
    state = noise_map(state)
    # ++ 273, 272
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(2, 3, 4, output_idx)])
    state = noise_map(state)
    # 276
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 4, output_idx)])
    state = noise_map(state)
    # 129 ++
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(2, 3)])
    state = noise_map(state)
    # ++ 213
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 2, 3, output_idx)])
    state = noise_map(state)
    # 144, 143, 165 -- 291, 312, 332
    noise_map = nsf.Map(weights=[(1 - p**6) / 2], noises=[(1, 2)])
    state = noise_map(state)
    # 167, 168, 185, 187 -- 290 ++
    noise_map = nsf.Map(weights=[(1 - p**5) / 2], noises=[(input_idx, 2, 3)])
    state = noise_map(state)
    # ++ 231, 250, 251
    noise_map = nsf.Map(weights=[(1 - p**3) / 2], noises=[(2, 3, output_idx)])
    state = noise_map(state)

    # transferring qubits
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx,)])
    state = noise_map(state)
    noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(output_idx,)])
    state = noise_map(state)

    # neighboring qubits
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(1, 2, 3, 4, 5)])
    state = noise_map(state)
    noise_map = nsf.Map(weights=[(1 - p**2) / 2], noises=[(1, 2, 3, 4, 5)])
    state = noise_map(state)

    # decoding qubits
    pauli_weights = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.pauli_noise(state, indices=[1, 2, 3, 4, 5], coefficients=pauli_weights)

    # input/output
    pauli_weights = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    state = nsf.pauli_noise(
        state, indices=[input_idx, output_idx], coefficients=pauli_weights
    )

    state = nsf.x_measurement(state, index=1, b0=input_idx)
    correction_mapping_dict = {2: 2, 3: 3, 4: 4, 5: 5}
    new_state = perform_correction(
        state,
        correction_mapping_dict,
        input_idx,
        output_idx,
        correction_strategy=correction_strategy,
    )

    output_rho = nsf.noisy_bp_dm(
        new_state,
        target_indices=[input_idx, output_idx],
    )

    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]

    if return_map:
        base_map = new_state.maps[0].as_standard_form()
        new_noises = []
        for noise in base_map.noises:
            new_noise = []
            if input_idx in noise:
                new_noise.append(0)
            if output_idx in noise:
                new_noise.append(1)
            new_noises.append(tuple(new_noise))
        return nsf.Map(base_map.weights, new_noises)
    else:
        return fid


def rewrite_output_map_as_paulis(output_map: nsf.Map):
    output_map = output_map.as_standard_form()
    pauli_weights = [0, 0, 0, 0]
    for w, noise in zip(output_map.weights, output_map.noises):
        if tuple(noise) == (1,):
            # z
            pauli_weights[3] += w
        elif tuple(noise) == (0,):
            # x
            pauli_weights[1] += w
        elif tuple(noise) == (0, 1):
            # y
            pauli_weights[2] += w
    pauli_weights[0] = 1 - np.sum(pauli_weights)
    return pauli_weights


def run_concatenated(
    diagonal_distance,
    noise_parameter,
    concatenation_levels=1,
    return_map=False,
    correction_strategy_inner="xz_optimized",
    correction_strategy_outer="xz_optimized",
):
    ## The way the noises are specified in this function recycle results from the modularization.
    ## For this to make exact sense here would mean having pre-established connections on auxiliary noiseless qubits
    ## However, luckily this is equivalent to a Bell state measurement in the sense we want
    p = noise_parameter
    inner_map = run_alternative_modularized(
        diagonal_distance,
        noise_parameter,
        return_map=True,
        correction_strategy=correction_strategy_inner,
    )
    for k in range(concatenation_levels):
        input_idx = 0
        output_idx = 6
        graph = nx.Graph(
            [(input_idx, i) for i in range(1, 6)]
            + [(i, output_idx) for i in range(1, 6)]
        )
        state = nsf.State(graph, maps=[])
        # apply noise from previous level
        inner_weights = rewrite_output_map_as_paulis(inner_map)
        state = nsf.pauli_noise(
            state, indices=[1, 2, 3, 4, 5], coefficients=inner_weights
        )
        # apply weights from outer preparation
        # first, from the y measurements
        q = p**2
        noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(1, 5)])
        state = noise_map(state)
        noise_map = nsf.Map(weights=[(1 - q) / 2], noises=[(4, 5)])
        state = noise_map(state)
        noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(input_idx, 2, 3)])
        state = noise_map(state)
        noise_map = nsf.Map(weights=[(1 - p) / 2], noises=[(2, 3, output_idx)])
        state = noise_map(state)

        # then from the measured "transport", i.e. measuring the 5 qubits from the input state
        def apply_pattern(state, idx, parity_type, amount, p):
            if amount == 0:
                return state
            q = p**amount
            weight = (1 - q) / 2
            if parity_type == "odd":
                new_state = nsf.z_noise(state, [idx], epsilon=weight)
            elif parity_type == "even":
                neighboring_indices = [((idx - 1) + 1) % 5 + 1, ((idx - 1) - 1) % 5 + 1]
                noise_pattern = (input_idx,) + tuple(neighboring_indices)
                noise_map = nsf.Map(weights=[weight], noises=[noise_pattern])
                new_state = noise_map(state)
            else:
                raise ValueError(
                    f"Unknown parity type {parity_type}. Must be odd or even."
                )
            return new_state

        for i in [1, 2, 3, 4, 5]:
            state = apply_pattern(state, i, parity_type="odd", amount=2, p=p)

        # input/output
        pauli_weights = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
        state = nsf.pauli_noise(
            state, indices=[input_idx, output_idx], coefficients=pauli_weights
        )

        state = nsf.x_measurement(state, index=1, b0=input_idx)
        correction_mapping_dict = {2: 2, 3: 3, 4: 4, 5: 5}
        new_state = perform_correction(
            state,
            correction_mapping_dict,
            input_idx,
            output_idx,
            correction_strategy=correction_strategy_outer,
        )

        base_map = new_state.maps[0].as_standard_form()
        new_noises = []
        for noise in base_map.noises:
            new_noise = []
            if input_idx in noise:
                new_noise.append(0)
            if output_idx in noise:
                new_noise.append(1)
            new_noises.append(tuple(new_noise))
        inner_map = nsf.Map(base_map.weights, new_noises)

    output_state = nsf.State(graph=gt.bipartite_graph, maps=[inner_map])

    output_rho = nsf.noisy_bp_dm(output_state, target_indices=[0, 1])

    fid = np.real_if_close(fidelity(gt.bell_pair_ket, output_rho))[0, 0]

    if return_map:
        return inner_map  # this is the one from the last iteration
    else:
        return fid


if __name__ == "__main__":
    pass
