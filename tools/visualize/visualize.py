from functools import lru_cache
from itertools import cycle

import networkx as nx
import noisy_graph_states as nsf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import graphepp as gg

COLORS = {
    "default": "paleturquoise",  # "#1f78b4",
    "measure_z": "gray",
    "measure_x": "orange",
    "measure_y": "pink",
    "targets": ["red", "purple", "lime", "maroon", "navy", "darkolivegreen"],
    "noise": "darkgray",
}

NODE_SIZE = 500
FIGURE_SIZE_INCHES = (10, 10)
ANIMATION_INITIAL_FRAMES = 5
ANIMATION_FINAL_FRAMES = 5
ANIMATION_INTERVAL = 400  # ms
ANIMATION_REPEAT_DELAY = 0


def get_colors(graph: nx.Graph):
    color_data = graph.nodes.data("color", default=COLORS["default"])
    colors = [color_data[node] for node in graph]
    return colors


def color_graph_by_instruction(graph: nx.Graph, instructions, requested_sets):
    """Modify color attribute of graph in place according to measurements and targets.

    Parameters
    ----------
    graph : nx.Graph
    instructions : Iterable[tuple]
    requested_sets : Iterable[tuple]

    Returns
    -------
    None
    """
    for targets, color in zip(requested_sets, cycle(COLORS["targets"])):
        for idx in targets:
            graph.nodes[idx]["color"] = color
    for instruction in instructions:
        instruction_type = instruction[0]
        qubit_index = instruction[1]
        if instruction_type == "x":
            graph.nodes[qubit_index]["color"] = COLORS["measure_x"]
        elif instruction_type == "z":
            graph.nodes[qubit_index]["color"] = COLORS["measure_z"]
        elif instruction_type == "y":
            graph.nodes[qubit_index]["color"] = COLORS["measure_y"]


@lru_cache(maxsize=100)
def grid_positions(shape, offset=0):
    num_vertices = shape[0] * shape[1]
    num_cols = shape[1]
    pos_dict = {offset + i: (i % num_cols, i // num_cols) for i in range(num_vertices)}
    return pos_dict


def draw(graph: nx.Graph, pos_dict: dict, ax=None, show=True, save: [str, None] = None):
    if ax is None:
        fig = plt.figure(figsize=FIGURE_SIZE_INCHES)
    nx.draw_networkx(
        graph,
        pos_dict,
        with_labels=True,
        node_color=get_colors(graph),
        node_size=NODE_SIZE,
        ax=ax,
    )
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    if show:
        plt.show()


def draw_on_grid(
    graph: [gg.Graph, nx.Graph], shape, ax=None, show=True, save: [str, None] = None
):
    if isinstance(graph, gg.Graph):
        graph = nx.Graph(graph.adj)

    pos_dict = grid_positions(shape)
    draw(graph, pos_dict, ax=ax, show=show, save=save)


def visualize_2d_cluster_strategy(
    strategy: nsf.Strategy,
    shape: [tuple, None],
    requested_sets=(),
    show=True,
    save: [str, None] = None,
):
    if shape is None:
        guessed_dimension = int(np.sqrt(len(strategy.graph)))
        if guessed_dimension**2 != len(strategy.graph):
            raise ValueError(
                f"Trying to guess shape of 2D Cluster with N={len(strategy.graph)} vertices failed. "
                + f"It cannot be a square graph."
            )
        else:
            shape = (guessed_dimension, guessed_dimension)
    else:
        assert shape[0] * shape[1] == len(strategy.graph)

    pos_dict = grid_positions(shape)  # put vertices on a grid
    visualize_strategy(
        strategy, pos_dict=pos_dict, requested_sets=requested_sets, show=show, save=save
    )


def visualize_strategy(
    strategy: nsf.Strategy,
    pos_dict,
    requested_sets=(),
    show=True,
    save: [str, None] = None,
):
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES)
    ims = []
    for i, g in enumerate(strategy._graph_sequence):
        color_graph_by_instruction(
            graph=g, instructions=strategy.sequence[:i], requested_sets=requested_sets
        )

        edge_artist = nx.draw_networkx_edges(g, pos_dict)
        node_artist = nx.draw_networkx_nodes(
            g, pos_dict, node_color=get_colors(g), node_size=NODE_SIZE
        )
        labels_dict = nx.draw_networkx_labels(g, pos_dict)
        im = []
        if edge_artist:
            im.append(edge_artist)
        if node_artist:
            im.append(node_artist)
        im += list(labels_dict.values())
        ims.append(im)

    ims = [ims[0]] * ANIMATION_INITIAL_FRAMES + ims + [ims[-1]] * ANIMATION_FINAL_FRAMES
    ani = animation.ArtistAnimation(
        fig,
        ims,
        interval=ANIMATION_INTERVAL,
        blit=False,
        repeat_delay=ANIMATION_REPEAT_DELAY,
    )

    if save is not None:
        ani.save(save)
    if show:
        plt.show()


def visualize_2d_cluster_end_state(
    strategy: nsf.Strategy,
    shape,
    requested_sets=(),
    ax=None,
    show=True,
    save: [str, None] = None,
):
    graph = strategy._graph_sequence[-1]
    color_graph_by_instruction(
        graph=graph, instructions=strategy.sequence, requested_sets=requested_sets
    )
    draw_on_grid(graph, shape, ax=ax, show=show, save=save)


def visualize_strategy_end_state(
    strategy: nsf.Strategy,
    pos_dict,
    requested_sets=(),
    ax=None,
    show=True,
    save: [str, None] = None,
):
    graph = strategy._graph_sequence[-1]
    color_graph_by_instruction(
        graph=graph, instructions=strategy.sequence, requested_sets=requested_sets
    )
    draw(graph, pos_dict, ax=ax, show=show, save=save)


def visualize_noise_interactive(strategy):
    pass


def color_according_to_output_noise(graph, strategy, source_noise: str):
    weight_vector = strategy.get_weight_vector_expression()
    noise = ()
    for k, v in weight_vector.items():
        if source_noise in v:
            noise = k
            break
    for node in noise:
        graph.nodes[node]["color"] = COLORS["noise"]


def reset_colors(graph):
    for node in graph.nodes:
        graph.nodes[node]["color"] = COLORS["default"]


def visualize_2d_cluster_noise_interactive(
    strategy: nsf.Strategy, shape, requested_sets=()
):
    graph = strategy._graph_sequence[-1]
    pos = grid_positions(shape=shape)
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES)
    ax = plt.gca()
    ax.set_title("")

    class OnClickHandler(object):
        def __init__(self, noise_types=("x_", "y_", "z_")):
            self.noise_types = noise_types
            self.counter = 0
            self.previous_node = None

        def on_click(self, event):
            if event.inaxes == ax:
                x = event.xdata
                y = event.ydata

                # is the mouse click near a node?
                closest_node = min(
                    graph.nodes,
                    key=lambda n: np.linalg.norm(np.array(pos[n]) - np.array([x, y])),
                )
                distance = np.linalg.norm(
                    np.array(pos[closest_node]) - np.array([x, y])
                )
                if not distance < 0.3:
                    return
                if closest_node == self.previous_node:
                    self.counter = (self.counter + 1) % len(self.noise_types)
                else:
                    self.counter = 0
                node_name = str(closest_node)
                reset_colors(graph)
                color_graph_by_instruction(
                    graph=graph,
                    instructions=strategy.sequence,
                    requested_sets=requested_sets,
                )
                noise_name = self.noise_types[self.counter] + node_name
                color_according_to_output_noise(
                    graph=graph, strategy=strategy, source_noise=noise_name
                )
                ax.clear()
                ax.set_title(noise_name)
                nx.draw_networkx(
                    graph,
                    grid_positions(shape),
                    with_labels=True,
                    node_color=get_colors(graph),
                    node_size=NODE_SIZE,
                    ax=ax,
                )
                self.previous_node = closest_node
                plt.draw()

    click_handler = OnClickHandler()
    fig.canvas.mpl_connect("button_press_event", click_handler.on_click)
    color_graph_by_instruction(
        graph=graph, instructions=strategy.sequence, requested_sets=requested_sets
    )
    nx.draw_networkx(
        graph,
        grid_positions(shape),
        with_labels=True,
        node_color=get_colors(graph),
        node_size=NODE_SIZE,
        ax=ax,
    )
    plt.show()
