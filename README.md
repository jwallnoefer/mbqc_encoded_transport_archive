# Measurement-based error correction for long-range entanglement generation

This repository is an archive for the code used in:

> Measurement-based error correction for long-range entanglement generation <br>
> A. Romanova, J. Wallnöfer and W. Dür <br>
> Preprint: on arXiv soon

## Repository structure
The file `src/collected_plots.py` has been used to generate the plots in the paper.
It also contains code for some additional figures we have used for related
posters and presentations. The main logic for the different scenarios considered
in this work can be found in `src/protocol_from_cluster_state.py`.

The `tools` directory contains some additional visualization tools to be used with
the `noisy_graph_states` package, which may be helpful in investigating the
code in detail (some interesting spots to use these have been left in the files
as comments).

## How to use
If you wish to run the code yourself, you will need a Python version >=3.12 and
install the packages `noisy_graph_states` and `matplotlib` from PyPI. Furthermore,
in order for the imports to work properly, you need to install the packages
provided in the `tools` directory.
The visualizations also require `graphepp` for historical reasons.

A surefire way to ensure you have all requirements at appropriate versions is to
set up a new virtual environment with Python 3.12 and run:
```bash
pip install -r requirements.txt
```


## Related projects
This project uses the `noisy_graph_states` package
([jwallnoefer/noisy_graph_states](https://github.com/jwallnoefer/noisy_graph_states)),
which is an implementation of the Noisy Stabilizer Formalism introduced in
M. F. Mor-Ruiz, W. Dür; [Phys. Rev. A **107**, 032424 (2023)](https://doi.org/10.1103/PhysRevA.107.032424)
applied specifically to graph states. It describes how noisy graph states transform
under operations and measurements by not only updating the underlying state
but also the individual noise channels according to graph transformation rules.
