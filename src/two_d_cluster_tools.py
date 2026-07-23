DIRECTION_UP = 1
DIRECTION_RIGHT = 2
DIRECTION_DOWN = 3
DIRECTION_LEFT = 4


def get_idx_by_direction(start_idx, direction, shape):
    """Get idx of vertex one step in given direction.

    Parameters
    ----------
    start_idx : int
    direction : int
        one of DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN or DIRECTION_LEFT
    shape : tuple[int]
        shape of the 2d cluster

    Returns
    -------
    int
        The idx of the requested vertex.

    Raises
    ------
    ValueError
        If the direction is not one of DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT.

    """
    num_rows, num_cols = shape
    if direction == DIRECTION_UP:
        new_idx = start_idx + num_cols
    elif direction == DIRECTION_RIGHT:
        new_idx = start_idx + 1
    elif direction == DIRECTION_DOWN:
        new_idx = start_idx - num_cols
    elif direction == DIRECTION_LEFT:
        new_idx = start_idx - 1
    else:
        raise ValueError(f"{direction=} is recognised.")
    return new_idx


def get_idx_by_directions(start_idx, directions, shape):
    """Get idx of vertex by following a sequence of directions.

    Parameters
    ----------
    start_idx : idx
    directions : list[int]
        A sequence of directions: DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN or DIRECTION_LEFT
    shape : tuple[int]
        shape of the 2d cluster

    Returns
    -------
    int
        The idx of the requested vertex.
    """
    current_idx = start_idx
    for direction in directions:
        current_idx = get_idx_by_direction(current_idx, direction, shape)
    return current_idx


def idx_sequence_by_directions(
    start_idx, directions, shape, start_inclusive=True, repeat=1
):
    """Get a list of all indices along a path described by a sequence of directions.

    Parameters
    ----------
    start_idx : int
    directions : list[int]
        A sequence of directions: DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN or DIRECTION_LEFT
    shape : tuple[int]
        shape of the 2d cluster
    start_inclusive : bool, optional
        If true, includes `start_idx` as the first element in the returned path. (default: True)
    repeat : int, optional
        The direction pattern in `directions` is repeated this many times. (default: 1)

    Returns
    -------
    list[int]
        A list of all indices along the path.
    """
    current_idx = start_idx
    sequence = []
    if start_inclusive:
        sequence.append(start_idx)
    directions = directions * repeat
    for direction in directions:
        current_idx = get_idx_by_direction(current_idx, direction, shape)
        sequence.append(current_idx)
    return sequence
