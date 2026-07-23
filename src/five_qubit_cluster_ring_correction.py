"""Helper functions for the measurement based decoding of the five qubit cluster ring code.

"""


def get_correction_rules_cluster_ring(idx_dict, input_idx, output_idx):
    """Get the standard correction rules for the five qubit cluster ring code.

    These are for the graph-based implementation with the first decoding qubit
    already measured in x (including LC correction).
    These are the standard way to assign syndromes that corrects any
    single qubit error.


    Parameters
    ----------
    idx_dict : dict
        keys have to be 2, 3, 4, 5 with the associated values mapping them
        to the corresponding decoding qubit indices.
    input_idx : int
        Index of the qubit connected on the input direction.
    output_idx : int
        Index of the qubit connected on the output direction.

    Returns
    -------
    dict
        With syndromes as keys and correction operations as values.

    """
    combinations_to_flip = {
        # single z noises (the others have id as correction)
        (2, 3, 4, 5): (input_idx,),
        # single x noises
        (2, 5): (output_idx,),
        (2, 4, 5): (input_idx, output_idx),
        (2, 4): (output_idx,),
        (3, 5): (output_idx,),
        (2, 3, 5): (input_idx, output_idx),
        # y noises
        (3, 4): (input_idx, output_idx),
        (4, 5): (input_idx, output_idx),
        (2, 3, 4): (output_idx,),
        (3, 4, 5): (output_idx,),
        (2, 3): (input_idx, output_idx),
    }
    return {
        tuple(idx_dict[i] for i in combination): value
        for combination, value in combinations_to_flip.items()
    }


def get_correction_rules_cluster_ring_xz_optimized(idx_dict, input_idx, output_idx):
    """Get the optimized correction rules for the five qubit cluster ring code.

    These are for the graph-based implementation with the first decoding qubit
    already measured in x (including LC correction).
    These are an alternative way to assign syndromes that reassigns the y-error syndromes
    (because these get mapped to z and x errors by MBQC-style transport) to the most common
    correlated noise pattern we found.

    Parameters
    ----------
    idx_dict : dict
        keys have to be 2, 3, 4, 5 with the associated values mapping them
        to the corresponding decoding qubit indices.
    input_idx : int
        Index of the qubit connected on the input direction.
    output_idx : int
        Index of the qubit connected on the output direction.

    Returns
    -------
    dict
        With syndromes as keys and correction operations as values.

    """
    combinations_to_flip = {
        # single z noises (the others have id as correction)
        (2, 3, 4, 5): (input_idx,),
        # single x noises
        (2, 5): (output_idx,),
        (2, 4, 5): (input_idx, output_idx),
        (2, 4): (output_idx,),
        (3, 5): (output_idx,),
        (2, 3, 5): (input_idx, output_idx),
        # leftover patterns
        # (3, 4): no correction,
        # (4, 5): no correction,
        (2, 3, 4): (input_idx,),
        (3, 4, 5): (input_idx,),
        # (2, 3): no correction,
    }
    return {
        tuple(idx_dict[i] for i in combination): value
        for combination, value in combinations_to_flip.items()
    }
