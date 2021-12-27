import numpy as np

class AffinityMatrix:
    import vector as __vector

    def __init__(self, pcl_array):
        self.__pcl_array = self.__array_to_vec(pcl_array)
        self.__num_pcls = self.__get_array_size()

    def __array_to_vec(self, array):
        """Takes input of (n,4)-d array of n particles 4-momenta, and
        returns an array of Lorentz 4-vectors.

        Note: assumes format x, y, z, e.
        """
        return self.__vector.array(
            [tuple(pcl) for pcl in array],
            dtype=[("x", float), ("y", float), ("z", float), ("e", float)]
            )

    def __get_array_size(self):
        array = self.__pcl_array
        if type(array) == self.__vector.MomentumNumpy4D:
            size = array.size
        elif type(array) == np.ndarray:
            size = array.shape[0]
        else:
            raise TypeError('input 4 momenta array not ndarray or vector')
        return size

    def __delta_R_cols(self):
        array = self.__pcl_array
        size = self.__num_pcls
        # slide the particle lists over all pairs
        for shift in range(size): # 0th shift is trivial as both same
            yield array[shift].deltaR(array[shift:])

    def delta_R(self):
        """Returns a symmetric matrix of delta R vals from input
        4-momentum array.
        """
        size = self.__num_pcls
        aff = np.zeros((size, size), dtype=np.float64)
        dR_cols = self.__delta_R_cols()
        for idx, col in enumerate(dR_cols):
            aff[idx:, idx] = col
            aff[idx, idx:] = col
        return aff

def knn_adj(matrix, self_loop=False, k=8, weighted=False, row=True,
            dtype=None):
    """Produce a directed adjacency matrix with outward edges
    towards the k nearest neighbours, determined from the input
    affinity matrix.
    
    Keyword arguments:
        matrix (2d numpy array): particle affinities
        self_loop (bool): if False will remove self-edges
        k (int): number of nearest neighbours in result
        weighted (bool): if True edges weighted by affinity,
            if False edge is binary
        row (bool): if True outward edges given by rows, if False cols
        dtype (numpy): data type: type of the output array
            note: must be floating point if weighted is True
    """
    axis = 0 # calculate everything row-wise
    if self_loop == False:
        k = k + 1 # for when we get rid of self-neighbours
    knn_idxs = np.argpartition(matrix, kth=k, axis=axis)
    near = knn_idxs[:k]
    edge_weights = 1
    if weighted == True:
        if dtype is None:
            dtype = matrix.dtype.type
        else:
            if not isinstance(dtype(1), np.floating):
                raise ValueError(
                    "Update the dtype parameter passed to this function "
                    + "to a numpy floating point type for weighted output."
                    )
        edge_weights = np.take_along_axis(matrix, near, axis=axis)
    else:
        dtype = np.bool_
    adj = np.zeros_like(matrix, dtype=dtype)
    np.put_along_axis(adj, near, edge_weights, axis=axis)
    if row == False:
        adj = adj.T
    if self_loop == False:
        np.fill_diagonal(adj, 0)
    return adj

def fc_adj(num_nodes, self_loop=False, dtype=np.bool_):
    """Create a fully connected adjacency matrix.
    """
    adj = np.ones((num_nodes, num_nodes), dtype=dtype)
    if self_loop == False:
        np.fill_diagonal(adj, 0)
    return adj
