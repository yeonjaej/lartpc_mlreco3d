# Defines cluster formation and feature extraction
import numpy as np
import numba as nb
import torch
from typing import List

from mlreco.utils.numba import numba_wrapper, cdist_nb, mean_nb, unique_nb

@numba_wrapper(cast_args=['data'], list_args=['cluster_classes'], keep_torch=True, ref_arg='data')
def form_clusters(data, min_size=-1, column=5, batch_index=0, cluster_classes=[-1], shape_index=-1):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up each of the clusters in the input tensor.

    Args:
        data (np.ndarray)      : (N,6-10) [x, y, z, batchid, value, id(, groupid, intid, nuid, shape)]
        min_size (int)         : Minimal cluster size
        column (int)           : Column in the tensor which contains cluster IDs
        batch_index (int)      : Column in the tensor which contains batch IDs
        cluster_classes ([int]): List of classes to include in the list of clusters
        shape_index (int)      : Column in the tensor which contains shape IDs
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    return _form_clusters(data, min_size, column, batch_index, cluster_classes, shape_index)

@nb.njit(cache=True)
def _form_clusters(data: nb.float64[:,:],
                   min_size: nb.int64 = -1,
                   column: nb.int64 = 5,
                   batch_index: nb.int64 = 0,
                   cluster_classes: nb.types.List(nb.int64) = nb.typed.List([-1]),
                   shape_index: nb.int64 = -1) -> nb.types.List(nb.int64[:]):

    # Create a mask which restricts the voxels to those with shape in cluster_classes
    restrict = False
    if cluster_classes[0] != -1:
        mask = np.zeros(len(data), dtype=np.bool_)
        for s in cluster_classes:
            mask |= (data[:, shape_index] == s)
        mask = np.where(mask)[0]
        restrict = True
    subdata = data[mask] if restrict else data

    # Loop over batches and cluster IDs, append cluster voxel lists
    clusts = []
    batch_ids = subdata[:, batch_index]
    for b in np.unique(batch_ids):
        binds = np.where(batch_ids == b)[0]
        clust_ids = subdata[binds, column]
        if restrict:
            binds = mask[binds]
        for c in np.unique(clust_ids):
            if c < 0:
                continue
            clust = np.where(clust_ids == c)[0]
            if len(clust) < min_size:
                continue
            clusts.append(binds[clust])

    return clusts


@numba_wrapper(cast_args=['data'], keep_torch=True, ref_arg='data')
def reform_clusters(data, clust_ids, batch_ids, column=5, batch_col=0):
    """
    Function that returns a list of of arrays of voxel IDs
    that make up the requested clusters.

    Args:
        data (np.ndarray)     : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clust_ids (np.ndarray): (C) List of cluster ids
        batch_ids (np.ndarray): (C) List of batch ids
        column (int)          : Column in the tensor which contains cluster IDs
    Returns:
        [np.ndarray]: (C) List of arrays of voxel IDs in each cluster
    """
    return _reform_clusters(data, clust_ids, batch_ids, column, batch_col)

@nb.njit(cache=True)
def _reform_clusters(data: nb.float64[:,:],
                     clust_ids: nb.int64[:],
                     batch_ids: nb.int64[:],
                     column: nb.int64 = 5,
                     batch_col: nb.int64 = 0) -> nb.types.List(nb.int64[:]):
    clusts = []
    for i in range(len(batch_ids)):
        clusts.append(np.where((data[:,batch_col] == batch_ids[i]) & (data[:,column] == clust_ids[i]))[0])
    return clusts


@numba_wrapper(cast_args=['data'], list_args=['clusts'])
def get_cluster_batch(data, clusts, batch_index=0):
    """
    Function that returns the batch ID of each cluster.
    This should be unique for each cluster, assert that it is.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of batch IDs
    """
    if len(clusts) > 0:
        return _get_cluster_batch(data, clusts, batch_index)
    else:
        return np.empty((0,), dtype=np.int32)

@nb.njit(cache=True)
def _get_cluster_batch(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       batch_index: nb.int64 = 0) -> nb.int64[:]:

    labels = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        assert len(np.unique(data[c, batch_index])) == 1
        labels[i] = data[c[0], batch_index]
    return labels


@numba_wrapper(cast_args=['data'], list_args=['clusts'])
def get_cluster_label(data, clusts, column=5):
    """
    Function that returns the majority label of each cluster,
    as specified in the requested data column.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        column (int)         : Column which specifies the cluster label
    Returns:
        np.ndarray: (C) List of cluster labels
    """
    return _get_cluster_label(data, clusts, column)

@nb.njit(cache=True)
def _get_cluster_label(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       column: nb.int64 = 5) -> nb.float64[:]:

    labels = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        v, cts = unique_nb(data[c, column])
        labels[i] = v[np.argmax(np.array(cts))]
    return labels


@numba_wrapper(cast_args=['data'], list_args=['clusts'])
def get_cluster_primary_label(data, clusts, column, cluster_column=5, group_column=6):
    """
    Function that returns the majority label of the primary component
    of a cluster, as specified in the requested data column.

    The primary component is identified by picking the set of label
    voxels that have a cluster_id identical to the cluster group_id.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        column (int)         : Column which specifies the cluster label
        cluster_column (int) : Column which specifies the cluster ID
        group_column (int)   : Column which specifies the cluster group ID
    Returns:
        np.ndarray: (C) List of cluster primary labels
    """
    return _get_cluster_primary_label(data, clusts, column, cluster_column, group_column)

@nb.njit(cache=True)
def _get_cluster_primary_label(data: nb.float64[:,:],
                               clusts: nb.types.List(nb.int64[:]),
                               column: nb.int64,
                               cluster_column: nb.int64 = 5,
                               group_column: nb.int64 = 6) -> nb.float64[:]:
    labels = np.empty(len(clusts), dtype=data.dtype)
    group_ids = _get_cluster_label(data, clusts, group_column)
    for i in range(len(clusts)):
        cluster_ids  = data[clusts[i], cluster_column]
        primary_mask = cluster_ids == group_ids[i]
        if len(data[clusts[i][primary_mask]]):
            v, cts = unique_nb(data[clusts[i][primary_mask], column])
        else: # If the primary is empty, use group
            v, cts = unique_nb(data[clusts[i], column])
        labels[i] = v[np.argmax(np.array(cts))]

    return labels


@numba_wrapper(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_momenta_label(data, clusts, column=8):
    """
    Function that returns the momentum unit vector of each cluster

    Args:
        data (np.ndarray)    : (N,12) [x, y, z, batchid, value, id, groupid, px, py, pz, p, pdg]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        column (int)         : Column which specifies the cluster ID
    Returns:
        np.ndarray: (C) List of cluster IDs
    """
    return _get_momenta_label(data, clusts, column)

@nb.njit(cache=True)
def _get_momenta_label(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       column: nb.int64 = 8) -> nb.float64[:]:
    labels = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        labels[i] = np.mean(data[c, column])
    return labels


@numba_wrapper(cast_args=['data'], list_args=['clusts', 'coords_index'], keep_torch=True, ref_arg='data')
def get_cluster_centers(data, clusts, coords_index=[1, 4]):
    """
    Function that returns the coordinate of the centroid
    associated with the listed clusters.

    Args:
        data (np.ndarray)    : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster centers
    """
    return _get_cluster_centers(data, clusts, coords_index)

@nb.njit(cache=True)
def _get_cluster_centers(data: nb.float64[:,:],
                         clusts: nb.types.List(nb.int64[:]),
                         coords_index: nb.types.List(nb.int64[:]) = [0, 3]) -> nb.float64[:,:]:
    centers = np.empty((len(clusts),3), dtype=data.dtype)
    for i, c in enumerate(clusts):
        centers[i] = np.sum(data[c, coords_index[0]:coords_index[1]], axis=0)/len(c)
    return centers


@numba_wrapper(cast_args=['data'], list_args=['clusts'])
def get_cluster_sizes(data, clusts):
    """
    Function that returns the sizes of
    each of the listed clusters.

    Args:
        data (np.ndarray)    : (N,5) [x, y, z, batchid, value]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster sizes
    """
    return _get_cluster_sizes(data, clusts)

@nb.njit(cache=True)
def _get_cluster_sizes(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:])) -> nb.int64[:]:
    sizes = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        sizes[i] = len(c)
    return sizes


@numba_wrapper(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_energies(data, clusts):
    """
    Function that returns the energies deposited by
    each of the listed clusters.

    Args:
        data (np.ndarray)    : (N,5) [x, y, z, batchid, value]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C) List of cluster energies
    """
    return _get_cluster_energies(data, clusts)

@nb.njit(cache=True)
def _get_cluster_energies(data: nb.float64[:,:],
                          clusts: nb.types.List(nb.int64[:])) -> nb.float64[:]:
    energies = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        energies[i] = np.sum(data[c, 4])
    return energies


@numba_wrapper(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_features(data: nb.float64[:,:],
                         clusts: nb.types.List(nb.int64[:]),
                         batch_col: nb.int64 = 0,
                         coords_col: nb.types.List(nb.int64[:]) = (1, 4)) -> nb.float64[:,:]:
    """
    Function that returns an array of 16 geometric features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,16) tensor of cluster features (center, orientation, direction, size)
    """
    return _get_cluster_features(data, clusts, batch_col=batch_col, coords_col=coords_col)

@nb.njit(cache=True)
def _get_cluster_features(data: nb.float64[:,:],
                          clusts: nb.types.List(nb.int64[:]),
                          batch_col: nb.int64 = 0,
                          coords_col: nb.types.List(nb.int64[:]) = (1, 4)) -> nb.float64[:,:]:
    feats = np.empty((len(clusts), 16), dtype=data.dtype)
    ids = np.arange(len(clusts)).astype(np.int64) # prange creates a uint64 iterator which is cast to int64 to access a list,
                                                  # and throws a warning. To avoid this, use a separate counter to acces clusts.
    for k in nb.prange(len(clusts)):
        # Get list of voxels in the cluster
        clust = clusts[ids[k]]
        x = data[clust, coords_col[0]:coords_col[1]]

        # Center data
        center = mean_nb(x, 0)
        x = x - center

        # Get orientation matrix
        A = x.T.dot(x)

        # Get eigenvectors, normalize orientation matrix and eigenvalues to largest
        # If points are superimposed, i.e. if the largest eigenvalue != 0, no need to keep going
        w, v = np.linalg.eigh(A)
        if w[2] == 0.:
            feats[k] = np.concatenate((center, np.zeros(12), np.array([len(clust)])))
            continue
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:,2]

        # Projection all points, x, along the principal axis
        x0 = x.dot(v0)

        # Evaluate the distance from the points to the principal axis
        xp0 = x - np.outer(x0, v0)
        np0 = np.empty(len(xp0), dtype=data.dtype)
        for i in range(len(xp0)):
            np0[i] = np.linalg.norm(xp0[i])

        # Flip the principal direction if it is not pointing towards the maximum spread
        sc = np.dot(x0, np0)
        if sc < 0:
            v0 = np.zeros(3, dtype=data.dtype)-v0 # (-)/negative doesn't work with numba for now...

        # Weight direction
        v0 = dirwt * v0

        # Append
        feats[k] = np.concatenate((center, B.flatten(), v0, np.array([len(clust)])))

    return feats


@numba_wrapper(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_features_extended(data, clusts, batch_col=0, coords_col=(1, 4)):
    """
    Function that returns the an array of 3 additional features for
    each of the clusters in the provided list.

    Args:
        data (np.ndarray)    : (N,X) Data tensor [x,y,z,batch_id,value,...,sem_type]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster features (mean value, std value, major sem_type)
    """
    return _get_cluster_features_extended(data, clusts, batch_col=batch_col, coords_col=coords_col)

def _get_cluster_features_extended(data: nb.float64[:,:],
                                   clusts: nb.types.List(nb.int64[:]),
                                   batch_col: nb.int64 = 0,
                                   coords_col: nb.types.List(nb.int64[:]) = (1, 4)) -> nb.float64[:,:]:
    feats = np.empty((len(clusts), 3), dtype=data.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get mean and RMS energy in the cluster
        clust = clusts[ids[k]]
        mean_value = np.mean(data[clust,4])
        std_value = np.std(data[clust,4])

        # Get the cluster semantic class
        types, cnts = unique_nb(data[clust,-1])
        major_sem_type = types[np.argmax(cnts)]

        feats[k] = [mean_value, std_value, major_sem_type]

    return feats


@numba_wrapper(cast_args=['data','particles'], list_args=['clusts','coords_index'], keep_torch=True, ref_arg='data')
def get_cluster_points_label(data, particles, clusts, random_order=True, batch_col=0, coords_index=[1, 4]):
    """
    Function that gets label points for each cluster.
    Returns start point of primary shower fragment twice if shower, delta or Michel
    and end points of tracks if track.

    Args:
        data (torch.tensor)     : (N,X) Voxel coordinates [batch_id, x, y, z, ...]
        particles (torch.tensor): (N,9) Point coordinates [batch_id, start_x, start_y, start_z, last_x, last_y, last_z, start_t, shape_id]
                                (obtained with parse_particle_coords)
        clusts ([np.ndarray])   : (C) List of arrays of voxel IDs in each cluster
        random_order (bool)     : Whether or not to shuffle the start and end points randomly
    Returns:
        np.ndarray: (N,6) cluster-wise start and end points (in RANDOMIZED ORDER by default)
    """
    return _get_cluster_points_label(data, particles, clusts, random_order,
                                    batch_col=batch_col,
                                    coords_index=coords_index)

@nb.njit(cache=True)
def _get_cluster_points_label(data: nb.float64[:,:],
                              particles: nb.float64[:,:],
                              clusts: nb.types.List(nb.int64[:]),
                              random_order: nb.boolean = True,
                              batch_col: nb.int64 = 0,
                              coords_index: nb.types.List(nb.int64[:]) = [1, 4]) -> nb.float64[:,:]:

    # Get start and end points (one and the same for all but track class)
    batch_ids = _get_cluster_batch(data, clusts)
    points = np.empty((len(clusts), 6), dtype=data.dtype)
    for i, c in enumerate(clusts):
        batch_mask = np.where(particles[:,batch_col] == batch_ids[i])[0]
        clust_ids  = np.unique(data[c, 5]).astype(np.int64)
        minid = np.argmin(particles[batch_mask][clust_ids,-2]) # Pick the first cluster in time
        order = np.arange(6) if (np.random.choice(2) or not random_order) else np.array([3, 4, 5, 0, 1, 2])
        points[i] = particles[batch_mask][clust_ids[minid]][order+1] # The first column is the batch ID

    # Bring the start points to the closest point in the corresponding cluster
    for i, c in enumerate(clusts):
        dist_mat = cdist_nb(points[i].reshape(-1,3), data[c,coords_index[0]:coords_index[1]])
        argmins  = np.empty(len(dist_mat), dtype=np.int64)
        for j in range(len(dist_mat)):
            argmins[j] = np.argmin(dist_mat[j])
        points[i] = data[c][argmins, coords_index[0]:coords_index[1]].reshape(-1)

    return points


@numba_wrapper(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_start_points(data, clusts):
    """
    Function that estimates the start point of clusters
    based on their PCA and local curvature.

    Args:
        data (np.ndarray)    : (N,3) Voxel coordinates [x, y, z]
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
    Returns:
        np.ndarray: (C,3) tensor of cluster start points
    """
    return _get_cluster_start_points(data, clusts)

@nb.njit(parallel=True, cache=True)
def _get_cluster_start_points(data: nb.float64[:,:],
                              clusts: nb.types.List(nb.int64[:])) -> nb.float64[:,:]:
    points = np.empty((len(clusts), 3))
    for k in nb.prange(len(clusts)):
        vid = cluster_end_points(data[clusts[k],:3])[-1]

    return points


@numba_wrapper(cast_args=['data','starts'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_directions(data, starts, clusts, max_dist=-1, optimize=False):
    """
    Finds the orientation of all the clusters.

    Args:
        data (torch.tensor)  : (N,3) Voxel coordinates [x, y, z]
        starts (torch.tensor): (C,3) Coordinates of the start points
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
        optimize (bool)      : Optimizes the number of points involved in the estimate
    Returns:
        torch.tensor: (3) Orientation
    """
    return _get_cluster_directions(data, starts, clusts, max_dist, optimize)

@nb.njit(parallel=True, cache=True)
def _get_cluster_directions(data: nb.float64[:,:],
                            starts: nb.float64[:,:],
                            clusts: nb.types.List(nb.int64[:]),
                            max_dist: nb.float64 = -1,
                            optimize: nb.boolean = False) -> nb.float64[:,:]:

    dirs = np.empty(starts.shape, data.dtype)
    ids  = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Weird bug here: without the cast (astype), throws a strange noncontiguous error on reshape...
        dirs[k] = cluster_direction(data[clusts[ids[k]],:3], starts[k].astype(np.float64), max_dist, optimize)

    return dirs


@numba_wrapper(cast_args=['data','values','starts'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_dedxs(data, values, starts, clusts, max_dist=-1):
    """
    Finds the start dEdxs of all the clusters.

    Args:
        data (torch.tensor)  : (N,3) Voxel coordinates [x, y, z]
        values (torch.tensor): (N) Voxel values
        starts (torch.tensor): (C,3) Coordinates of the start points
        clusts ([np.ndarray]): (C) List of arrays of voxel IDs in each cluster
        max_dist (float)     : Max distance between start voxel and other voxels
    Returns:
        torch.tensor: (N) dEdx values for each cluster
    """
    return _get_cluster_dedxs(data, values, starts, clusts, max_dist)

@nb.njit(parallel=True)
def _get_cluster_dedxs(data: nb.float64[:,:],
                       values: nb.float64[:],
                       starts: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       max_dist: nb.float64 = -1) -> nb.float64[:,:]:

    dedxs = np.empty(len(clusts), data.dtype)
    ids   = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Weird bug here: without the cast (astype), throws a strange noncontiguous error on reshape...
        dedxs[k] = cluster_dedx(data[clusts[ids[k]],:3], values[clusts[ids[k]]], starts[k].astype(np.float64), max_dist)

    return dedxs


@nb.njit(cache=True)
def cluster_end_points(voxels: nb.float64[:,:]) -> (nb.float64[:], nb.float64[:]):
    """
    Finds the start point of a cluster by:
    1. Find the principal axis a of the point cloud
    2. Find the coordinate a_i of each point along this axis
    3. Find the points with minimum and maximum coordinate
    4. Find the point that has the largest umbrella curvature

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
    Returns:
        int: ID of the start voxel
    """
    # Get the axis of maximum spread
    axis = principal_axis(voxels)

    # Compute coord values along that axis
    coords = np.empty(len(voxels))
    for i in range(len(coords)):
        coords[i] = np.dot(voxels[i], axis)

    # Compute curvature of the extremities
    ids = [np.argmin(coords), np.argmax(coords)]

    # Sort the voxel IDs by increasing order of curvature order
    curvs = [umbrella_curv(voxels, ids[0]), umbrella_curv(voxels, ids[1])]
    ids[np.argsort(curvs)]

    # Return extrema
    return voxels[ids[0]], voxels[ids[1]]


@nb.njit(cache=True)
def cluster_direction(voxels: nb.float64[:,:],
                      start: nb.float64[:],
                      max_dist: nb.float64 = -1,
                      optimize: nb.boolean = False) -> nb.float64[:]:
    """
    Estimates the orientation of a cluster:
    - By default takes the normalized mean direction
      from the cluster start point to the cluster voxels
    - If max_dist is specified, restricts the cluster voxels
      to those within a max_dist radius from the start point
    - If optimize is True, selects the neighborhood which
      minimizes the transverse spread w.r.t. the direction

    Args:
        voxels (torch.tensor): (N,3) Voxel coordinates [x, y, z]
        start (torch.tensor) : (C,3) Coordinates of the start point
        max_dist (float)     : Max distance between start voxel and other voxels
        optimize (bool)      : Optimizes the number of points involved in the estimate
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist
    if not optimize and max_dist > 0:
        dist_mat = cdist_nb(start.reshape(1,-1), voxels).flatten()
        voxels = voxels[dist_mat <= max_dist]
        if len(voxels) < 2:
            return np.zeros(3, dtype=voxels.dtype)

    # If optimize is set, select the radius by minimizing the transverse spread
    elif optimize:
        # Order the cluster points by increasing distance to the start point
        dist_mat = cdist_nb(start.reshape(1,-1), voxels).flatten()
        order = np.argsort(dist_mat)
        voxels = voxels[order]
        dist_mat = dist_mat[order]

        # Find the PCA relative secondary spread for each point
        labels = np.zeros(len(voxels), dtype=voxels.dtype)
        meank = mean_nb(voxels[:3], 0)
        covk = (np.transpose(voxels[:3]-meank) @ (voxels[:3]-meank))/3
        for i in range(2, len(voxels)):
            # Get the eigenvalues and eigenvectors, identify point of minimum secondary spread
            w, _ = np.linalg.eigh(covk)
            labels[i] = np.sqrt(w[2]/(w[0]+w[1])) if (w[0]+w[1]) else 0.
            if dist_mat[i] == dist_mat[i-1]:
                labels[i-1] = 0.

            # Increment mean and matrix
            if i != len(voxels)-1:
                meank = ((i+1)*meank+voxels[i+1])/(i+2)
                covk = (i+1)*covk/(i+2) + (voxels[i+1]-meank).reshape(-1,1)*(voxels[i+1]-meank)/(i+1)

        # Subselect voxels that are most track-like
        max_id = np.argmax(labels)
        voxels = voxels[:max_id+1]

    # Compute mean direction with respect to start point, normalize it
    rel_voxels = np.empty((len(voxels), 3), dtype=voxels.dtype)
    for i in range(len(voxels)):
        rel_voxels[i] = voxels[i]-start
    mean = mean_nb(rel_voxels, 0)
    if np.linalg.norm(mean):
        return mean/np.linalg.norm(mean)
    return mean


@nb.njit(cache=True)
def umbrella_curv(voxels: nb.float64[:,:],
                  voxid: nb.int64) -> nb.float64:
    """
    Computes the umbrella curvature as in equation 9 of "Umbrella Curvature:
    A New Curvature Estimation Method for Point Clouds" by A.Foorginejad and K.Khalili
    (https://www.sciencedirect.com/science/article/pii/S2212017313006828)

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
        voxid  (int)       : Voxel ID in which to compute the curvature
    Returns:
        int: Value of the curvature in voxid with respect to the rest of the point cloud
    """
    # Find the mean direction from that point
    refvox = voxels[voxid]
    axis = np.mean([v-refvox for v in voxels], axis=0)
    axis /= np.linalg.norm(axis)

    # Find the umbrella curvature (mean angle from the mean direction)
    return abs(np.mean([np.dot((voxels[i]-refvox)/np.linalg.norm(voxels[i]-refvox), axis) for i in range(len(voxels)) if i != voxid]))


@nb.njit(cache=True)
def principal_axis(voxels:nb.float64[:,:]) -> nb.float64[:]:
    """
    Computes the direction of the principal axis of a cloud of points
    by computing its eigenvectors.

    Args:
        voxels (np.ndarray): (N,3) Voxel coordinates [x, y, z]
    Returns:
        int: (3) Coordinates of the principal axis
    """
    # Center data
    center = mean_nb(voxels, 0)
    x = voxels - center

    # Get orientation matrix
    A = x.T.dot(x)

    # Get eigenvectors, select the one which corresponds to the maximal spread
    _, v = np.linalg.eigh(A)
    return v[:,2]


@nb.njit
def cluster_dedx(voxels: nb.float64[:,:],
                 values: nb.float64[:],
                 start: nb.float64[:],
                 max_dist: nb.float64 = 5) -> nb.float64[:]:
    """
    Estimates the initial dEdx of a cluster

    Args:
        voxels (torch.tensor): (N,4) Voxel coordinates [x, y, z]
        values (torch.tensor): (N) Voxel values
        starts (torch.tensor): (C,3) Coordinates of the start points
        max_dist (float)     : Max distance between start voxel and other voxels
    Returns:
        torch.tensor: (3) Orientation
    """
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist
    dist_mat = cdist_nb(start.reshape(1,-1), voxels).flatten()
    if max_dist > 0:
        voxels = voxels[dist_mat <= max_dist]
        if len(voxels) < 2:
            return 0.
        values = values[dist_mat <= max_dist]
        dist_mat = dist_mat[dist_mat <= max_dist]

    # Compute the total energy in the neighborhood and the max distance, return ratio
    return np.sum(values)/np.max(dist_mat)
