# Defines GNN network accuracy metrics
import numpy as np
from mlreco.utils.metrics import SBD, AMI, ARI, purity_efficiency
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import softmax

def edge_assignment(edge_index, groups, binary=False):
    """
    Function that determines which edges are turned on based
    on the group ids of the clusters they are connecting.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        groups (np.ndarray)    : (C) List of group ids
        binary (bool)          : True if the assigment must be adapted to binary loss
    Returns:
        np.ndarray: (E) Boolean array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that belong to the same batch and the same group
    edge_assn = np.array([groups[e[0]] == groups[e[1]] for e in edge_index], dtype=int)

    # If binary loss will be used, transform to -1,+1 instead of 0,1
    if binary:
        edge_assn = 2*edge_assn - 1
    return edge_assn


def edge_assignment_from_graph(edge_index, true_edge_index, binary=False):
    """
    Function that determines which edges are turned on based
    on the group ids of the clusters they are connecting.

    Args:
        edge_index (np.ndarray): (E,2) Constructed incidence matrix
        edge_index (np.ndarray): (E,2) True incidence matrix
    Returns:
        np.ndarray: (E) Boolean array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that belong to the same batch and the same group
    edge_assn = np.array([np.any([(e == pair).all() for pair in true_edge_index]) for e in edge_index], dtype=int)

    # If binary loss will be used, transform to -1,+1 instead of 0,1
    if binary:
        edge_assn = 2*edge_assn - 1
    return edge_assn


def cluster_to_voxel_label(clusts, node_label):
    """
    Function that turns an array of labels on clusters
    to an array of labels on voxels.

    Args:
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        node_label (np.ndarray): (C) List of node labels
    Returns:
        np.ndarray: (N) List of voxel labels
    """
    nvoxels = np.sum([len(c) for c in clusts])
    vlabel = np.empty(nvoxels)
    stptr = 0
    for i, c in enumerate(clusts):
        endptr = stptr + len(c)
        vlabel[stptr:endptr] = node_label[i]
        stptr = endptr

    return vlabel


def find_parent(parent, i):
    """
    Function that recursivey finds the parent node id.

    Args:
        parent (np.ndarray): (C) List of current group ids for all the node
        i (int)              : Index of the node of which to find the parent
    Returns:
        int: Parent id
    """
    if i != parent[i]:
        parent[i] = find_parent(parent, parent[i])

    return parent[i]


def node_assignment(edge_index, edge_label, n):
    """
    Function that assigns each node to a group, based
    on the edge assigment provided. This uses a simple
    union find implementation.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        edge_assn (np.ndarray) : (E) Boolean array (1 if edge is on)
        n (int)                  : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    groups = {}
    group_ids = np.arange(n)
    on_edges = edge_index[np.where(edge_label)[0]]
    for i, j in on_edges: 
        leaderi = group_ids[i]
        leaderj = group_ids[j]
        if leaderi in groups:
            if leaderj in groups:
                if leaderi == leaderj: continue # nothing to do
                groupi = groups[leaderi]
                groupj = groups[leaderj]
                if len(groupi) < len(groupj):
                    i, leaderi, groupi, j, leaderj, groupj = j, leaderj, groupj, i, leaderi, groupi
                groupi |= groupj
                del groups[leaderj]
                for k in groupj:
                    group_ids[k] = leaderi
            else:
                groups[leaderi].add(j)
                group_ids[j] = leaderi
        else:
            if leaderj in groups:
                groups[leaderj].add(i)
                group_ids[i] = leaderj
            else:
                group_ids[i] = group_ids[j] = i
                groups[i] = set([i, j])

    return group_ids

def node_assignment_UF(edge_index, edge_wt, n, thresh=0.0):
    """
    Function that assigns each node to a group, based on the edge
    weights provided, by using the topologylayer implementation
    of union find.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        edge_wt (np.ndarray)   : (E) Array of edge weights
        n (int)                : Total number of clusters C
        thresh (double)        : Threshold for edge association
    Returns:
        np.ndarray: (C) List of group ids
    """
    from topologylayer.functional.persistence import getClustsUF_raw

    edges = edge_index.flatten()

    val = edge_wt

    cs = getClustsUF_raw(edges, val, n, thresh)
    un, cinds = np.unique(cs, return_inverse=True)
    return cinds


def node_assignment_bipartite(edge_index, edge_label, primaries, n):
    """
    Function that assigns each node to a group represented
    by a primary node. This function loops over secondaries and
    associates it to the primary with that is connected to it
    with the strongest edge.

    Args:
        edge_index (np.ndarray): (E,2) Incidence matrix
        edge_label (np.ndarray): (E) Array of edge scores
        primaries (np.ndarray) : (P) List of primary ids
        n (int)                : Total number of clusters C
    Returns:
        np.ndarray: (C) List of group ids
    """
    group_ids = np.arange(n)
    others = [i for i in range(n) if i not in primaries]
    for i in others:
        inds = edge_index[:,1] == i
        if sum(inds) == 0:
            continue
        indmax = np.argmax(edge_label[inds])
        group_ids[i] = edge_index[inds,0][indmax].item()

    return group_ids


def node_assignment_group(group_ids, batch_ids):
    """
    Function that assigns each node to a group, given
    group ids at each batch and corresponding batch ids

    Args:
        group_ids (np.ndarray): (C) List of cluster group ids within each batch
        batch_ids (np.ndarray): (C) List of cluster batch ids
    Returns:
        np.ndarray: (C) List of unique group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    joined = np.vstack((group_ids, batch_ids))
    _, unique_ids = np.unique(joined, axis=1, return_inverse=True)
    return unique_ids


def clustering_metrics(clusts, node_assn, node_pred):
    """
    Function that assigns each node to a group, based
    on the edge assigment provided.

    Args:
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        node_assn (np.ndarray): (C) List of true node group labels
        node_pred (np.ndarray): (C) List of predicted node group labels
    Returns:
        double: Adjusted Rand Index
        double: Adjusted Mutual Information
        double: Symmetric Best Dice
        double: Purity
        double: Efficiency
    """
    pred_vox = cluster_to_voxel_label(clusts, node_pred)
    true_vox = cluster_to_voxel_label(clusts, node_assn)
    ari = ARI(pred_vox, true_vox)
    ami = AMI(pred_vox, true_vox)
    sbd = SBD(pred_vox, true_vox)
    pur, eff = purity_efficiency(pred_vox, true_vox)
    return ari, ami, sbd, pur, eff


def voxel_efficiency_bipartite(clusts, node_assn, node_pred, primaries):
    """
    Function that evaluates the fraction of secondary
    voxels that are associated to the corresct primary.

    Args:
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        node_assn (np.ndarray): (C) List of true node group labels
        node_pred (np.ndarray): (C) List of predicted node group labels
        primaries (np.ndarray): (P) List of primary ids
    Returns:
        double: Fraction of correctly assigned secondary voxels
    """
    others = [i for i in range(n) if i not in primaries]
    tot_vox = np.sum([len(clusts[i]) for i in others])
    int_vox = np.sum([len(clusts[i]) for i in others if node_pred[i] == node_assn[i]])
    return int_vox * 1.0 / tot_vox


# Function that computes the graph score
def graph_score(on_mat, edge_index):
    # Assign groups based of the current index
    pred_groups = node_assignment(edge_index, np.ones(len(edge_index)), on_mat.shape[0])
    # Make a matrix which is True when two clusters are in the same group, False otherwise
    group_mat = np.array([int(i == j) for i in pred_groups for j in pred_groups]).reshape(on_mat.shape)
    # Use on score if same group, off score otherwise
    return np.sum(group_mat*on_mat+(1-group_mat)*(1-on_mat))

# Function that finds the graph with the highest score
def find_optimal_graph(edge_index, edge_scores, nclusters):
    # Interpret the score as a distance matrix, build an MST based on score
    edge_scores = softmax(edge_scores, axis=1)
    on_mat = np.ones((nclusters, nclusters))
    for i, e in enumerate(edge_index):
        on_mat[e[0], e[1]] = edge_scores[i,1]
    mst_mat = minimum_spanning_tree(1-on_mat).toarray()
    mst_index = np.array(np.where(mst_mat != 0)).T
    # Order the mst index by increasing order of ON score
    args = np.argsort([on_mat[e[0], e[1]] for e in mst_index])
    mst_index = mst_index[args]
    # Now iteratively remove edges, until the total score cannot be improved any longer
    best_score = graph_score(on_mat, mst_index)
    best_index = mst_index
    found_better = True
    while found_better:
        found_better = False
        for i in range(len(best_index)):
            # Update index
            last_index = np.vstack((best_index[:i],best_index[i+1:]))
            last_score = graph_score(on_mat, last_index)
            if last_score > best_score:
                best_score = last_score
                best_index = last_index
                found_better = True
                break
    return best_index


def node_assignment_advance(edge_index, edge_pred, n):
    '''
    Advanced node assignment using Francois's functions above
    '''
    best_index = find_optimal_graph(edge_index, edge_pred, n)
    return node_assignment(best_index, np.ones(len(best_index)), n)
