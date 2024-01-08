from pathlib import Path
from numbers import Integral, Real
from heapq import heapify, heappop, heappush
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster._agglomerative import _fix_connectivity, _hc_cut
from sklearn.cluster import _hierarchical_fast as _hierarchical
from sklearn.utils import check_array
from sklearn.utils._param_validation import (
    Interval,
    StrOptions
)
from contarg.utils import _cross_corr
from scipy.cluster.hierarchy import set_link_color_palette
from contarg.hierarchical import plot_dendrogram
from matplotlib.colors import ListedColormap, rgb2hex
from matplotlib import pyplot as plt
import seaborn as sns


def repdist_tree(X, X_for_connected_components, connectivity,
                 n_clusters=None, return_distance=False, spearman=False, use_median=False):
    """Clustering that maximizes the feature correlation between each cluster member and the
    representative member of that cluster. The representative member is the cluster member
    with a feature set closest to the median feature set of the cluster.

    Recursively merges the pair of clusters that minimally increases
    within-cluster maximal distance between each cluster member and the representative member.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Adapted from sklearn.cluster._agglomerative.ward_tree

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    X_for_connected_components : array-like of shape (n_samples, n_features_for_connected_components)
        Feature matrix representing `n_samples` samples to be used when reattaching disconnected components.
        This is different from X so that you can cluster time-series but reattached disconnected
        components based on spatial proximity (for example).

    connectivity : {array-like, sparse matrix}
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        `n_clusters` should be less than `n_samples`.  Stop early the
        construction of the tree at `n_clusters.` This is useful to decrease
        computation time if the number of clusters is not small compared to the
        number of samples. In this case, the complete tree is not computed, thus
        the 'children' output is of limited use, and the 'parents' output should
        rather be used. This option is valid only when specifying a connectivity
        matrix.

    return_distance : bool, default=False
        If `True`, return the distance between the clusters.

    spearman : bool, default=False
        If `True`, use Spearman rank correlation instead of Pearson for all correlations.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes,) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Only returned if `return_distance` is set to `True` (for compatibility).
        The maximal correlation distance between any cluster member and
        the representative member.

    use_median : bool, default=False
        If `True`, use the median feature set as the representative feature set instead of the
        feature set most similar to the median.
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]

    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        if n_clusters > n_samples:
            raise ValueError(
                "Cannot provide more clusters than samples. "
                "%i n_clusters was asked, and there are %i "
                "samples." % (n_clusters, n_samples)
            )
        n_nodes = 2 * n_samples - n_clusters

    connectivity, n_connected_components = _fix_connectivity(X_for_connected_components, connectivity, affinity='euclidean')

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(
            len(row)
            * [
                ind,
            ]
        )
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.intp, order="C")
    coord_col = np.array(coord_col, dtype=np.intp, order="C")

    rep_features = np.zeros((n_nodes, n_features), order="C")
    rep_features[:n_samples] = X

    distances = (eval_rep_features(X[[cr, cc]], spearman=spearman, use_median=use_median) for cr, cc in zip(coord_row, coord_col))

    inertia = list(zip(distances, coord_row, coord_col))
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []
    if return_distance:
        distances = np.empty(n_nodes - n_samples)

    not_visited = np.empty(n_nodes, dtype=bool, order="C")

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j] = k, k
        children.append((i, j))
        used_node[i] = used_node[j] = False
        if return_distance:  # store inertia value
            distances[k - n_samples] = inert


        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _hierarchical._get_parents(A[i], coord_col, parent, not_visited)
        _hierarchical._get_parents(A[j], coord_col, parent, not_visited)

        [A[col].append(k) for col in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.intp, order="C")
        coord_row = np.empty(coord_col.shape, dtype=np.intp, order="C")
        coord_row.fill(k)
        n_additions = len(coord_row)
        # get representative features for new cluster
        member_ixs = _hierarchical._hc_get_descendent(k, children, n_samples)
        update_rep_features(X[member_ixs], rep_features, k, spearman=spearman)
        potential_reps = np.empty((n_additions, n_features), dtype=X.dtype, order="C")
        # generator function for calculate new distances for all potential next additions
        ini = \
        (eval_rep_features(X[member_ixs + _hierarchical._hc_get_descendent(pp, children, n_samples)], spearman=spearman)
        for pp in coord_col)

        [heappush(inertia, (ini_ix, k, cc_ix)) for ini_ix, cc_ix in zip(ini, coord_col)]

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    # sort children to get consistent output with unstructured version
    children = [c[::-1] for c in children]
    children = np.array(children)  # return numpy array for efficient caching

    if return_distance:
        return children, n_connected_components, n_leaves, parent, distances
    else:
        return children, n_connected_components, n_leaves, parent


def eval_rep_features(x, spearman=False, use_median=False):
    med_features = np.median(x, axis=0)
    if use_median:
        x_to_rep, _, _ = _cross_corr(med_features.reshape(1, -1), x, corr_only=True, spearman=spearman)
        return 1 - x_to_rep.min()
    else:
        x_to_med, _, _ = _cross_corr(med_features.reshape(1, -1), x, corr_only=True, spearman=spearman)
        x_to_med = x_to_med[0]
        try:
            ix = (x_to_med == x_to_med.max())
            rep_feat = x[ix]
            other_feat = x[~ix]
            x_to_rep, _, _ = _cross_corr(rep_feat, other_feat, corr_only=True, spearman=spearman)
            return 1 - x_to_rep.min()
        except ValueError:
            ix = (x_to_med == x_to_med.max())
            tmp = np.zeros(ix.shape, dtype=bool)
            tmp[np.nonzero(ix)[0][0]] = True
            ix = tmp
            rep_feat = x[ix]
            other_feat = x[~ix]
            x_to_rep, _, _ = _cross_corr(rep_feat, other_feat, corr_only=True, spearman=spearman)
            return 1 - x_to_rep.min()


def update_rep_features(x, rep_features, k, spearman=False, use_median=False):
    med_features = np.median(x, axis=0)
    if use_median:
        rep_features[k] = med_features.squeeze()
    else:
        x_to_med, _, _ = _cross_corr(med_features.reshape(1, -1), x, corr_only=True, spearman=spearman)
        try:
            rep_features[k] = x[(x_to_med == x_to_med.max()).squeeze()]
        except ValueError:
            rep_features[k] = x[(x_to_med == x_to_med.max()).squeeze()][0]


class RepdistClustering(ClusterMixin, BaseEstimator):
    """
    Representative Distance Clustering.

    The representative member is the cluster member with a feature set
    closest to the median feature set of the cluster.

    Recursively merges the pair of clusters that minimally increases
    within-cluster maximal distance between each cluster member and the
    representative member.


    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    connectivity : array-like or callable
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.

    X_for_connected_components : array-like of shape (n_samples, n_features_for_connected_components)
        Feature matrix representing `n_samples` samples to be used when
        reattaching disconnected components. This is different from X so that
        you can cluster time-series but reattache disconnected components based
        on spatial proximity (for example).

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

    spearman : bool, default=False
        If `True`, use Spearman rank correlation instead of Pearson for all correlations.

    use_median : bool, default=False
        If `True`, use the median feature set as the representative feature set instead of the
        feature set most similar to the median.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    Notes
    -----
    Based on Scikit-Learns AgglomerativeClustering Class
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left"), None],
        "connectivity": ["array-like"],
        "X_for_connected_components": ["array-like"],
        "compute_full_tree": [StrOptions({"auto"}), "boolean"],
        "distance_threshold": [Interval(Real, 0, None, closed="left"), None],
        "compute_distances": ["boolean"],
        "spearman": ["boolean"],
        "use_median": ["boolean"]
    }

    def __init__(
            self,
            n_clusters=2,
            *,
            connectivity=None,
            X_for_connected_components=None,
            compute_full_tree="auto",
            distance_threshold=None,
            compute_distances=False,
            spearman=False,
            use_median=False
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.connectivity = connectivity
        self.X_for_connected_components = X_for_connected_components
        self.compute_full_tree = compute_full_tree
        self.compute_distances = compute_distances
        self.spearman = spearman
        self.use_median = use_median

    def fit(self, X, y=None):
        """Fit the representative distance custering from features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        self._validate_params()
        X = self._validate_data(X, ensure_min_samples=2)
        return self._fit(X)

    def _fit(self, X):
        """Fit without validation

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """

        if not ((self.n_clusters is None) ^ (self.distance_threshold is None)):
            raise ValueError(
                "Exactly one of n_clusters and "
                "distance_threshold has to be set, and the other "
                "needs to be None."
            )

        if self.distance_threshold is not None and not self.compute_full_tree:
            raise ValueError(
                "compute_full_tree must be True if distance_threshold is set."
            )

        tree_builder = repdist_tree

        connectivity = self.connectivity
        if self.connectivity is not None:
            connectivity = check_array(
                connectivity, accept_sparse=["csr", "coo", "lil"]
            )

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree

        if compute_full_tree == "auto":
            if self.distance_threshold is not None:
                compute_full_tree = True
            else:
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, 0.02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None

        # Construct the tree
        kwargs = {}

        distance_threshold = self.distance_threshold

        return_distance = (distance_threshold is not None) or self.compute_distances

        out = tree_builder(
            X,
            X_for_connected_components=self.X_for_connected_components,
            connectivity=connectivity,
            n_clusters=n_clusters,
            return_distance=return_distance,
            spearman=self.spearman,
            use_median=self.use_median,
            **kwargs,
        )
        (self.children_, self.n_connected_components_, self.n_leaves_, parents) = out[
                                                                                  :4
                                                                                  ]

        if return_distance:
            self.distances_ = out[-1]

        if self.distance_threshold is not None:  # distance_threshold is used
            self.n_clusters_ = (
                    np.count_nonzero(self.distances_ >= distance_threshold) + 1
            )
        else:  # n_clusters is used
            self.n_clusters_ = self.n_clusters

        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reassign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self

    def fit_predict(self, X, y=None):
        """Fit and return the result of each sample's clustering assignment.

        In addition to fitting, this method also returns the result of the
        clustering assignment for each sample in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)


def cluster_and_plot(ts, thresh, verts, coords, connectivity=None, min_verts=None, spearman=False,
                     use_median=False, plot=True, plot_path=None, plot_kwargs=None):
    """
    Cluster time series based on the previously chosen threshold and plot results.

    Parameters
    ----------
    ts : ndarray
        A numpy array of shape (n_vertices, n_timepoints) representing the time series data.
    thresh : float
        The clustering threshold to use.
    verts : ndarray
        A numpy array of shape (n_vertices) representing the indicies of the verticies in the original surface.
    coords : ndarray
        A numpy array of shape (n_vertices, 3 dimension) representing the spatial coordinates of each vertex.
    connectivity : ndarray, optional
        A numpy array of shape (n_vertices, n_vertices) representing the connectivity matrix. Default is None.
    min_verts : int, optional
        The minimum number of vertices for a cluster to be considered. Default is None.  If none, 2% of number of vertices is used.
    plot : bool, optional
        Whether to generate a plot of the clustering results. Default is True.
    use_median : bool, default=False
        If `True`, use the median feature set as the representative feature set instead of the
        feature set most similar to the median.
    plot_path : pathlib Path, optional
        If provided, save plots to this path
    plot_kargs : dict
        kwargs for saving the plot

    Returns
    -------
    labels : ndarray
        A numpy array of shape (n_vertices,) representing the cluster labels for each time series.
    good_idx : ndarray
        A boolean numpy array of shape (n_vertices,) indicating which vertices pass the entropy and minimum cluster size threholds.
    """

    rdc = RepdistClustering(n_clusters=None,
                            connectivity=connectivity,
                            X_for_connected_components=coords,
                            distance_threshold=thresh,
                            compute_full_tree=True,
                            compute_distances=True,
                            spearman=spearman,
                            use_median=use_median
                            )
    labels = rdc.fit_predict(ts)

    if min_verts is None:
        min_verts = len(labels) * 0.02

    label_ids, label_counts = np.unique(labels, return_counts=True)
    big_labels = label_ids[label_counts >= min_verts]
    bl_idx = np.isin(labels, big_labels)

    good_idx = bl_idx

    good_labels = labels[good_idx]
    good_label_ids = np.unique(good_labels)

    # relabel clusters to make them more plotable
    transdict = dict(zip(good_label_ids, range(0, len(good_label_ids))))
    idx = np.nonzero(list(transdict.keys()) == good_labels[:, None])[1]
    labels_to_plot = np.asarray(list(transdict.values()))[idx]
    nlabels = len(label_ids)
    clust_colors = ListedColormap(
        np.array(sns.color_palette("hls", nlabels))[np.random.choice(nlabels, nlabels, replace=False)])
    label_colors = clust_colors(labels_to_plot)

    if plot:
        set_link_color_palette([rgb2hex(color) for color in clust_colors.colors])
        fig, axes = plt.subplots(2, 1, figsize=(5, 10))
        ax = axes[0]
        plot_dendrogram(rdc, color_threshold=thresh, ax=axes[0])
        xlim = ax.get_xlim()
        ax.hlines(thresh, xlim[0], xlim[1], linestyle='dashed')

        ax = axes[1]
        ax.scatter(coords[:, 1][good_idx], coords[:, 2][good_idx], c=clust_colors(labels_to_plot))
        ax.scatter(coords[:, 1][~good_idx], coords[:, 2][~good_idx], c="#5d5d5d", marker='v')
        ax.set_aspect('equal')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        if plot_path:
            if not plot_kwargs:
                plot_kwargs = {}
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(plot_path, **plot_kwargs)
    return labels, good_idx, label_colors