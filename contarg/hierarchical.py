from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import nilearn as nl
from nilearn import image, masking
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from scipy import stats
from scipy.spatial import distance
from .utils import select_confounds, idxs_to_flat, idxs_to_zoomed, idxs_to_mask

Cluster = namedtuple("Cluster", "id idxs nvox concentration repts medts")


def clust_within_region(
    mask_path,
    timeseries_path,
    confound_path,
    n_dummy=4,
    t_r=2.5,
    smoothing_fwhm=None,
    confound_selectors=None,
    distance_threshold=0.5,
    adjacency=False,
    metric="precomputed",
    linkage="single",
    clustering=None,
    clustering_kwargs=None,
):
    if confound_selectors is None:
        confound_selectors = ["-motion", "-dummy"]
    if isinstance(mask_path, Path):
        mask_path = mask_path.as_posix()
    if isinstance(timeseries_path, Path):
        timeseries_path = timeseries_path.as_posix()
    if isinstance(confound_path, Path):
        confound_path = confound_path.as_posix()
    if clustering_kwargs is None:
        clustering_kwargs = {}

    sub_mask = nl.image.load_img(mask_path)
    sub_img = nl.image.load_img(timeseries_path)
    cfds = select_confounds(confound_path, confound_selectors)

    rawts = nl.masking.apply_mask(sub_img, sub_mask, smoothing_fwhm=smoothing_fwhm)[
        n_dummy:
    ]
    clnts = nl.signal.clean(
        rawts,
        confounds=cfds[n_dummy:],
        t_r=t_r,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        standardize="psc",
    )
    spr, p = stats.spearmanr(clnts)
    if adjacency:
        connectivity = grid_to_graph(
            sub_mask.shape[0],
            sub_mask.shape[1],
            sub_mask.shape[2],
            mask=sub_mask.get_fdata(),
            return_as=np.ndarray,
        )
    else:
        connectivity = None
    if clustering is None:
        agg = AgglomerativeClustering(
            n_clusters=None,
            metric=metric,
            compute_full_tree=True,
            connectivity=connectivity,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
    else:
        agg = clustering(**clustering_kwargs)
    # use 1 - spr here to convert the correlation matrix to a distance matrix
    vox_clust_labels = agg.fit_predict(1 - spr)
    cluster_ids, cluster_counts = np.unique(vox_clust_labels, return_counts=True)
    vcl_img = nl.masking.unmask(vox_clust_labels + 1, sub_mask)

    # create cluster objects and append to list
    clusters = []
    for nvox, cid in zip(cluster_counts, cluster_ids):
        idxs = np.where(vox_clust_labels == cid)[0]
        clust_tses = clnts[:, idxs]
        if nvox == 1:
            median_ts = clust_tses.squeeze()
            rep_ts = clust_tses.squeeze()
            concentration = np.nan
            vox_idxs = np.nonzero(vcl_img.get_fdata() == (cid + 1))
            vox_idxs = np.array([(x, y, z, 1) for x, y, z in list(zip(*vox_idxs))])
        else:
            median_ts = np.percentile(clust_tses, 50, axis=1)
            # calculate spearmanr between median ts and each clust_ts
            clust_to_median, _ = stats.spearmanr(median_ts, clust_tses)
            clust_to_median = clust_to_median[0, 1:]
            rep_ts = clust_tses[:, clust_to_median == clust_to_median.max()].squeeze()
            # get a list of voxel indicies of the cluster
            vox_idxs = np.nonzero(vcl_img.get_fdata() == (cid + 1))
            vox_idxs = np.array([(x, y, z, 1) for x, y, z in list(zip(*vox_idxs))])
            vox_locs = np.matmul(vcl_img.affine, vox_idxs.T).T[:, :3]
            assert len(vox_locs) == nvox
            concentration = len(vox_idxs) / distance.pdist(vox_locs).mean()
        cluster = Cluster(cid, vox_idxs[:, :3], nvox, concentration, rep_ts, median_ts)
        clusters.append(cluster)

    return clnts, spr, vox_clust_labels, clusters


def find_target(
    row,
    img_col,
    targout_col,
    distance_threshold=0.5,
    adjacency=False,
    return_df=False,
    write_pkl=False,
    **kwargs,
):
    stimts_cln, stim_spr, stim_vcl, stim_clusters = clust_within_region(
        row.clnstim_mask,
        row[img_col],
        row.confounds,
        distance_threshold=distance_threshold,
        adjacency=adjacency,
        **kwargs,
    )

    refts_cln, ref_spr, ref_vcl, ref_clusters = clust_within_region(
        row.clnref_mask,
        row[img_col],
        row.confounds,
        distance_threshold=distance_threshold,
        adjacency=adjacency,
        **kwargs,
    )

    stim_clust_df = pd.DataFrame(stim_clusters)
    stim_clust_ts = np.array([cc.medts for cc in stim_clusters]).T
    stim_sizes = np.array([cc.nvox for cc in stim_clusters])

    ref_clust_ts = np.array([cc.medts for cc in ref_clusters]).T
    ref_sizes = np.array([cc.nvox for cc in ref_clusters])

    clust_corr, _ = stats.spearmanr(stim_clust_ts, ref_clust_ts)
    try:
        clust_corr = clust_corr[: stim_clust_ts.shape[1], stim_clust_ts.shape[1] :]
        stim_ref_corr = (clust_corr * ref_sizes).sum(1)
    except IndexError:
        stim_ref_corr = (clust_corr * ref_sizes)[0]
    stim_clust_df["net_reference_correlation"] = stim_ref_corr
    for kk, vv in row["entities"].items():
        stim_clust_df[kk] = vv
    if "session" not in stim_clust_df.columns:
        stim_clust_df["session"] = None
    stim_clust_df["adjacency"] = adjacency
    stim_clust_df["distance_threshold"] = distance_threshold
    if write_pkl:
        stim_clust_df.to_pickle(row[targout_col])
    if return_df:
        return stim_clust_df


def rank_clusters(
    targouts_mp, nvox_weight, concentration_weight, net_reference_correlation_weight
):
    targouts_sel = targouts_mp.query("nvox > 50 & net_reference_correlation < 0").copy()
    targouts_sel["flatmask"] = targouts_sel.apply(
        lambda row: idxs_to_flat(row.idxs, row.clnstim_mask), axis=1
    )
    targouts_sel["zoommask"] = targouts_sel.apply(
        lambda row: idxs_to_zoomed(row.idxs, row.clnstim_mask), axis=1
    )
    targouts_sel["nvox_rank"] = targouts_sel.groupby(
        ["run", "session", "subject"], dropna=False
    ).nvox.rank(ascending=False)
    targouts_sel["concentration_rank"] = targouts_sel.groupby(
        ["run", "session", "subject"], dropna=False
    ).concentration.rank(ascending=False)
    targouts_sel["nrc_rank"] = targouts_sel.groupby(
        ["run", "session", "subject"], dropna=False
    ).net_reference_correlation.rank(ascending=True)
    targouts_sel["weighted_nvox_rank"] = targouts_sel.nvox_rank * nvox_weight
    targouts_sel["weighted_concentration_rank"] = (
        targouts_sel.concentration_rank * concentration_weight
    )
    targouts_sel["weighted_nrc_rank"] = (
        targouts_sel.nrc_rank * net_reference_correlation_weight
    )
    targouts_sel["rank_product"] = (
        targouts_sel.weighted_nvox_rank
        * targouts_sel.weighted_concentration_rank
        * targouts_sel.weighted_nrc_rank
    )

    # sorting and using "first" method in rank ensure that there are no ties
    # Consider reordering this based on the weights
    metrics = ["net_reference_correlation", "concentration", "nvox"]
    targouts_sel = targouts_sel.sort_values(metrics).reset_index(drop=True)
    targouts_sel["overall_rank"] = targouts_sel.groupby(
        ["run", "session", "subject"], dropna=False
    ).rank_product.rank(ascending=True, method="first")
    return targouts_sel


def custom_metric(*args, **kwargs):
    if len(args) == 1:
        A = args[0]
        spr, p = stats.spearmanr(A.T)
        return 1 - np.abs(spr)
    elif len(args) == 2:
        A = args[0]
        B = args[1]
        spr, p = stats.spearmanr(A, B)
        return 1 - np.abs(spr)
    else:
        foo = args
        raise NotImplementedError(f"Number of args ({len(args)}) is not implemented.")


def get_clust_image(row):
    clnstim_mask = nl.image.load_img(row.clnstim_mask)
    clnstim_dat = clnstim_mask.get_fdata()

    rowstimmask_dat = idxs_to_mask(row.idxs, row.clnstim_mask)
    rowstimmask = nl.image.new_img_like(
        clnstim_mask, rowstimmask_dat, affine=clnstim_mask.affine, copy_header=True
    )
    # make sure you've got the correct cluster mask
    assert rowstimmask.get_fdata().sum() == row.nvox
    return rowstimmask


def get_vox_ref_corr(
    row,
    distance_threshold,
    adjacency,
    smoothing_fwhm,
    confound_selectors,
    metric,
    linkage,
    t_r,
    n_dummy,
):
    # just using clnstim_mask for now, at some point, it'd be better to use a grey matter mask
    refts_cln, ref_spr, ref_vcl, ref_clusters = clust_within_region(
        row.clnref_mask,
        row.bold_path,
        row.confounds,
        distance_threshold=distance_threshold,
        adjacency=adjacency,
        smoothing_fwhm=smoothing_fwhm,
        confound_selectors=confound_selectors,
        metric=custom_metric,
        linkage=linkage,
        t_r=t_r,
        n_dummy=n_dummy,
    )

    ref_clust_ts = np.array([cc.medts for cc in ref_clusters]).T
    ref_sizes = np.array([cc.nvox for cc in ref_clusters])

    sub_mask = nl.image.load_img(row.clnstim_mask)
    sub_img = nl.image.load_img(row.bold_path)
    cfds = select_confounds(row.confounds, confound_selectors)

    rawts = nl.masking.apply_mask(sub_img, sub_mask, smoothing_fwhm=smoothing_fwhm)[
        n_dummy:
    ]
    clnts = nl.signal.clean(
        rawts,
        confounds=cfds[n_dummy:],
        t_r=t_r,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        standardize="psc",
    )

    clust_corr, _ = stats.spearmanr(clnts, ref_clust_ts)
    try:
        clust_corr = clust_corr[: clnts.shape[1], clnts.shape[1] :]
        vox_ref_corr = (clust_corr * ref_sizes).sum(1)
    except IndexError:
        vox_ref_corr = (clust_corr * ref_sizes)[0]
    vox_ref_corr_img = nl.masking.unmask(vox_ref_corr, sub_mask)
    return vox_ref_corr_img


def get_com_in_mm(row):
    clnstim_mask = nl.image.load_img(row.clnstim_mask)
    try:
        vox_idxs = np.array([list(rr) + [1] for rr in row.idxs])
    except ValueError:
        vox_idxs = np.array([list(rr) + [1] for rr in row.idxs.values[0]])
    vox_locs = np.matmul(clnstim_mask.affine, vox_idxs.T).T[:, :3]
    return vox_locs.mean(0)
