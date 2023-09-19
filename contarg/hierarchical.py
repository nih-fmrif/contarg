from collections import namedtuple
from pathlib import Path
import warnings
from itertools import product
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import nilearn as nl
from nilearn import image, masking
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from scipy import stats
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from statsmodels.stats import weightstats

from kneed import KneeLocator
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex
import seaborn as sns
from .utils import select_confounds, idxs_to_flat, idxs_to_zoomed, idxs_to_mask, parse_bidsname, update_bidspath, find_bids_files, select_confounds, add_censor_columns, cross_spearman
from contarg import bootstrap as bs

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
    if confound_path is None:
        cfds=None
    else:
        cfds = select_confounds(confound_path, confound_selectors)[n_dummy:]

    rawts = nl.masking.apply_mask(sub_img, sub_mask, smoothing_fwhm=smoothing_fwhm)[
        n_dummy:
    ]
    clnts = nl.signal.clean(
        rawts,
        confounds=cfds,
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
    spr = np.nan_to_num(spr, posinf=1, neginf=-1)
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
    if len(targouts_sel) == 0:
        warnings.warn("No clusters survived 50 voxel threshold, removing minimum voxel size limit.")
        targouts_sel = targouts_mp.query("net_reference_correlation < 0").copy()
        if len(targouts_sel) == 0:
            raise ValueError("No clusters have a negative net_reference_correlation.")
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
        spr = np.nan_to_num(spr, posinf=1, neginf=-1)
        return 1 - np.abs(spr)
    elif len(args) == 2:
        A = args[0]
        B = args[1]
        spr, p = stats.spearmanr(A, B)
        spr = np.nan_to_num(spr, posinf=1, neginf=-1)
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
    if row.confounds is None:
        cfds=None
    else:
        cfds = select_confounds(row.confounds, confound_selectors)[n_dummy:]

    rawts = nl.masking.apply_mask(sub_img, sub_mask, smoothing_fwhm=smoothing_fwhm)[
        n_dummy:
    ]

    clnts = nl.signal.clean(
        rawts,
        confounds=cfds,
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


def prep_tedana_for_hierarchical(
        boldtd_path,
        fmriprep_dir,
        tedana_dir,
        out_dir,
        t_r,
        n_dummy,
        nthreads,
        drop_rundir=True,
        overwrite=False,
        max_outfrac=None,
        max_fd=None,
        frames_before=0,
        frames_after=0,
        minimum_segment_length=None,
        minimum_total_length=None,
        # if the following aren't defined, they'll be assumed to be in default locations
        scanner_to_t1w_path=None,
        t1w_to_MNI152NLin6Asym_path=None,
        confounds_path=None,
        boldmask_path=None,
        tdboldmask_path=None,
        boldref_t1_path=None,
        boldref_MNI152NLin6Asym_path=None,
):
    boldtd_path = Path(boldtd_path)
    fmriprep_dir = Path(fmriprep_dir)
    out_dir = Path(out_dir)

    if drop_rundir & ('run' in boldtd_path.parts[-2]):
        boldtd_path_for_building = boldtd_path.parent.parent / boldtd_path.parts[-1]
        ents = parse_bidsname(boldtd_path_for_building)
    else:
        boldtd_path_for_building = boldtd_path
        ents = parse_bidsname(boldtd_path)

    # find the root of the tedana dir
    tedana_root = []
    for pp in boldtd_path.parts[:-1]:
        if not pp.split("-")[-1] in ents.values():
            tedana_root.append(pp)
    tedana_root = Path(*tedana_root)

    # input paths
    if scanner_to_t1w_path is None:
        scanner_to_t1w_ents = {
            "suffix": "xfm",
            "extension": "txt",
            "from": "scanner",
            "to": "T1w",
            "mode": "image",
        }
        scanner_to_t1w_path = update_bidspath(
            boldtd_path_for_building,
            fmriprep_dir,
            scanner_to_t1w_ents,
            exclude=["desc"],
            exists=True,
        )

    if t1w_to_MNI152NLin6Asym_path is None:
        t1w_to_MNI152NLin6Asym_ents = {
            "suffix": "xfm",
            "extension": "h5",
            "from": "T1w",
            "to": "MNI152NLin6Asym",
            "mode": "image",
            "type": "anat",
        }
        t1w_to_MNI152NLin6Asym_path = update_bidspath(
            boldtd_path_for_building,
            fmriprep_dir,
            t1w_to_MNI152NLin6Asym_ents,
            exclude=["desc", "ses", "task", "acq", "run"],
        )
        if not t1w_to_MNI152NLin6Asym_path.exists():
            try:
                t1w_to_MNI152NLin6Asym_path = \
                find_bids_files(fmriprep_dir, **parse_bidsname(t1w_to_MNI152NLin6Asym_path))[0]
            except IndexError:
                t1w_to_MNI152NLin6Asym_path = update_bidspath(
                    boldtd_path_for_building,
                    fmriprep_dir,
                    t1w_to_MNI152NLin6Asym_ents,
                    exclude=["desc", "task", "acq", "run"],
                )
                t1w_to_MNI152NLin6Asym_path = \
                find_bids_files(fmriprep_dir, **parse_bidsname(t1w_to_MNI152NLin6Asym_path))[0]
            if not t1w_to_MNI152NLin6Asym_path.exists():
                raise FileNotFoundError(t1w_to_MNI152NLin6Asym_path.as_posix())

    if boldref_t1_path is None:
        boldref_t1_ents = dict(suffix="boldref", space="T1w")
        boldref_t1_path = update_bidspath(
            boldtd_path_for_building, fmriprep_dir, boldref_t1_ents, exclude=["desc"], exists=True
        )

    if boldref_MNI152NLin6Asym_path is None:
        boldref_MNI152NLin6Asym_ents = dict(
            suffix="boldref", space="MNI152NLin6Asym", res="2"
        )
        boldref_MNI152NLin6Asym_path = update_bidspath(
            boldtd_path_for_building,
            fmriprep_dir,
            boldref_MNI152NLin6Asym_ents,
            exclude=["desc"],
            exists=True,
        )
    # if there is a tedana mask, use that one for calculating censoring, otherwise, use fmriprep
    if tdboldmask_path is None:
        tdmask_ents = dict(
            desc="adaptiveGoodSignal",
            suffix="mask"
        )
        tdboldmask_path = update_bidspath(
            boldtd_path, tedana_dir, tdmask_ents, keep_rundir=True
        )
    if boldmask_path is None:
        boldmask_ents = dict(
            desc="brain",
            suffix="mask",
        )
        boldmask_path = update_bidspath(
            boldtd_path_for_building, fmriprep_dir, boldmask_ents, exists=True
        )

    if not tdboldmask_path.exists():
        tdboldmask_path = boldmask_path

    if confounds_path is None:
        confounds_ents = dict(desc="confounds", suffix="timeseries", extension="tsv")
        confounds_path = update_bidspath(
            boldtd_path_for_building, fmriprep_dir, confounds_ents, exists=True
        )

    # output paths
    cleaned_bold_ents = dict(desc=parse_bidsname(boldtd_path).get("desc", "") + "GSR")
    cleaned_bold_path = update_bidspath(boldtd_path_for_building, out_dir, cleaned_bold_ents)
    cleaned_bold_path.parent.mkdir(exist_ok=True, parents=True)

    T1wboldtd_ents = dict(space="T1w")
    T1wboldtd_path = update_bidspath(
        cleaned_bold_path, out_dir, T1wboldtd_ents
    )
    T1wboldtd_path.parent.mkdir(exist_ok=True, parents=True)

    MNIboldtd_ents = dict(space="MNI152NLin6Asym", res="2")
    MNIboldtd_path = update_bidspath(
        cleaned_bold_path, out_dir, MNIboldtd_ents
    )

    trunc_confounds_ents = {}
    trunc_confounds_path = update_bidspath(
        confounds_path, out_dir, trunc_confounds_ents
    )
    used_confounds_ents = {'desc': 'usedconfounds'}
    used_confounds_path = update_bidspath(
        confounds_path, out_dir, used_confounds_ents
    )
    if not overwrite and cleaned_bold_path.exists() and MNIboldtd_path.exists() and T1wboldtd_path.exists() and trunc_confounds_path.exists():
        return T1wboldtd_path, MNIboldtd_path, trunc_confounds_path

    cfds = pd.read_csv(confounds_path, sep="\t")
    # check if confounds are the correct length, and write out truncated ones if they're not
    boldtd_img = nl.image.load_img(boldtd_path)
    if len(cfds) != boldtd_img.shape[-1]:
        if len(cfds) != (boldtd_img.shape[-1] + n_dummy):
            raise ValueError(f"Confounds files (length = {len(cfds)}) is not n_dummy ({n_dummy}) TRs longer than boldtd "
                             f"({boldtd_img.shape[-1]}).")
        cfds = cfds.loc[n_dummy:, :].copy().reset_index(drop=True)

    #set n_dummy to 0, we've dealt with it
    n_dummy = 0
    #cfds.to_csv(trunc_confounds_path, index=None, sep="\t")

    # clean timeseries before resampling
    confound_selectors = ["-gs", "-motion", "-cosine", "-censor"]
    cfds = add_censor_columns(cfds, tdboldmask_path, boldtd_path, max_outfrac=max_outfrac, max_fd=max_fd,
                              frames_before=frames_before, frames_after=frames_after,
                              minimum_segment_length=minimum_segment_length, minimum_total_length=minimum_total_length,
                              n_dummy=n_dummy)
    cfds.to_csv(trunc_confounds_path, index=None, sep='\t')
    cfds = select_confounds(cfds, confound_selectors)
    cfds.to_csv(used_confounds_path, index=None, sep='\t')
    # clean bold
    cleaned = nl.image.clean_img(
        nl.image.load_img(boldtd_path),
        detrend=False,
        confounds=cfds,
        high_pass=0.01,
        low_pass=0.1,
        mask_img=nl.image.load_img(boldmask_path),
        t_r=t_r,
    )
    cleaned.to_filename(cleaned_bold_path)


    # transform from native space to T1w space
    at = ApplyTransforms(interpolation="LanczosWindowedSinc", float=True)
    at.inputs.num_threads = nthreads
    at.inputs.input_image = boldtd_path
    at.inputs.output_image = T1wboldtd_path.as_posix()
    at.inputs.reference_image = boldref_t1_path
    at.inputs.input_image_type = 3
    at.inputs.transforms = [scanner_to_t1w_path]
    _ = at.run()

    # transform from native space to MNI152NLin6Asym space
    at = ApplyTransforms(interpolation="LanczosWindowedSinc", float=True)
    at.inputs.num_threads = nthreads
    at.inputs.input_image = boldtd_path
    at.inputs.output_image = MNIboldtd_path.as_posix()
    at.inputs.reference_image = boldref_MNI152NLin6Asym_path
    at.inputs.input_image_type = 3
    at.inputs.transforms = [t1w_to_MNI152NLin6Asym_path, scanner_to_t1w_path]
    _ = at.run()

    return T1wboldtd_path, MNIboldtd_path, trunc_confounds_path


def block_bootstrap(ts, nsamples, block_length=30, seed=None):
    """
    Perform block bootstrapping with overlapping blocks on the second axis of a 2d-array.

    Parameters
    ==========
    ts : array
        Timeseries with time on the second axis.
    nsamples : int
        How many bootstrap samples to create.
    block_length : int, default = 10
        How long should each block be. Some blocks will be longer to account for the remainder.
    seed : int
        Seed for the rng

    Returns
    =======
    samples : array
        Bootstrapped samples with samples along a new first dimension
    """
    npoints = ts.shape[-1]
    nblocks = int(np.floor(npoints / block_length))
    remainder = npoints - (nblocks * block_length)

    rng = np.random.default_rng(seed)

    # overly verbose way to figure out how many extra points need to be added to each block
    block_additions = np.zeros(nblocks)
    bai = 0
    while remainder - block_additions.sum() > 0:
        block_additions[bai] += 1
        bai += 1
        if bai == len(block_additions):
            bai = 0
    block_lengths = block_length + block_additions
    block_lengths = np.matmul(np.ones((nsamples, 1)), block_lengths.reshape(1, -1)).astype(int)

    # have to use nested for loops so I can deal with variable block lengths
    samples = []
    for bls in block_lengths:
        sample = []
        for bl in bls:
            bix = rng.choice(range(npoints - bl))
            assert (bix + bl) < npoints
            segment = ts[:, bix:bix + bl]
            sample.append(segment)
        sample = np.hstack(sample)
        if sample.shape != ts.shape:
            raise ValueError("something broke")
        samples.append(sample)
    samples = np.array(samples)
    return np.array(samples)

def run_cluster(ts, dt, connectivity, metric='cosine', linkage='complete'):
    agglomerative = AgglomerativeClustering(n_clusters=None, metric=metric,
                                        connectivity=connectivity, linkage=linkage, distance_threshold=dt)
    run_labels = agglomerative.fit_predict(ts)
    return run_labels


def build_coocurence(labels):
    coocurence = np.zeros((len(labels), len(labels)))
    for label in np.unique(labels):
        inclust = (labels == label).nonzero()[0]
        row_cords, col_cords = list(zip(*product(inclust, inclust)))
        row_cords = np.array(row_cords)
        col_cords = np.array(col_cords)
        coocurence[row_cords, col_cords] += 1
    return coocurence

def find_cluster_threshold(ts, connectivity=None,
                           course_threshes=10, fine_threshes=15, minimum_fine_nclust=3, nbootstraps=100,
                           block_length=60, seed=None, nthreads=1, metric="cosine", linkage="complete"):
    """
    Find clustering distance threshold for passed time series.

    Parameters
    ----------
    ts : ndarray of shape (n_features, n_timepoints)
        Time series data to be clustered.

    connectivity : ndarray of shape (n_features, n_features), default=None
        Connectivity matrix.

    course_threshes : int, default=10
        Number of thresholds used in the course search.

    fine_threshes : int, default=20
        Number of thresholds used in the fine search.

    nbootstraps : int, default=100
        Number of bootstraps.

    block_length : int, default=60
        Length of each block used in block bootstrap.

    seed : int, default=None
        Seed for the random number generator.

    nthreads : int, default=1
        Number of threads to use.

    metric : str, defalut="cosine"
        Metric to use for clustering

    linkage : str, default="average"
        Linkage for clustering
    Returns
    =======
    thresh : float
        Optimal distance threshold

    entropy : array(float)
        Entropy of elements at the optimal distance threshold

    cluster_stats : DataFrame
        cluster statistics for all of the bootstraps
    """
    bs_ts = block_bootstrap(ts, nsamples=nbootstraps, block_length=block_length, seed=seed)

    dts = []
    all_ks = []
    all_labels = []

    print("Running course search of threshold space.")
    for dti, dt in enumerate(np.linspace(0.01, 0.5, course_threshes)):
        jobs = []
        for ts in bs_ts:
            jobs.append(delayed(run_cluster)(ts, dt, connectivity, metric=metric, linkage=linkage))
        print(f"Running clusters for threshold {dt}. {dti + 1} of {course_threshes}")
        bs_labels = Parallel(n_jobs=20)(jobs)
        ks = [len(np.unique(bbl)) for bbl in bs_labels]
        all_labels.append(bs_labels)
        all_ks.append(ks)
        dts.append(dt)
    # all_vs = np.array(all_vs)
    all_labels = np.array(all_labels)
    all_ks = np.array(all_ks)
    dts = np.array(dts)
    mean_ks = all_ks.mean(1)
    top_dt = dts[mean_ks > minimum_fine_nclust].max()
    bottom_dt = dts[mean_ks < (len(ts)/2)].min()

    print("Running fine search of threshold space.")
    fine_dts = []
    fine_ks = []
    fine_labels = []
    for dti, dt in enumerate(np.linspace(bottom_dt, top_dt, fine_threshes)):
        jobs = []
        for ts in bs_ts:
            jobs.append(delayed(run_cluster)(ts, dt, connectivity))
        print(f"Running clusters for threshold {dt}. {dti + 1} of {fine_threshes}", flush=True)
        bs_labels = Parallel(n_jobs=nthreads)(jobs)
        ks = [len(np.unique(bbl)) for bbl in bs_labels]
        fine_labels.append(bs_labels)
        fine_ks.append(ks)
        fine_dts.append(dt)
    # all_vs = np.array(all_vs)
    fine_labels = np.array(fine_labels)
    fine_ks = np.array(fine_ks)

    jobs = []
    for bs_labels in fine_labels:
        for ix, ibs in enumerate(bs_labels):
            for jx, jbs in enumerate(bs_labels[ix + 1:]):
                jobs.append(delayed(adjusted_mutual_info_score)(ibs, jbs))
    print(f"Calculating Adjusted Mutual Information for {len(jobs)} cluster pairs from fine thresholds", flush=True)
    fine_vs = Parallel(n_jobs=nthreads, verbose=7)(jobs)
    fine_vs = np.array(fine_vs)
    fine_vs = fine_vs.reshape(len(fine_labels), -1)

    mean_ks = fine_ks.mean(1)
    mean_vs = fine_vs.mean(1)

    kl = KneeLocator(mean_ks, mean_vs, curve="concave", direction='increasing', interp_method='polynomial',
                     online=True, polynomial_degree=3)
    kl.plot_knee(xlabel='Mean Number of Clusters', ylabel='Mean Adjusted Mutual Information')
    thresh = np.array(fine_dts)[mean_ks == kl.knee][0]
    print(f"Threshold = {thresh} at AMI = {kl.knee_y} and mean number of clusters = {kl.knee}", flush=True)

    jobs = []
    for labels in fine_labels[mean_ks == kl.knee][0]:
        jobs.append(delayed(build_coocurence)(labels))
    print("Getting coocurence matrix and calculating entropy")
    coocurence = Parallel(n_jobs=nthreads)(jobs)
    coocurence = np.array(coocurence)

    p1 = coocurence.sum(0) / coocurence.shape[0]
    p0 = 1 - p1

    # taking the mean here is equivalent to dividing by the maximum possible entropy
    entropy = stats.entropy(np.dstack([p1, p0]), base=2, axis=-1).mean(1)

    cluster_stats = []
    for dt, bs_labels in zip(fine_dts, fine_labels):
        for bsix, label in enumerate(bs_labels):
            cluster_ids, cluster_counts = np.unique(label, return_counts=True)
            row = dict(
                dt=dt,
                bsix=bsix,
                n_clusters=len(cluster_ids),
                cluster_size_mean=cluster_counts.mean(),
                cluster_size_std=cluster_counts.std(),
                cluster_size_min=cluster_counts.min(),
                cluster_size_max=cluster_counts.max(),
                cluster_size_small=(cluster_counts < 10).sum()
            )
            cluster_stats.append(row)
    cluster_stats = pd.DataFrame(cluster_stats)

    return thresh, entropy, cluster_stats


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def cluster_and_plot(ts, thresh, verts, coords, entropy, entropy_thresh=None, connectivity=None, metric="cosine",
                     linkage="complete", min_verts=None, plot=True):
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
    entropy : ndarray
        A numpy array of shape (n_vertices,) representing the entropy values for each vertex.
    entropy_thresh : float, optional
        The threshold value for entropy. If None, drop 10% of vertices with highest entropy.
    connectivity : ndarray, optional
        A numpy array of shape (n_vertices, n_vertices) representing the connectivity matrix. Default is None.
    metric : str, optional
        The metric to use for distance calculation. Default is "cosine".
    linkage : str, optional
        The linkage criterion to use. Default is "average".
    min_verts : int, optional
        The minimum number of vertices for a cluster to be considered. Default is None.  If none, 2% of number of vertices is used.
    plot : bool, optional
        Whether to generate a plot of the clustering results. Default is True.

    Returns
    -------
    labels : ndarray
        A numpy array of shape (n_vertices,) representing the cluster labels for each time series.
    good_idx : ndarray
        A boolean numpy array of shape (n_vertices,) indicating which vertices pass the entropy and minimum cluster size threholds.
    """

    if entropy_thresh is None:
        entropy_thresh = np.percentile(entropy, 90)
    good_entropy = entropy < entropy_thresh
    good_coords = coords[good_entropy]
    good_ts = ts[good_entropy]
    good_verts = verts[good_entropy]
    agglomerative = AgglomerativeClustering(n_clusters=None, metric=metric,
                                            connectivity=connectivity[good_entropy][:, good_entropy], linkage=linkage,
                                            distance_threshold=thresh)
    agglomerative.set_params(compute_full_tree=True, compute_distances=True)
    labels = agglomerative.fit_predict(good_ts)

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
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 5))
        ax = axes[0]
        plot_dendrogram(agglomerative, color_threshold=thresh, ax=axes[0])
        xlim = ax.get_xlim()
        ax.hlines(thresh, xlim[0], xlim[1], linestyle='dashed')

        ax = axes[1]
        ax.scatter(good_coords[:, 1][good_idx], good_coords[:, 2][good_idx], c=clust_colors(labels_to_plot))
        ax.scatter(coords[:, 1][~good_entropy], coords[:, 2][~good_entropy], c="#5d5d5d", marker='X')
        ax.scatter(good_coords[:, 1][~good_idx], good_coords[:, 2][~good_idx], c="#5d5d5d", marker='v')
        ax.set_aspect('equal')
        ax.set_xlabel('y')
        ax.set_ylabel('z')
    return good_labels, good_ts[good_idx], good_verts[good_idx], good_coords[good_idx], label_colors

SurfaceCluster = namedtuple("SurfaceCluster", "id idxs nvert concentration vert_to_rep repts medts")


def cluster(ts, thresh, connectivity=None, metric="cosine",
            linkage="complete", min_verts=None):
    agglomerative = AgglomerativeClustering(n_clusters=None, metric=metric,
                                            connectivity=connectivity,
                                            linkage=linkage,
                                            distance_threshold=thresh)
    labels = agglomerative.fit_predict(ts)

    return labels


SurfaceCluster = namedtuple("SurfaceCluster", "id idxs nvert mean_vert_to_rep std_vert_to_rep repts")


def get_surface_cluster_stats(labels, ts, verts, coords):
    cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
    clusters = []
    for nvox, cid in zip(cluster_counts, cluster_ids):
        idxs = np.nonzero(labels == cid)[0]
        clust_tses = ts[idxs].T
        if nvox == 1:
            median_ts = clust_tses.squeeze()
            rep_ts = clust_tses.squeeze()
            clust_to_median = np.array([1])
            vert_idxs = verts[idxs]
            vert_locs = coords[idxs]
            mean_vert_to_rep = 1
            std_vert_to_rep = 0
        else:
            median_ts = np.percentile(clust_tses, 50, axis=1)
            # calculate spearmanr between median ts and each clust_ts
            clust_to_median, _, _ = cross_spearman(median_ts.reshape(1, -1), clust_tses.T, corr_only=True)
            clust_to_median = clust_to_median.squeeze()
            rep_ts = clust_tses[:, clust_to_median == clust_to_median.max()].squeeze()
            # if there's a tie for the highest correlation with the median time series, just take the first one
            if len(rep_ts.shape) > 1:
                rep_ts = rep_ts[:, 0].squeeze()
            mean_vert_to_rep = clust_to_median.mean()
            std_vert_to_rep = clust_to_median.std()
            # get a list of voxel indicies of the cluster
            vert_idxs = verts[idxs]
            vert_locs = coords[idxs]
            assert len(vert_locs) == nvox

        cluster = SurfaceCluster(cid, vert_idxs, nvox, mean_vert_to_rep, std_vert_to_rep, rep_ts)
        clusters.append(cluster)
    return pd.DataFrame(clusters)


def get_normed_neg_minus_pos(pears_rs, mcsig, cluster_sizes):
    """
    Calculate the normalized difference between the number of negative and positive correlations for each cluster.

    Parameters
    ----------
    pears_rs : np.ndarray
        Pearson correlation coefficients between time series pairs.
    mcsig : np.ndarray
        Boolean array indicating which correlation coefficients are significant based on multiple comparisons correction.
    cluster_sizes : np.ndarray
        Array containing the size of each cluster.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The number of negative correlations, the number of positive correlations, and the normalized difference between the two.

    """
    neg_corr = (mcsig & (pears_rs < 0))
    num_neg_corr = (neg_corr * cluster_sizes).sum(1)

    pos_corr = (mcsig & (pears_rs > 0))
    num_pos_corr = (pos_corr * cluster_sizes).sum(1)

    neg_minus_pos = num_neg_corr - num_pos_corr
    norm_neg_minus_pos = neg_minus_pos / cluster_sizes.sum()
    return num_neg_corr, num_pos_corr, norm_neg_minus_pos


def bootstrap_clustering(bsi, bts, dstim_roi, dstim_thresh, rref_roi, rref_thresh, lref_roi, lref_thresh,
                         other_roi=None, ref_clust_min_size=3, pairwise_sig_thresh=0.05):
    # Get timeseries
    dstim_ts = bts[:len(dstim_roi.ts), :]
    rref_ts = bts[len(dstim_roi.ts):len(dstim_roi.ts) + len(rref_roi.ts), :]
    lref_ts = bts[len(dstim_roi.ts) + len(rref_roi.ts):len(dstim_roi.ts) + len(rref_roi.ts) + len(lref_roi.ts), :]
    other_ts = bts[len(dstim_roi.ts) + len(rref_roi.ts) + len(lref_roi.ts):, :]
    ref_ts = np.vstack([rref_ts, lref_ts])

    # cluster the rois using the threshes
    dstim_labels = cluster(dstim_ts, dstim_thresh, dstim_roi.connectivity)
    rref_labels = cluster(rref_ts, rref_thresh, rref_roi.connectivity)
    lref_labels = cluster(lref_ts, lref_thresh, lref_roi.connectivity)

    # get cluster level stats
    dstim_clusters = get_surface_cluster_stats(dstim_labels, dstim_ts, dstim_roi.idxs, dstim_roi.coords)
    rref_clusters = get_surface_cluster_stats(rref_labels, rref_ts, rref_roi.idxs, rref_roi.coords)
    lref_clusters = get_surface_cluster_stats(lref_labels, lref_ts, lref_roi.idxs, lref_roi.coords)
    rref_clusters['hemisphere'] = 'right'
    lref_clusters['hemisphere'] = 'left'
    ref_clusters = pd.concat([rref_clusters, lref_clusters])
    ref_clusters = ref_clusters.query('nvert >= @ref_clust_min_size')

    dstim_refrep_corr, _, _ = cross_spearman(np.array(list(dstim_clusters.repts.values)),
                                             np.array(list(ref_clusters.repts)), corr_only=True)
    dstim_clusters['refrep_corr'] = np.tanh(
        (np.arctanh(dstim_refrep_corr) * ref_clusters.nvert.values).sum(1) / ref_clusters.nvert.values.sum())

    # calculate dstim vertexwise stats
    dstim_verts = pd.DataFrame(index=dstim_roi.idxs)
    dstim_verts['bootstrap'] = bsi
    dstim_verts['idx'] = dstim_roi.idxs
    dstim_rep_corr, _, dstim_rep_sig = cross_spearman(dstim_roi.ts, np.array(list(ref_clusters.repts.values)),
                                                      alpha=pairwise_sig_thresh, method='fdr_bh')
    num_neg_corr, num_pos_corr, norm_neg_minus_pos = get_normed_neg_minus_pos(dstim_rep_corr, dstim_rep_sig,
                                                                              ref_clusters.nvert.values)
    dstim_rep_zs = np.arctanh(dstim_rep_corr)
    dstim_rep_ds = weightstats.DescrStatsW(dstim_rep_zs.T, weights=ref_clusters.nvert.values)
    dstim_rep_mean = np.tanh(dstim_rep_ds.mean)
    dstim_rep_std = np.tanh(dstim_rep_ds.std)
    dstim_rep_t, _ = dstim_rep_ds.ztest_mean()

    # add mean correlation with ref without ref clusters
    dstim_ref_corr, _, _ = cross_spearman(dstim_roi.ts, ref_ts, corr_only=True)
    dstim_ref_zs = np.arctanh(dstim_ref_corr)
    dstim_ref_ds = weightstats.DescrStatsW(dstim_ref_zs.T)
    dstim_ref_mean = np.tanh(dstim_ref_ds.mean)
    dstim_ref_std = np.tanh(dstim_ref_ds.std)
    dstim_ref_t, _ = dstim_ref_ds.ztest_mean()

    dstim_verts['norm_neg_minus_pos'] = norm_neg_minus_pos
    dstim_verts['cluster'] = dstim_labels
    dstim_verts['refrep_corr'] = dstim_rep_mean
    dstim_verts['refrep_std'] = dstim_rep_std
    dstim_verts['refrep_t'] = dstim_rep_t
    dstim_verts['ref_corr'] = dstim_ref_mean
    dstim_verts['ref_std'] = dstim_ref_std
    dstim_verts['ref_t'] = dstim_ref_t
    to_merge = dstim_clusters.loc[:, ['id', 'nvert', 'mean_vert_to_rep', 'std_vert_to_rep', 'refrep_corr']]
    dstim_verts = dstim_verts.merge(to_merge, left_on='cluster', right_on='id', how='left', suffixes=['', '_cluster'])

    # get stats for vertices outside dstim_roi
    other_verts = pd.DataFrame(index=other_roi.idxs)
    other_verts['bootstrap'] = bsi
    other_verts['idx'] = other_roi.idxs
    other_rep_corr, _, other_rep_sig = cross_spearman(other_roi.ts, np.array(list(ref_clusters.repts.values)),
                                                      alpha=pairwise_sig_thresh, method='fdr_bh')
    num_neg_corr, num_pos_corr, norm_neg_minus_pos = get_normed_neg_minus_pos(other_rep_corr, other_rep_sig,
                                                                              ref_clusters.nvert.values)
    other_rep_zs = np.arctanh(other_rep_corr)
    other_rep_ds = weightstats.DescrStatsW(other_rep_zs.T, weights=ref_clusters.nvert.values)
    other_rep_mean = np.tanh(other_rep_ds.mean)
    other_rep_std = np.tanh(other_rep_ds.std)
    other_rep_t, _ = other_rep_ds.ztest_mean()

    other_verts['norm_neg_minus_pos'] = norm_neg_minus_pos
    other_verts['refrep_corr'] = other_rep_mean
    other_verts['refrep_std'] = other_rep_std
    other_verts['refrep_t'] = other_rep_t

    # add mean correlation with ref without ref clusters
    other_ref_corr, _, _ = cross_spearman(other_roi.ts, ref_ts, corr_only=True)
    other_ref_zs = np.arctanh(other_ref_corr)
    other_ref_ds = weightstats.DescrStatsW(other_ref_zs.T)
    other_ref_mean = np.tanh(other_ref_ds.mean)
    other_ref_std = np.tanh(other_ref_ds.std)
    other_ref_t, _ = other_ref_ds.ztest_mean()

    other_verts['ref_corr'] = other_ref_mean
    other_verts['ref_std'] = other_ref_std
    other_verts['ref_t'] = other_ref_t

    dstim_verts = pd.concat([other_verts, dstim_verts])
    return dstim_verts


def get_stim_stats_with_uncertainty(pos_ix, angle_ix, coord_ix, uncert_df, merged_verts, magne_min_percentile=None,
                                    nboots=100):
    uncert_series = uncert_df.loc[:, pos_ix].rename(index='weight')
    uncert_series = uncert_series[uncert_series > 1e-6]
    uncert_df = merged_verts.join(uncert_series, how='inner', )
    if magne_min_percentile is not None:
        # weight based on magne percentiles so that the minimum gets about 1/10 the weight of the max
        # this is equivalent to taking 10 values from the linspace between the max and min and averaging,
        # but way faster
        magne_weights = ((uncert_df.percentile - magne_min_percentile) / (100 - magne_min_percentile))
        magne_weights = (magne_weights + 0.11) / (1.11)
        uncert_df['weight'] = uncert_df.weight * magne_weights
    weights = uncert_df.weight.values.reshape(-1, nboots)[:, 0]
    # rescale_weights to keep weighted stats from breaking
    weights = weights / weights.max()
    bs_refrep_t_mean, bs_refrep_t_bst, bs_refrep_t_bsp = get_weighted_stats(uncert_df.refrep_t.values.reshape(-1, nboots),
                                                                            weights)
    bs_refrep_corr_mean, bs_refrep_corr_bst, bs_refrep_corr_bsp = get_weighted_stats(
        uncert_df.refrep_corr.values.reshape(-1, nboots), weights)
    bs_ref_corr_mean, bs_ref_corr_bst, bs_ref_corr_bsp = get_weighted_stats(uncert_df.ref_corr.values.reshape(-1, nboots),
                                                                            weights)

    clust_corr = uncert_df.refrep_corr_cluster.values.reshape(-1, nboots)
    clust_corr_mask = pd.isnull(clust_corr).sum(1) == 0
    bs_clustrefrep_t_mean, bs_clustrefrep_t_bst, bs_clustrefrep_t_bsp = get_weighted_stats(clust_corr[clust_corr_mask],
                                                                                           weights[clust_corr_mask])
    tmp = uncert_df.groupby(['pos_ix_col', 'bootstrap', 'weight']).cluster.nunique().reset_index()
    tmp['weighted_nclust'] = tmp.cluster * tmp.weight
    nclust = tmp.weighted_nclust.sum() / tmp.weight.sum()

    res = dict(
        pos_ix=pos_ix,
        angle_ix=angle_ix,
        coord_ix=coord_ix,
        bs_refrep_t_mean=bs_refrep_t_mean,
        bs_refrep_t_bst=bs_refrep_t_bst,
        bs_refrep_t_bsp=bs_refrep_t_bsp,
        bs_refrep_corr_mean=bs_refrep_corr_mean,
        bs_refrep_corr_bst=bs_refrep_corr_bst,
        bs_refrep_corr_bsp=bs_refrep_corr_bsp,
        bs_ref_corr_mean=bs_ref_corr_mean,
        bs_ref_corr_bst=bs_ref_corr_bst,
        bs_ref_corr_bsp=bs_ref_corr_bsp,
        bs_clustrefrep_t_mean=bs_clustrefrep_t_mean,
        bs_clustrefrep_t_bst=bs_clustrefrep_t_bst,
        bs_clustrefrep_t_bsp=bs_clustrefrep_t_bsp,
        nclust=nclust
    )
    return res


def get_weighted_stats(stats, weights):
    d_z = weightstats.DescrStatsW(stats, weights=weights)
    bs_refrep_t, _ = d_z.ztest_mean()
    bs_refrep_t_mean = bs_refrep_t.mean()
    bs_refrep_t_bst = bs_refrep_t_mean / bs_refrep_t.std()
    bs_refrep_t_bsp = bs.get_bs_p(bs_refrep_t, side="upper")
    return bs_refrep_t_mean, bs_refrep_t_bst, bs_refrep_t_bsp