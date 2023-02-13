import collections.abc as collections
import pandas as pd
import numpy as np
import nilearn as nl
import six
from nilearn import image, masking
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)  # pip
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import (
    label,
    generate_binary_structure,
)

# code for transforming things to subject space
def make_path(
    ents,
    updates,
    pattern,
    derivatives_dir,
    build_path,
    check_exist=True,
    check_parent=True,
    mkdir=False,
    make_parent=False,
):
    mp_ents = ents.copy()
    mp_dir = derivatives_dir
    mp_ents.update(updates)
    mp_file = mp_dir / build_path(
        mp_ents, pattern, validate=False, absolute_paths=False
    )
    if check_exist and not mp_file.exists():
        raise FileNotFoundError(mp_file.as_posix())
    elif check_parent and not mp_file.parent.exists():
        raise FileNotFoundError(mp_file.parent.as_posix())
    if mkdir:
        mp_file.mkdir(parents=True, exist_ok=True)
    elif make_parent:
        mp_file.parent.mkdir(parents=True, exist_ok=True)
    return mp_file


def transform_mask_to_t1w(
    row, inmask_col=None, inmask=None, outmask_col="boldmask_path"
):
    if inmask_col is None and inmask is None:
        raise ValueError("one of mask_col or mask must be defined")
    elif inmask_col is not None:
        inmask = row[inmask_col]

    at = ApplyTransforms()
    at.inputs.input_image = inmask
    at.inputs.interpolation = "NearestNeighbor"
    at.inputs.reference_image = row.boldref
    at.inputs.transforms = [row.mnitoT1w]
    at.inputs.output_image = row[outmask_col].as_posix()
    at.inputs.float = True
    _ = at.run()


def clean_mask(
    sub_mask, brain_mask, max_drop_frac=None, clean_mask_path=None, error="raise"
):
    """
    Drop all but the largest of the connected components.
    """
    # mask by subject brain mask
    sub_mask_brain_dat = sub_mask.get_fdata()
    sub_mask_brain_dat[brain_mask.get_fdata() == 0] = 0
    sub_mask_brain = nl.image.new_img_like(
        sub_mask, sub_mask_brain_dat, affine=sub_mask.affine, copy_header=True
    )
    adjacency = grid_to_graph(
        sub_mask_brain.shape[0],
        sub_mask_brain.shape[1],
        sub_mask_brain.shape[2],
        mask=sub_mask_brain_dat,
        return_as=np.ndarray,
    )
    n_connected_components, labels = connected_components(adjacency)
    # add one to labels to make creating a mask easier
    labels = labels + 1
    label_ids, label_counts = np.unique(labels, return_counts=True)
    id_to_keep = label_ids[label_counts == label_counts.max()]

    drop_frac = 1 - (label_counts.max() / len(labels))
    if max_drop_frac is not None:
        if drop_frac > max_drop_frac:
            if error == "raise":
                raise ValueError(
                    f"Keeping only the largest connected component drops\
                {drop_frac:0.3f} of voxels, the maximum dropped fraction is f{max_drop_frac}"
                )
            else:
                print(
                    f"Keeping only the largest connected component drops\
                {drop_frac:0.3f} of voxels, the maximum dropped fraction is f{max_drop_frac}, skipping"
                )
                return None

    print(
        f"Dropping {drop_frac:0.3g} of voxels that are disconnected from the largest component."
    )

    cleaned_mask = nl.masking.unmask(labels, sub_mask_brain)
    cleaned_mask_dat = cleaned_mask.get_fdata()
    cleaned_mask_dat[cleaned_mask_dat != id_to_keep] = 0
    cleaned_mask_dat[cleaned_mask_dat == id_to_keep] = 1
    assert cleaned_mask_dat.sum() > 0
    cleaned_mask = nl.image.new_img_like(
        cleaned_mask,
        data=cleaned_mask_dat,
        affine=cleaned_mask.affine,
        copy_header=True,
    )
    # make sure there's only a single component
    adjacency = grid_to_graph(
        cleaned_mask.shape[0],
        cleaned_mask.shape[1],
        cleaned_mask.shape[2],
        mask=cleaned_mask.get_fdata(),
        return_as=np.ndarray,
    )
    n_connected_components, labels = connected_components(adjacency)
    assert n_connected_components == 1

    if clean_mask_path is not None:
        cleaned_mask.to_filename(clean_mask_path)
    return cleaned_mask


def select_confounds(cfds_path, cfds_sel):
    cfd = pd.read_csv(cfds_path, sep="\t")
    cols = []
    # cfd = cfd.drop('censored', axis=1)
    if "-censor" in cfds_sel:
        cols.extend(
            [cc for cc in cfd.columns if ("censor" in cc) and (cc != "censored")]
        )
    if "-strictcensor" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if "censfdplus1_" in cc])
        cols.extend([cc for cc in cfd.columns if "strictfd_" in cc])
        cols.extend([cc for cc in cfd.columns if "strictfdplus1_" in cc])
    if "-cosine" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if "cosine" in cc])
    if "-aroma" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if "aroma" in cc])
    if "-motion" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if ("trans" in cc) or ("rot" in cc)])
    if "-physio" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if "phys" in cc])
    if "-gs" in cfds_sel:
        cols.extend(["global_signal"])
    if "-gs+" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if "global_signal" in cc])
    if "-dummy" in cfds_sel:
        cols.extend([cc for cc in cfd.columns if "non_steady_state_outlier" in cc])
    cfds = cfd.loc[:, cols].copy()
    return cfds


def idxs_to_mask(idxs, mask_path):
    mask = nl.image.load_img(mask_path)
    mask_dat = mask.get_fdata()
    null_mask = np.zeros(mask.shape)
    idxs_split = np.array(list((zip(*idxs))))
    null_mask[idxs_split[0], idxs_split[1], idxs_split[2]] = 1
    return null_mask


def idxs_to_zoomed(idxs, mask_path):
    mask = nl.image.load_img(mask_path)
    mask_dat = mask.get_fdata()
    null_mask = np.zeros(mask.shape)
    idxs_split = np.array(list((zip(*idxs))))
    null_mask[idxs_split[0], idxs_split[1], idxs_split[2]] = 1
    maskxs, maskys, maskzs = np.where(mask_dat != 0)
    minx = maskxs.min()
    miny = maskys.min()
    minz = maskzs.min()
    maxx = maskxs.max() + 1
    maxy = maskys.max() + 1
    maxz = maskzs.max() + 1
    clust = null_mask[minx:maxx, miny:maxy, minz:maxz] != 0
    assert (clust != 0).sum() == len(idxs)

    return clust


def idxs_to_flat(idxs, mask_path):
    mask = nl.image.load_img(mask_path)
    mask_dat = mask.get_fdata()
    null_mask = np.zeros(mask.shape)
    idxs_split = np.array(list((zip(*idxs))))
    null_mask[idxs_split[0], idxs_split[1], idxs_split[2]] = 1
    assert len(idxs) == null_mask.sum()
    flat_clust = null_mask[mask_dat != 0]
    assert len(flat_clust) == (mask_dat != 0).sum()
    return flat_clust


def iterable(arg):
    # from https://stackoverflow.com/a/44328500
    return isinstance(arg, collections.Iterable) and not isinstance(
        arg, six.string_types
    )


def cluster(stat_img_path, out_path=None, stim_roi_path=None, percentile=10, sign='negative', connectivity='NN3'):
    """
    Cluster a statmap for values with the sign of your choice within a particular ROI after threholding based on a percentile of values with that sign.
    Outputs a mask for the largest cluster.

    Parameters
    ----------
    stat_img_path : string or path object
        Path to statistical image on which to cluster.
    out_path : string or path object
        Path to write output to, if None, no output is written
    stim_roi_path : string or path object
        Path to roi within which to cluster, if None, use all non-zero voxels in stat_img
    percentile : float
        All values more extreme than percentile will be kept for clustering 
    sign : str ["negative", "positive"]
        Sign of values to operate on
    connectivit : str ["NN1", "faces", "NN2", "edges", "NN3", "vertices"]
        Deffinition of connectivity to use for clustering accepts either description (faces, edges, verticies) or AFNI label (NN1, NN2, NN3).

    Returns
    -------
    biggest_clsut_img : niiimage
        Mask for the biggest cluster

    Code inspired by https://github.com/nilearn/nilearn/blob/b7e5efdd37a6b4cc276763fc5f2a2cde81f7af73/nilearn/image/image.py#L790
    """
    if ((connectivity == "NN1") | (connectivity == "faces")):
        bin_struct = generate_binary_structure(3, 1)
    elif ((connectivity == "NN2") | (connectivity == "edges")):
        bin_struct = generate_binary_structure(3, 2)
    elif ((connectivity == "NN3") | (connectivity == "vertices")):
        bin_struct = generate_binary_structure(3, 3)
    else:
        raise ValueError(f"You specified connectivity={connectivity}, but the only supported terms are:"
                         "NN1 or faces"
                         "NN2 or edges"
                         "NN3 or vertices"
                         )
    stat_img = nl.image.load_img(stat_img_path)
    if stim_roi_path is None:
        stimroi_dat = stat_img.get_fdata() != 0
        stimroi_img = nl.image.new_img_like(stat_img)
    else:
        stimroi = nl.image.load_img(stim_roi_path)
    masked_stat = nl.masking.apply_mask(stat_img, stimroi)
    threshed = masked_stat.copy()

    if sign == 'negative':
        sign_masked_stat = masked_stat[masked_stat < 0]
        threshold = np.percentile(sign_masked_stat, percentile)
        threshed[(masked_stat > 0) | (masked_stat > threshold)] = 0
        threshed[threshed != 0] = 1
    elif sign == 'positive':
        sign_masked_stat = masked_stat[masked_stat > 0]
        threshold = np.percentile(sign_masked_stat, percentile)
        threshed[(masked_stat < 0) | (masked_stat < threshold)] = 0
        threshed[threshed != 0] = 1

    threshed_img = nl.masking.unmask(threshed, stimroi)
    threshed_dat = threshed_img.get_fdata().squeeze()

    label_map = label(threshed_dat, bin_struct)[0]
    clust_ids = sorted(list(np.unique(label_map)[1:]))
    clust_counts = [(label_map == c_val).sum() for c_val in clust_ids]

    biggest_clust_id = clust_ids[np.argsort(clust_counts)[-1]]
    biggest_clust_dat = (label_map == biggest_clust_id)
    biggest_clust_img = nl.image.new_img_like(stat_img, biggest_clust_dat, affine=stat_img.affine, copy_header=True)
    if out_path is not None:
        biggest_clust_img.to_filename(out_path)
    return biggest_clust_img