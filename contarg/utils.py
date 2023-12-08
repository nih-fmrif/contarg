import os
import collections.abc as collections
from pathlib import Path
import pandas as pd
import numpy as np
import nilearn as nl
import six
from nilearn import image, masking
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)  # pip
from niworkflows.interfaces.cifti import _prepare_cifti, CIFTI_STRUCT_WITH_LABELS
from nipype.interfaces.freesurfer import MRIConvert
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import (
    label,
    generate_binary_structure,
)
import nibabel as nb
import networkx as nx
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pkg_resources import resource_filename
import templateflow
from nibabel import cifti2 as ci



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
    row=None, inmask_col=None, inmask=None, outmask_col="boldmask_path",
    reference=None, transforms=None, output_image=None
):
    if inmask_col is None and inmask is None:
        raise ValueError("one of mask_col or mask must be defined")
    elif inmask_col and row is not None:
        inmask = row[inmask_col]


    if row is not None:
        at = ApplyTransforms()
        at.inputs.input_image = inmask
        at.inputs.interpolation = "NearestNeighbor"
        at.inputs.reference_image = row.boldref
        at.inputs.transforms = [row.mnitoT1w]
        at.inputs.output_image = row[outmask_col].as_posix()
        at.inputs.float = True
    else:
        at = ApplyTransforms()
        at.inputs.input_image = inmask
        at.inputs.interpolation = "NearestNeighbor"
        at.inputs.reference_image = reference
        at.inputs.transforms = transforms
        at.inputs.output_image = output_image
        at.inputs.float = True
    _ = at.run()


def transform_stat_to_t1w(
    row, inmask_col=None, inmask=None, outmask_col="boldmask_path"
):
    if inmask_col is None and inmask is None:
        raise ValueError("one of mask_col or mask must be defined")
    elif inmask_col is not None:
        inmask = row[inmask_col]

    at = ApplyTransforms()
    at.inputs.input_image = inmask
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


def select_confounds(cfds, cfds_sel):
    if not isinstance(cfds, pd.DataFrame):
        cfd = pd.read_csv(cfds, sep="\t")
    else:
        cfd = cfds.copy()
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
    if "-gm" in cfds_sel:
        cols.extend(["gm"])
    cfds = cfd.loc[:, cols].copy()
    return cfds


def add_censor_columns(cfds, boldmask_path, bold_path,
                       max_outfrac=None, max_fd=None, frames_before=0, frames_after=0,
                       minimum_segment_length=None, minimum_total_length=None, n_dummy=0):
    """
    Add censor columns to a confounds dataframe and return the dataframe.
    !NOTE! will remove any columns containing "censor_".

    Parameters
    ==========

    cfds : str or Path or pandas DataFrame
        Path to fMRIPrep confounds tsv.
    boldmask_path : str or Path
        Path to mask for bold image.
    bold_path : str or Path
        Path to bold timeseries.
    max_outfrac : float or None, default None
        Maximum allowed fraction of outlier voxels in a frame.
    max_fd : float or None, default None
        Maximum allowed framewise displacement.
    frames_before : int, default 0
        How many frames to exclude prior to a frame excluded because of framewise displacement.
    frames_after : int, default 0
        How many frames to exclude after a frame excluded because of framewise displacement.
    minimum_segment_length : int or None, default None
        Minimum number of consecutive non-censored frames allowed.
    n_dummy : int, default 0
        Number of dummy frames to censor at the begining of the time series.
    minimum_total_length  : int or None, default None
        Minimum number of uncesored frames. Raises a value error if fewer frames survive censoring.

    Returns
    =======
    cfds : Dataframe
        Pandas dataframe with censor columns added.
    """
    if not isinstance(cfds, pd.DataFrame):
        cfds = pd.read_csv(cfds, sep='\t')
    cfds['censored'] = False

    if max_outfrac is not None:
        cfds['outfrac'] = toutcount(bold_path, cfds, boldmask_path, n_dummy)
        cfds['censored'] |= cfds.outfrac > max_outfrac

    if max_fd is not None:
        shifted_before = [cfds.framewise_displacement.shift(-1 * (ff + 1)) for ff in range(frames_before)]
        shifted_after = [cfds.framewise_displacement.shift((ff + 1)) for ff in range(frames_after)]
        shifted = pd.DataFrame(shifted_before + [cfds.framewise_displacement] + shifted_after).T
        cfds['max_fd_with_offsets'] = shifted.max(1)
        cfds['censored'] |= cfds.max_fd_with_offsets > max_fd

    if minimum_segment_length is not None:
        segs = np.diff(np.pad((~cfds['censored']).astype(int), 1, 'constant'))
        seg_srt = np.where(segs == 1)[0]
        seg_end = np.where(segs == -1)[0] - 1
        cfds['segment_length'] = 0
        for srt, end in zip(seg_srt, seg_end):
            cfds.loc[srt:end, 'segment_length'] = end - srt + 1

        cfds['censored'] |= cfds.segment_length < minimum_segment_length

    cfds.censored = cfds.censored.astype(int)

    # censor dummy scans
    if n_dummy != 0:
        cfds.loc[:n_dummy - 1, 'censored'] = 1

    if (minimum_total_length is not None) and ((len(cfds) - cfds.censored.sum()) < minimum_total_length):
        raise ValueError(
            f'Only {(len(cfds) - cfds.censored.sum())} frames remain after censoring. This is fewer than the {minimum_total_length} frame minimum specified.')

    # drop previous censor columns
    cfds = cfds.loc[:, ~cfds.columns.str.contains('censor_')]
    # make censor columns
    c_cols = np.zeros((len(cfds.censored), len(cfds.censored))).astype(int)
    c_cols[np.diag_indices(len(cfds.censored))] = cfds.censored.values.astype(int)
    c_cols_names = [f'censor_{nn:03d}' for nn in range(cfds.censored.sum())]
    c_cols = pd.DataFrame(c_cols[:, cfds.censored.astype(bool)],
                          columns=c_cols_names)
    cfds = pd.concat([cfds, c_cols], axis=1)
    cfds = cfds.fillna(0)

    return cfds


def toutcount(bold_path, cfds, boldmask_path, n_dummy=0):
    """
    Based on AFNI's 3dToutCount. Returns the fraction of voxels inside the mask that are outliers at each TR.
    Parameters
    ==========
    bold_path : str or path
        Path to bold timeseries
    confound_path : str or path or pandas DataFrame
        Path to confound timeseries
    boldmask_path : str or path
        Path to boldmask
    n_dummy : int, default 0
        If passed, and bold and confounds are off by exactly n_dummy,
        then drop those from the front of confounds and carry on.

    Returns
    =======
    outfrac : array of floats
        Fraction of voxels inside the mask that are outliers at each TR
    """
    if not isinstance(cfds, pd.DataFrame):
        cfds = pd.read_csv(cfds, sep='\t')
    detrend_cfds = select_confounds(cfds, ['-cosine', '-dummy'])
    bold_img = nl.image.load_img(bold_path)
    pad_result = False
    if (bold_img.shape[-1] == len(cfds)):
        detrended_img = nl.image.clean_img(bold_path, standardize=False, detrend=False, confounds=detrend_cfds)
    elif ((bold_img.shape[-1] + n_dummy) == len(cfds)):
        detrend_cfds = detrend_cfds.loc[n_dummy:, :]
        detrended_img = nl.image.clean_img(bold_path, standardize=False, detrend=False, confounds=detrend_cfds)
        pad_result = True
    try:
        detrended_dat = nl.masking.apply_mask(detrended_img, boldmask_path)
    except ValueError:
        # THis throws a value error if the mask has more than one value
        # if you're passing the tedana good signal mask, it has more than one value, so this will deal with that
        boldmask_img = nl.image.load_img(boldmask_path)
        boldmask = nl.image.new_img_like(boldmask_img, data=boldmask_img.get_fdata() != 0, affine=boldmask_img.affine,
                                        copy_header=True)
        detrended_dat = nl.masking.apply_mask(detrended_img, boldmask)

    medians = np.median(detrended_dat, axis=0)
    deviations = np.abs(detrended_dat - medians)
    mad = np.median(deviations, axis=0)
    n = len(detrended_dat)
    alpha = stats.norm().ppf(1 - (0.001 / n))
    thresh = alpha * np.sqrt(np.pi / 2) * mad
    res = (deviations >= thresh).sum(axis=1) / mad.shape[0]
    if pad_result:
        padded_res = np.zeros(len(cfds))
        padded_res[n_dummy:] = res
        return padded_res
    else:
        return res


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


def cluster(
    stat_img_path,
    out_path=None,
    stim_roi_path=None,
    percentile=10,
    sign="negative",
    connectivity="NN3",
):
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
    connectivity : str ["NN1", "faces", "NN2", "edges", "NN3", "vertices"]
        Deffinition of connectivity to use for clustering accepts either description (faces, edges, verticies) or AFNI label (NN1, NN2, NN3).

    Returns
    -------
    biggest_clsut_img : niiimage
        Mask for the biggest cluster

    Code inspired by https://github.com/nilearn/nilearn/blob/b7e5efdd37a6b4cc276763fc5f2a2cde81f7af73/nilearn/image/image.py#L790
    """
    if (connectivity == "NN1") | (connectivity == "faces"):
        bin_struct = generate_binary_structure(3, 1)
    elif (connectivity == "NN2") | (connectivity == "edges"):
        bin_struct = generate_binary_structure(3, 2)
    elif (connectivity == "NN3") | (connectivity == "vertices"):
        bin_struct = generate_binary_structure(3, 3)
    else:
        raise ValueError(
            f"You specified connectivity={connectivity}, but the only supported terms are:"
            "NN1 or faces"
            "NN2 or edges"
            "NN3 or vertices"
        )
    stat_img = nl.image.load_img(stat_img_path)
    if stim_roi_path is None:
        stimroi_dat = stat_img.get_fdata() != 0
        stimroi = nl.image.new_img_like(stat_img, stimroi_dat, affine=stat_img.affine)
    else:
        stimroi = nl.image.load_img(stim_roi_path)
    masked_stat = nl.masking.apply_mask(stat_img, stimroi)
    threshed = masked_stat.copy()

    if sign == "negative":
        sign_masked_stat = masked_stat[masked_stat < 0]
        threshold = np.percentile(sign_masked_stat, percentile)
        threshed[(masked_stat > 0) | (masked_stat > threshold)] = 0
        threshed[threshed != 0] = 1
    elif sign == "positive":
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
    biggest_clust_dat = label_map == biggest_clust_id
    biggest_clust_img = nl.image.new_img_like(
        stat_img, biggest_clust_dat, affine=stat_img.affine, copy_header=True
    )
    if out_path is not None:
        biggest_clust_img.to_filename(out_path)
    return biggest_clust_img


def make_fmriprep_t2(fmriprep_dir, subject, out_dir):
    if subject[:4] == "sub-":
        subject = subject
    else:
        subject = f"sub-{subject}"

    t1_paths = find_bids_files(
        fmriprep_dir / f"{subject}", type="anat", suffix="T1w", extension=".nii.gz"
    )
    t1_paths = [tp for tp in t1_paths if "space" not in tp.parts[-1]]
    if len(t1_paths) > 1:
        raise ValueError(f"Looking for a single T1, found {len(t1_paths)}: {t1_paths}")
    elif len(t1_paths) == 0:
        t1_paths = find_bids_files(
            fmriprep_dir / f"{subject}", ses="*", type="anat", suffix="T1w", extension=".nii.gz"
        )
        t1_paths = [tp for tp in t1_paths if "space" not in tp.parts[-1]]
        if len(t1_paths) > 1:
            raise ValueError(f"Looking for a single T1, found {len(t1_paths)}: {t1_paths}")
        elif len(t1_paths) == 0:
            find_bids_files(
                fmriprep_dir / f"{subject}", debug=True, ses="*", type="anat", suffix="T1w", extension=".nii.gz",
            )
            raise ValueError(f"Couldn't find a T1.")
    fmriprep_t1 = t1_paths[0]

    fs_subjects_dir = fmriprep_dir / "sourcedata/freesurfer"
    fst2 = fs_subjects_dir / f"{subject}/mri/T2.mgz"

    if not fst2.exists():
        raise FileNotFoundError(fst2.as_posix())
    fsnative_to_t1_ents = {
        "suffix": "xfm",
        "extension": "txt",
        "from": "fsnative",
        "to": "T1w",
        "mode": "image",
        "type": "anat",
    }
    fsnative_to_t1 = update_bidspath(fmriprep_t1, fmriprep_dir, fsnative_to_t1_ents)
    if not fsnative_to_t1.exists():
        raise FileNotFoundError(fsnative_to_t1)


    out_dir.mkdir(exist_ok=True, parents=True)
    out_t2fsn = out_dir / f"{subject}_space-fsnative_desc-preproc_T2w.nii.gz"
    out_t2 = out_dir / f"{subject}_desc-preproc_T2w.nii.gz"

    os.environ["SUBJECTS_DIR"] = fs_subjects_dir.as_posix()

    mc = MRIConvert()
    mc.inputs.in_file = fst2
    mc.inputs.out_file = out_t2fsn.as_posix()
    mc.inputs.out_type = "niigz"
    _ = mc.run()

    at = ApplyTransforms()
    at.inputs.input_image = out_t2fsn
    at.inputs.reference_image = fmriprep_t1
    at.inputs.transforms = [fsnative_to_t1]
    at.inputs.output_image = out_t2.as_posix()
    at.inputs.float = True
    _ = at.run()


def parse_bidsname(filename):
    res = {}
    levels = Path(filename).parts
    if len(levels) > 1:
        res["type"] = Path(filename).parts[-2]
        parts = Path(filename).parts[-1].split("_")
    else:
        parts = filename.split("_")
    np = len(parts)
    for ii, part in enumerate(parts):
        if ii != (np - 1):
            pp = part.split("-")
            res[pp[0]] = "-".join(pp[1:])
        else:
            pp = part.split(".")
            res["suffix"] = pp[0]
            res["extension"] = ".".join(pp[1:])
    return res


BIDS_ORDER = [
    "type",
    "sub",
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "dir",
    "run",
    "mod",
    "echo",
    "flip",
    "inv",
    "mt",
    "part",
    "recording",
    "hemi",
    "space",
    "den",
    "res",
    "atlas",
    "label",
    "from",
    "to",
    "mode",
    "desc",
    "suffix",
    "extension",
]


def build_bidsname(parts, exclude=None, order=None):
    filename = ""
    containing = ""
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]
    if order is None:
        order = BIDS_ORDER
    for k in parts.keys():
        if (not k in order) and (not k in exclude):
            raise ValueError(f"Got key {k}, which is not in entity order {order}.")
    tmp = {}
    for oo in order:
        if oo in parts:
            tmp[oo] = parts[oo]
    for k, v in tmp.items():
        if k in exclude:
            continue
        if k == "type":
            containing = v
        elif k == "suffix":
            filename += f"{v}."
        elif k == "extension":
            if v[0] == ".":
                v = v[1:]
            filename += v
        elif v:
            filename += f"{k}-{v}_"
        else:
            filename += f"{k}_"
    if containing:
        return (Path(containing) / filename).as_posix()
    else:
        return filename


def find_bids_files(search_root, exclude=None, order=None, debug=False, try_rundir=True, **ents):
    if exclude is None:
        exclude = []
    else:
        if isinstance(exclude, str):
            exclude = [exclude]
        for ee in exclude:
            ents.pop(ee, None)
    search_dir = Path(search_root)
    if order is None:
        order = BIDS_ORDER
    for k in ents.keys():
        if not k in order:
            raise ValueError(f"Got key {k}, which is not in entity order {order}.")
    if "sub" in ents:
        search_dir /= f'sub-{ents["sub"]}'
    if "ses" in ents:
        search_dir /= f'ses-{ents["ses"]}'
    if "type" in ents:
        search_dir /= f'{ents["type"]}'
    tmp = {}
    for oo in order:
        if oo in ents and oo != "type":
            if ents[oo] is not None:
                tmp[oo] = ents[oo]
    glob_string = ""
    ending = ""
    for k, v in tmp.items():
        if k == "suffix":
            ending += f"{v}."
        elif k == "extension":
            if v[0] == ".":
                v = v[1:]
            ending += f"{v}"
        elif v:
            glob_string += f"*{k}-{v}_"
        else:
            glob_string += f"*{k}_"
    glob_string += f"*{ending}"
    if '*' in search_dir.as_posix():
        new_search_dir = Path(search_dir.parts[0])
        wc_found = False
        glob_prefix=''
        for pp in search_dir.parts[1:]:
            if not wc_found and '*' not in pp:
                new_search_dir /= pp
            else:
                wc_found = True
                glob_prefix += pp + '/'
        search_dir = new_search_dir
        glob_string = glob_prefix + glob_string


    if debug:
        print(search_dir, glob_string, flush=True)
    res = sorted(search_dir.glob(glob_string))
    if (len(res) == 0) and try_rundir:
        glob_string = f'run-{ents.get("run", "*")}/{glob_string}'
        res = sorted(search_dir.glob(glob_string))
    return res


def get_rel_path(source_path, target_path):
    """
    Find the point at which a source path and target path share a common folder
    and returns a relative path from the source path to the target path.

    Parameters:
    -----------
    source_path : (str or pathlib.Path)
        The path to the source folder.
    target_path : (str or pathlib.Path)
        The path to the target folder.

    Returns:
    pathlib.Path
        The relative path from the source path to the target path.

    Example:
        >>> get_rel_path('a/b/c/d', 'a/e/f')
        Path('../../e/f')
    """
    path_a = Path(target_path)
    path_b = Path(source_path)

    res = None
    tmp_path = path_b
    n_up = -1
    while res is None:
        try:
            res = path_a.relative_to(tmp_path)
        except ValueError:
            tmp_path = tmp_path.parent
            n_up += 1
    return Path("/".join([".."] * n_up)) / res


def make_rel_symlink(source_path, target_path):
    """
    Create a relative symlink from the source path to the target path.

    Parameters:
        source_path (str or pathlib.Path): The path to the source folder.
        target_path (str or pathlib.Path): The path to the target folder.

    Returns:
        None.

    Raises:
        ValueError: If source_path already exists but does not point to the target_path.
    """
    if source_path.exists():
        if source_path.resolve() == target_path.resolve():
            return
        else:
            raise ValueError(
                f"{source_path} exists and does not point to target. Check your paths and maybe delete {source_path}."
            )
    source_path = Path(source_path)
    target_path = Path(target_path)
    rel_path = get_rel_path(source_path, target_path)
    assert (source_path.parent / rel_path).exists()
    source_path.symlink_to(get_rel_path(source_path, target_path))


def update_bidspath(
    orig_path, new_bids_root, updates, exists=False, exclude=None, order=None, keep_rundir=False
):
    """
    Creates a BIDS-ish file path based on the entities in orig_paths with the specified changes.

    Parameters
    ----------
    orig_path : str or Path-like
        The original BIDS-like file path.
    new_bids_root : str or Path-like
        The the root directory for the new path
    updates : dict
        A dictionary of key-value pairs representing the new labels to add to the BIDS file path.
    exists : bool, optional
        If True, check whether the new file path already exists and raise an error if it does not. Defaults to False.
    exclude : list, optional
        A list of BIDS keys to exclude when constructing the new file path. Defaults to None.
    order : list, optional
        A list specifying the order in which BIDS keys should be included in the new file path. Defaults to None.
    keep_rundir : bool, optional
        If true and if there is a run directory between datatype and files, make sure it's there in the output too.

    Returns
    -------
    Path
        A new Path object representing the updated BIDS-compliant file path.

    Raises
    ------
    FileNotFoundError
        If the new file path does not exist and the `exists` parameter is set to True.

    Description
    -----------
    This function updates a BIDS-compliant file path with new subject, session, or other label(s) specified in the
    `updates` dictionary. It constructs a new file path based on the new labels and the `new_bids_root` directory,
    and returns a new `Path` object representing the updated file path. The `exclude` parameter can be used to
    exclude certain BIDS keys from the new file path, and the `order` parameter can be used to specify the order
    in which BIDS keys should be included in the new file path.

    If the `exists` parameter is set to True, the function will check whether the new file path already exists and
    raise a `FileNotFoundError` if it does not. The `exclude` and `order` parameters are both optional and default
    to None if not specified. The `exclude` parameter can be used to exclude certain BIDS keys from the new file path,
    and the `order` parameter can be used to specify the order in which BIDS keys should be included in the new file path.

    """
    orig_path = Path(orig_path)
    insert_rundir = False
    if keep_rundir:
        if 'run-' in orig_path.parts[-2]:
            orig_path = orig_path.parent.parent / orig_path.parts[-1]
            insert_rundir = True

    if exclude is None:
        exclude = []
    new_bids_root = Path(new_bids_root)
    ents = parse_bidsname(orig_path)
    new_ents = ents.copy()
    new_ents.update(updates)
    new_name = build_bidsname(new_ents, exclude=exclude, order=order)

    if (
        ("ses" in ents)
        and (f'ses-{ents["ses"]}' in orig_path.parts)
        and (not "ses" in exclude)
    ):
        new_path = (
            new_bids_root / f"sub-{new_ents['sub']}/ses-{new_ents['ses']}/{new_name}"
        )
    else:
        new_path = new_bids_root / f"sub-{new_ents['sub']}/{new_name}"
    if insert_rundir:
        new_path = new_path.parent / f"run-{new_ents['run']}"/new_path.parts[-1]
    if exists:
        if not new_path.exists():
            raise FileNotFoundError(f'{new_path}')

    return new_path


STIMROIS = ["dilatedDLPFCspheres", "DLPFCspheres", "BA46sphere", "coleBA46", "expandedcoleBA46", "chexpandedcoleBA46"]


def get_stimroi_path(stimroi_name, stimroi_path=None, cifti=False, masked=False):
    """
    Return the path to the ROI file for the region stimulation is to be delivered to.

    Parameters
    ----------
    stimroi_name : str
        Name of the stimulated region ROI file, options are "dilatedDLPFCspheres", "DLPFCspheres", "BA46sphere".
    stimroi_path : str or None
        The path of the custom stimulated region ROI file. Only needed if name is not recognized
    cifti : bool, optional
        If True, return a cifti file instead of a nifti file. Default is False.
    masked : bool, optional
        If True, return an roi masked by the MNI152NLin6Asym brain mask. Masked and cifti can't both be true.

    Returns
    -------
    Path
        The path to the stimulated region ROI file.

    Raises
    ------
    ValueError
        If a custom stimulated region ROI name is provided but no path to that ROI file is provided.
        If cifti and masked are both True.
    FileNotFoundError
        If the stimulated region ROI file does not exist.
    """

    roi_dir = Path(resource_filename("contarg", "data/rois"))

    if stimroi_name in STIMROIS:
        if cifti and masked:
            raise ValueError("cifti and masked can't both be True.")
        elif masked:
            stim_roi_2mm_path = (
                roi_dir / f"{stimroi_name + 'masked'}_space-MNI152NLin6Asym_res-02.nii.gz"
            )
        elif cifti:
            stim_roi_2mm_path = (
                roi_dir / f"{stimroi_name}_space-fsLR_den-91k.dtseries.nii"
            )
        else:
            stim_roi_2mm_path = (
                roi_dir / f"{stimroi_name}_space-MNI152NLin6Asym_res-02.nii.gz"
            )
    elif stimroi_path is None:
        raise ValueError(
            f"Custom roi name passed for stimroi, {stimroi_name}, but no path to that roi was provided."
        )
    else:
        if cifti or masked:
            raise NotImplementedError("Getting masked or cifti versions of custom rois is not yet supported.")
        stim_roi_2mm_path = stimroi_path

    if not stim_roi_2mm_path.exists():
        raise FileNotFoundError(stim_roi_2mm_path.as_posix())
    return stim_roi_2mm_path


REFROIS = [
    "SGCsphere",
    "bilateralSGCspheres",
    "bilateralfullSGCsphere",
    "DepressionCircuit",
]


def get_refroi_path(refroi_name, refroi_path=None, cifti=False):
    """
    Return the path to the ROI file for the region stimulation is to be delivered to.

    Parameters
    ----------
    refroi_name : str
        Name of the reference region ROI file, options are "SGCsphere", "bilateralSGCspheres","bilateralfullSGCsphere", "DepressionCircuit".
    refroi_path : str or None
        The path of the custom reference ROI file. Only needed if name is not recognized.
    cifti : bool, optional
        If True, return a cifti file instead of a nifti file. Default is False.

    Returns
    -------
    Path
        The path to the reference region ROI file.

    Raises
    ------
    ValueError
        If a custom reference region ROI name is provided but no path to that ROI file is provided.
    FileNotFoundError
        If the reference region ROI file does not exist.
    """
    roi_dir = Path(resource_filename("contarg", "data/rois"))

    if refroi_name in REFROIS:
        if cifti:
            ref_roi_2mm_path = (
                roi_dir / f"{refroi_name}_space-fsLR_den-91k.dtseries.nii"
            )
        else:
            ref_roi_2mm_path = (
                roi_dir / f"{refroi_name}_space-MNI152NLin6Asym_res-02.nii.gz"
            )
    elif refroi_path is None:
        raise ValueError(
            f"Custom roi name passed refroi, {refroi_name}, but no path to that roi was provided."
        )
    else:
        ref_roi_2mm_path = refroi_path

    if not ref_roi_2mm_path.exists():
        raise FileNotFoundError(ref_roi_2mm_path.as_posix())
    return ref_roi_2mm_path


def t1w_mask_to_mni(mask_path, fmriprep_dir, out_dir, t1w_to_MNI152NLin6Asym_path=None,
                    nthreads=1):
    mask_path = Path(mask_path)
    fmriprep_dir = Path(fmriprep_dir)
    out_dir = Path(out_dir)

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
            mask_path,
            fmriprep_dir,
            t1w_to_MNI152NLin6Asym_ents,
            exclude=["desc", "ses", "task", "acq", "run", "atlas", "space"],
        )
        if not t1w_to_MNI152NLin6Asym_path.exists():
            t1w_to_MNI152NLin6Asym_path = update_bidspath(
                mask_path,
                fmriprep_dir,
                t1w_to_MNI152NLin6Asym_ents,
                exclude=["desc", "task", "acq", "run", "atlas", "space"],
                exists=True
            )

    ref = templateflow.api.get('MNI152NLin6Asym', resolution=2, suffix='T1w')[-1]

    mnimask_ents = dict(
        space="MNI152NLin6Asym",
        res="2"
    )
    mnimask_path = update_bidspath(
        mask_path,
        out_dir,
        mnimask_ents
    )
    mnimask_path.parent.mkdir(exist_ok=True, parents=True)

    at = ApplyTransforms(interpolation="NearestNeighbor", float=True)
    at.inputs.num_threads = nthreads
    at.inputs.input_image = mask_path
    at.inputs.output_image = mnimask_path.as_posix()
    at.inputs.reference_image = ref
    at.inputs.input_image_type = 0
    at.inputs.transforms = [t1w_to_MNI152NLin6Asym_path]
    _ = at.run()

    return mnimask_path


def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, ci.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def average_ranks(arr):
    """
    Computes the ranks of the elements of the given array along the last dimension.
    For ties, the ranks are _averaged_. Returns an array of the same dimension of `arr`.

    From: https://alextseng.net/blog/posts/20191115-vectorizing-ml-metrics/
    """
    sorted_inds = np.argsort(arr, axis=-1)  # Sorted indices
    ranks, ranges = np.empty_like(arr), np.empty_like(arr)
    ranges = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))

    np.put_along_axis(ranks, sorted_inds, ranges, -1)
    ranks = ranks.astype(int)

    sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
    diffs = np.diff(sorted_arr, axis=-1)
    # Pad with an extra zero at the beginning of every subarray
    pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
    # Wherever the diff is not 0, assign a value of 1; this gives a set of small indices
    # for each set of unique values in the sorted array after taking a cumulative sum
    pad_diffs[pad_diffs != 0] = 1
    unique_inds = np.cumsum(pad_diffs, axis=-1).astype(int)

    unique_maxes = np.zeros_like(arr)  # Maximum ranks for each unique index
    # Using `put_along_axis` will put the _last_ thing seen in `ranges`, which will result
    # in putting the maximum rank in each unique location
    np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
    # We can compute the average rank for each bucket (from the maximum rank for each bucket)
    # using some algebraic manipulation
    diff = np.diff(unique_maxes, prepend=-1, axis=-1)  # Note: prepend -1!
    unique_avgs = unique_maxes - ((diff - 1) / 2)

    avg_ranks = np.take_along_axis(
        unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1
    )
    return avg_ranks


def _cross_corr(x, y=None, corr_only=False, spearman=False, output_mcps=False, **kwargs):
    """
        Compute correlation coefficient (spearman or pearson) between each pair of time series
    from x and y and apply multiple hypothesis testing correction.

    Parameters
    ----------
    x : ndarray
        An array of shape (n_samples_x, n_features) containing time series data.
    y : ndarray
        An array of shape (n_samples_y, n_features) containing time series data.
    corr_only : bool, default = False
        Only run the correlation
    spearman : bool, defalut = False
        Use Spearman correlation
    output_mcps : bool, default = False
        If True, output multiple comparison corrected p-values intead of boolean mc-sig.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the `multipletests` function.

    Returns
    -------
    rs : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the
        correlation coefficients between each pair of time series from x and y.
    ps : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the p-values
        associated with the Spearman's correlation coefficients.
    mcsig : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the multiple hypothesis
        testing corrected significant results (True if significant, False otherwise).
    """
    if y is None:
        if spearman:
            rs = squareform(1 - pdist(average_ranks(x), metric="correlation"))
        else:
            rs = squareform(1 - pdist(x,metric="correlation"))
    else:
        if spearman:
            rs = 1 - cdist(average_ranks(x), average_ranks(y), metric="correlation")
        else:
            rs = 1 - cdist(x, y, metric="correlation")
    if corr_only:
        return rs, None, None
    n = x.shape[-1]
    ts = rs * np.sqrt((n - 2) / ((rs + 1.0) * (1.0 - rs)))
    ps = stats.t.sf(np.abs(ts), n - 2) * 2
    mcsig, mcps, _, _ = multipletests(ps.flatten(), **kwargs)
    mcsig = mcsig.reshape(ps.shape)
    if output_mcps:
        return rs, ps, mcps
    else:
        return rs, ps, mcsig

def cross_spearman(x, y, corr_only=False, **kwargs):
    """
        Compute Spearman's correlation coefficient between each pair of time series
    from x and y using a nested loop and apply multiple hypothesis testing
    correction.

    Parameters
    ----------
    x : ndarray
        An array of shape (n_samples_x, n_features) containing time series data.
    y : ndarray
        An array of shape (n_samples_y, n_features) containing time series data.
    corr_only : bool, default = False
        Only run the correlation
    kwargs : dict, optional
        Additional keyword arguments to be passed to the `multipletests` function.

    Returns
    -------
    spear_rs : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the Spearmans's
        correlation coefficients between each pair of time series from x and y.
    spear_ps : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the p-values
        associated with the Spearman's correlation coefficients.
    mcsig : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the multiple hypothesis
        testing corrected significant results (True if significant, False otherwise).
    """
    return _cross_corr(x, y, corr_only=corr_only, spearman=True, **kwargs)

def cross_pearson(x, y, corr_only=False, **kwargs):
    """
        Compute Pearson's correlation coefficient between each pair of time series
    from x and y using a nested loop and apply multiple hypothesis testing
    correction.

    Parameters
    ----------
    x : ndarray
        An array of shape (n_samples_x, n_features) containing time series data.
    y : ndarray
        An array of shape (n_samples_y, n_features) containing time series data.
    corr_only : bool, default = False
        Only run the correlation
    kwargs : dict, optional
        Additional keyword arguments to be passed to the `multipletests` function.

    Returns
    -------
    spear_rs : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the Pearsons's
        correlation coefficients between each pair of time series from x and y.
    spear_ps : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the p-values
        associated with the Spearman's correlation coefficients.
    mcsig : ndarray
        An array of shape (n_samples_x, n_samples_y) containing the multiple hypothesis
        testing corrected significant results (True if significant, False otherwise).
    """
    return _cross_corr(x, y, corr_only=corr_only, spearman=False, **kwargs)


def graph_from_triangles(triangles):
    pairs = np.vstack([triangles[:,[0,1]], triangles[:,[0,2]], triangles[:,[1,2]]])
    G = nx.Graph()
    G.add_edges_from(pairs)
    return G


class SurfROI(object):
    def __init__(self, surface, hemisphere, timeseries=None, roi=None, idxs=None, take_largest_cc=False, dilate=0,
                 exclude_mask=None):

        self.take_largest_cc = take_largest_cc
        self._dilated = dilate
        self.ts = None
        self._surf_ts_data = None

        if roi is None and idxs is None:
            raise ValueError('One of roi or idxs must be provided')

        # load surface
        if not isinstance(surface, nb.gifti.gifti.GiftiImage):
            surface_path = Path(surface)
            self.surface = nb.load(surface)
        else:
            self.surface = surface
        self._surf_coords, self._surf_tris = self.surface.agg_data()
        self._G = graph_from_triangles(self._surf_tris)

        # load roi
        if roi is not None:
            if not isinstance(roi, nb.cifti2.cifti2.Cifti2Image):
                roi_path = Path(roi)
                self.roi_img = ci.load(roi_path)
            else:
                self.roi_img = roi
        else:
            self.roi_img = None

        # process hemisphere
        if hemisphere.lower() in ['r', 'right']:
            self._hemisphere = 'right'
            self._structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
        elif hemisphere.lower() in ['l', 'left']:
            self._structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
            self._hemisphere = 'left'
        else:
            raise ValueError(f"hemisphere should be one of ['r', 'right', 'l', 'left'], got {hemisphere}")

        if roi is not None:
            self._roi_dat = surf_data_from_cifti(self.roi_img.get_fdata(dtype=np.float32),
                                                 self.roi_img.header.get_axis(1), self._structure).squeeze()
            if not len(self._roi_dat) == len(self._surf_coords):
                raise ValueError("ROI and surface don't have the same number of points")

        if roi is not None:
            roi_idxs = (self._roi_dat != 0).nonzero()[0]
        else:
            roi_idxs = idxs
        if self.take_largest_cc:
            self.idxs = np.array(list(max(nx.connected_components(self._G.subgraph(roi_idxs)), key=len)))
        else:
            self.idxs = roi_idxs

        while dilate > 0:
            self.idxs = np.unique(np.array(list(self._G.edges(self.idxs))).flatten())
            dilate -= 1

        if exclude_mask is not None:
            self.idxs = self.idxs[~np.isin(self.idxs, np.nonzero(exclude_mask)[0])]

        # load timeseries
        if timeseries is not None:
            self.set_timeseries(timeseries)

        if not self._surf_coords.shape[-1] == 3:
            raise ValueError(
                f"Expected first element of surface to have 3 values per row, found {self._surf_points.shape[-1]}")
        if not self._surf_tris.shape[-1] == 3:
            raise ValueError(
                f"Expected second element of surface to have 3 values per row, found {self._tris_points.shape[-1]}")
        if not self._surf_tris.shape[0] >= self._surf_coords.shape[0]:
            raise ValueError(
                "Second element of surface should encode triangles and have more values than the number of points")

        self.connectivity = nx.to_numpy_array(self._G, self.idxs)
        self.coords = self._surf_coords[self.idxs]

    def set_timeseries(self, timeseries):
        if not isinstance(timeseries[0], np.ndarray):
            ts_datas = []
            for ts in timeseries:
                timeseries_img = ci.load(ts)
                ts = surf_data_from_cifti(timeseries_img.get_fdata(dtype=np.float32), timeseries_img.header.get_axis(1),
                                          self._structure).squeeze()
                ts_datas.append(ts)
            ts_data = np.hstack(ts_datas)
        else:
            ts_data = timeseries

        if ts_data.shape[0] != self._surf_coords.shape[0]:
            raise ValueError(
                "Timeseries first dimension should be the same size as first dimension of the first element of the surface gifti")
        self._surf_ts_data = ts_data
        self.ts = self._surf_ts_data[self.idxs]


def dilate_subgraph(G, nodes, times=1):
    """
    Return a new array that includes nodes and all nodes connected to them,
    effectivly dilating the list of nodes by one hop.
    """
    while times > 0:
        nodes = np.unique(np.array(list(G.edges(nodes))).flatten())
        times -= 1
    return nodes


def load_timeseries(timeseries):
    rts_datas = []
    lts_datas = []
    ts_datas = []
    for ts in timeseries:
        timeseries_img = ci.load(ts)
        lts = surf_data_from_cifti(timeseries_img.get_fdata(dtype=np.float32), timeseries_img.header.get_axis(1),
                                   'CIFTI_STRUCTURE_CORTEX_LEFT').squeeze()
        rts = surf_data_from_cifti(timeseries_img.get_fdata(dtype=np.float32), timeseries_img.header.get_axis(1),
                                   'CIFTI_STRUCTURE_CORTEX_RIGHT').squeeze()

        lts_datas.append(lts)
        rts_datas.append(rts)
        ts_datas.append(timeseries_img.get_fdata())
    lts_data = np.hstack(lts_datas)
    rts_data = np.hstack(rts_datas)
    ts_data = np.vstack(ts_datas).T

    return lts_data, rts_data, ts_data


def new_cifti_like(data, ref_path, dtype=None, surface_labels=None, volume_label=None, metadata=None):
    """Create a new CIFTI-2 image with the same structure as a reference image. Only 1-d data is implemented right now.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be stored in the new CIFTI-2 image.
    ref_path : str
        The path to the reference image that defines the structure of the new image.
    dtype : optional
        The data type of the new image. If not specified, the data type of the reference image is used.
    surface_labels : optional
        A dictionary of paths to surface label files. Each key should be a string that identifies the hemisphere
        (either "LEFT" or "RIGHT"). The corresponding value should be the path to the surface label file for that
        hemisphere. If not specified, default surface labels for the 91k surface will be used.
    volume_label : optional
        The path to a volume label file. If not specified, a default volume label for the 91k volume will be used.
    metadata : optional
        A dictionary of metadata to be attached to the CIFTI-2 image. If not specified, default metadata for the 91k
        structure will be used.

    Returns
    -------
    nb.cifti2.Cifti2Image
        A new CIFTI-2 image with the same structure as the reference image and the specified data.

    Raises
    ------
    ValueError
        If the data does not have the same number of values as teh reference image, or if the reference image does not
        have brain model axis elements that match those in niworkflows.interfaces.cifti.CIFTI_STRUCT_WITH_LABELS.
    """

    surface_labels_91k, volume_label_91k, metadata_91k = _prepare_cifti("91k")
    if surface_labels is None:
        surface_labels = surface_labels_91k
    if volume_label is None:
        volume_label = volume_label_91k
    if metadata is None:
        metadata = metadata_91k

    label_img = nb.load(volume_label)
    label_data = np.asanyarray(label_img.dataobj).astype("int16")

    ref_img = nb.load(ref_path)

    timepoints = 1
    bmaxis = None
    bmaxis_num = None
    for ii in ref_img.header.mapped_indices:
        if isinstance(ref_img.header.get_axis(ii), nb.cifti2.cifti2_axes.BrainModelAxis):
            bmaxis = ref_img.header.get_axis(ii)
            bmaxis_num = ii
    if bmaxis is None:
        raise ValueError("couldn't find a brain model axis")
    idx_offset = 0
    brainmodels = []
    if ref_img.get_fdata().shape[bmaxis_num] != data.shape[0]:
        raise ValueError("Data's not the right shape")

    for (name, data_indices, model), (structure, labels) in zip(bmaxis.iter_structures(),
                                                                CIFTI_STRUCT_WITH_LABELS.items()):
        assert name == structure
        if labels is None:  # surface model
            model_type = "CIFTI_MODEL_TYPE_SURFACE"
            # use the corresponding annotation
            hemi = structure.split("_")[-1]
            # currently only supports L/R cortex
            surf_verts = len(model.vertex)
            labels = nb.load(surface_labels[hemi == "RIGHT"])
            medial = np.nonzero(labels.darrays[0].data)[0]
            vert_idx = ci.Cifti2VertexIndices(model.vertex)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vert_idx),
                model_type=model_type,
                brain_structure=structure,
                vertex_indices=vert_idx,
                n_surface_vertices=surf_verts,
            )
            idx_offset += len(vert_idx)
        else:
            model_type = "CIFTI_MODEL_TYPE_VOXELS"
            for label in labels:
                ijk = np.nonzero(label_data == label)
                if ijk[0].size == 0:  # skip label if nothing matches
                    continue

            vox = ci.Cifti2VoxelIndicesIJK(model.voxel)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vox),
                model_type=model_type,
                brain_structure=structure,
                voxel_indices_ijk=vox,
            )
            idx_offset += len(vox)
        # add each brain structure to list
        brainmodels.append(bm)

    # add volume information
    brainmodels.append(
        ci.Cifti2Volume(
            bmaxis.volume_shape,
            ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, bmaxis.affine),
        )
    )

    # generate Matrix information
    series_map = ci.Cifti2MatrixIndicesMap(
        (0,),
        "CIFTI_INDEX_TYPE_SERIES",
        number_of_series_points=timepoints,
        series_exponent=0,
        series_start=0.0,
        series_step=0.0,
        series_unit="SECOND",
    )
    geometry_map = ci.Cifti2MatrixIndicesMap(
        (1,), "CIFTI_INDEX_TYPE_BRAIN_MODELS", maps=brainmodels
    )
    # provide some metadata to CIFTI matrix
    if not metadata:
        metadata = {
            "surface": "fsLR",
            "volume": "MNI152NLin6Asym",
        }
    # generate and save CIFTI image
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    matrix.metadata = ci.Cifti2MetaData(metadata)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=data.reshape(1, -1), header=hdr)
    if dtype is None:
        img.set_data_dtype(ref_img.get_data_dtype())
    else:
        img.set_data_dtype(dtype)
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")

    return img


def replace_cifti_data(data, ref_path):
    """from: https://neurostars.org/t/alter-size-of-matrix-for-new-cifti-header-nibabel/20903"""
    cii = nb.load(ref_path)
    h = cii.header
    f = cii.get_fdata()

    f_new = data

    ax_0 = nb.cifti2.SeriesAxis(start=1, step=1, size=f_new.shape[0])
    ax_1 = h.get_axis(1)

    # Create new header and cifti object
    new_h = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
    cii_new = nb.cifti2.Cifti2Image(f_new, new_h)

    return cii_new