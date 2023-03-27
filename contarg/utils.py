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
from nipype.interfaces.freesurfer import MRIConvert
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import (
    label,
    generate_binary_structure,
)
from pkg_resources import resource_filename


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
    fs_subjects_dir = fmriprep_dir / "sourcedata/freesurfer"
    fst2 = fs_subjects_dir / f"{subject}/mri/T2.mgz"
    fsnative_to_t1 = (
        fmriprep_dir
        / f"{subject}/anat/{subject}_from-fsnative_to-T1w_mode-image_xfm.txt"
    )
    fmriprep_t1 = fmriprep_dir / f"{subject}/anat/{subject}_desc-preproc_T1w.nii.gz"

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


def find_bids_files(search_root, exclude=None, order=None, debug=False, **ents):
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
    if debug:
        print(search_dir, glob_string, flush=True)
    return sorted(search_dir.glob(glob_string))


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
    orig_path, new_bids_root, updates, exists=False, exclude=None, order=None
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
    if exists:
        if not new_path.exists():
            raise FileNotFoundError(new_path)

    return new_path


STIMROIS = ["dilatedDLPFCspheres", "DLPFCspheres", "BA46sphere", "coleBA46"]


def get_stimroi_path(stimroi_name, stimroi_path=None, cifti=False):
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

    Returns
    -------
    Path
        The path to the stimulated region ROI file.

    Raises
    ------
    ValueError
        If a custom stimulated region ROI name is provided but no path to that ROI file is provided.
    FileNotFoundError
        If the stimulated region ROI file does not exist.
    """

    roi_dir = Path(resource_filename("contarg", "data/rois"))

    if stimroi_name in STIMROIS:
        if cifti:
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
        stim_roi_2mm_path = stimroi_path

    if not stim_roi_2mm_path.exists():
        raise FileNotFoundError(stim_roi_2mm_path.as_posix())
    return stim_roi_2mm_path


REFROIS = ["SGCsphere", "bilateralSGCspheres", "bilateralfullSGCsphere", "DepressionCircuit"]


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
