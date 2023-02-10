import pandas as pd
import numpy as np
import nilearn as nl
from nilearn import image, masking, maskers, plotting, datasets, connectome
from pathlib import Path
from .utils import iterable


def get_ref_vox_con(
    bold_path, mask_path, out_path, refroi_path, tr, smoothing_fwhm=4.0
):
    """
    Get the voxel wise connectivity map of a passed bold image with the reference roi.
    If an iterable of bold_paths is passed,they'll be concatenated. Runs global signal regression.

    Parameters
    ----------
    bold_path : str or path or iterable of strings or paths
    mask_path : str or path
        Path to whole brain mask to apply
    out_path : str or path
        Path to write connectivity map to
    refroi_path: str or path
        Path for reference roi
    tr : float
        TR of functional time series
    smoothing_fwhm : float default 4.0
        FWHM of gaussian smoothing to be applied

    """
    if not iterable(bold_path):
        bold_paths = [bold_path]
    else:
        bold_paths = bold_path

    subj_mask = nl.image.load_img(mask_path)
    ref_mask = nl.image.load_img(refroi_path)
    masked_ref_mask = nl.masking.apply_mask(ref_mask, subj_mask)
    gs_masker = nl.maskers.NiftiMasker(mask_img=subj_mask)
    subj_masker = nl.maskers.NiftiMasker(
        mask_img=subj_mask,
        low_pass=0.1,
        high_pass=0.01,
        smoothing_fwhm=smoothing_fwhm,
        t_r=tr,
        standardize=True,
    )
    # process each run
    clean_tses = []
    for bold_path in bold_paths:
        gs = gs_masker.fit_transform(bold_path).mean(1).reshape(-1, 1)
        cleaned = subj_masker.fit_transform(bold_path, confounds=gs)
        clean_tses.append(cleaned)
    cat_clean_tses = np.vstack(clean_tses)
    ref_ts = cat_clean_tses[:, masked_ref_mask.astype(bool)].mean(1).reshape(-1, 1)
    ref_vox = np.dot(cat_clean_tses.T, ref_ts) / ref_ts.shape[0]
    ref_vox_img = nl.masking.unmask(ref_vox.T, subj_mask)
    ref_vox_img.to_filename(out_path)

    return ref_vox_img
