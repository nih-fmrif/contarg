import pandas as pd
import numpy as np
import nilearn as nl
from nilearn import image, masking, maskers, plotting, datasets, connectome
from nilearn.interfaces.fmriprep import load_confounds
from pathlib import Path
from .utils import iterable

def get_nlearn_confounds(subj_mask,bold_path,confounds_strategy=None, custom_gsr=True, ndummy=None, **kwargs):
    if confounds_strategy :
        confounds,sample_mask = load_confounds(bold_path,strategy=confounds_strategy, **kwargs)
        if 'global_signal' in confounds_strategy:
            return confounds.to_numpy(),sample_mask # This will override custom_gsr and ndummy
        elif custom_gsr:
            gs_masker = nl.maskers.NiftiMasker(mask_img=subj_mask)
            confounds.loc[:,'gs'] = gs_masker.fit_transform(bold_path).mean(1).reshape(-1, 1)
            if ndummy :
                sample_mask = confounds.index[ndummy:].to_numpy() # This will override the sample mask
            return confounds.to_numpy(),sample_mask
        else :
            return confounds.to_numpy(),sample_mask
    else :
        gs_masker = nl.maskers.NiftiMasker(mask_img=subj_mask)
        confounds = gs_masker.fit_transform(bold_path).mean(1).reshape(-1, 1)
        _,sample_mask = load_confounds(bold_path,strategy=["non_steady_state"]) # This is just to get the non-steady in the same format.
        if ndummy :
                sample_mask = confounds.index[ndummy:].to_numpy() # This will override the sample mask
        
        return confounds, sample_mask
    
def get_ref_vox_con(
    bold_path, 
    mask_path, 
    refroi_path, 
    tr, 
    out_path=None, 
    smoothing_fwhm=4.0, 
    confounds_strategy=None, 
    **kwargs
):
    """
    Get the voxel wise connectivity map of a passed bold image with the reference roi.
    If an iterable of bold_paths is passed,they'll be concatenated. Runs global signal regression.

    Parameters
    ----------
    bold_path : str or path or iterable of strings or paths
    mask_path : str or path
        Path to whole brain mask to apply
    refroi_path: str or path
        Path for reference roi
    tr : float
        TR of functional time series
    out_path : str or path
        Path to write connectivity map to
    smoothing_fwhm : float default 4.0
        FWHM of gaussian smoothing to be applied
    confounds : 

    """
    if not iterable(bold_path):
        bold_paths = [bold_path]
    else:
        bold_paths = bold_path

    subj_mask = nl.image.load_img(mask_path)
    print(refroi_path)
    ref_mask = nl.image.load_img(refroi_path)
    masked_ref_mask = nl.masking.apply_mask(ref_mask, subj_mask)
    #gs_masker = nl.maskers.NiftiMasker(mask_img=subj_mask)
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
        #gs = gs_masker.fit_transform(bold_path).mean(1).reshape(-1, 1)
        confounds,sample_mask = get_nlearn_confounds(subj_mask,bold_path,confounds_strategy=confounds_strategy, custom_gsr=True, **kwargs)
        cleaned = subj_masker.fit_transform(bold_path, confounds=confounds, sample_mask=sample_mask) # Add the confounds from nilearn.interfaces.fmriprep.load_confounds
        clean_tses.append(cleaned)
    cat_clean_tses = np.vstack(clean_tses)
    ref_ts = cat_clean_tses[:, masked_ref_mask.astype(bool)].mean(1).reshape(-1, 1)
    ref_vox = np.dot(cat_clean_tses.T, ref_ts) / ref_ts.shape[0]
    ref_vox_img = nl.masking.unmask(ref_vox.T, subj_mask)
    if out_path is not None:
        ref_vox_img.to_filename(out_path)

    return ref_vox_img


def get_seedmap_vox_con(
    bold_path,
    mask_path,
    seedmap_path,
    stimroi_path,
    n_dummy,
    tr,
    out_path=None,
    smoothing_fwhm=4.0,
    confounds_strategy=None, 
    **kwargs
):
    """
    Get the representative time series of a passed bold image based on a seedmap.
    If an iterable of bold_paths is passed,they'll be concatenated. Runs global signal regression.

    Parameters
    ----------
    bold_path : str or path or iterable of strings or paths
    mask_path : str or path
        Path to whole brain mask to apply
    stimroi_path: str or path
        Path for stim roi
    tr : float
        TR of functional time series
    out_path : str or path
        Path to write connectivity map to
    smoothing_fwhm : float default 4.0
        FWHM of gaussian smoothing to be applied

    """
    if not iterable(bold_path):
        bold_paths = [bold_path]
    else:
        bold_paths = bold_path

    subj_mask = nl.image.load_img(mask_path)
    stim_mask = nl.image.load_img(stimroi_path)
    masked_stim_mask = nl.masking.apply_mask(stim_mask, subj_mask)
    seedmap = nl.image.load_img(seedmap_path)
    masked_seedmap = nl.masking.apply_mask(seedmap, subj_mask)
    masked_seedmap[masked_stim_mask != 0] = 0
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
        #gs = gs_masker.fit_transform(bold_path).mean(1).reshape(-1, 1)
        confounds,sample_mask = get_nlearn_confounds(subj_mask,bold_path,confounds_strategy=confounds_strategy, ndummy=n_dummy,custom_gsr=True, **kwargs)
        cleaned = subj_masker.fit_transform(bold_path, confounds=confounds, sample_mask=sample_mask) #[n_dummy:] # Add the confounds from nilearn.interfaces.fmriprep.load_confounds Should already remove the dummys if sample_mask works.
        clean_tses.append(cleaned)
    cat_clean_tses = np.vstack(clean_tses)
    seedmap_ts = np.average(cat_clean_tses, axis=1, weights=masked_seedmap)
    seedmap_ts = (seedmap_ts - seedmap_ts.mean(0)) / seedmap_ts.std(0)
    cat_clean_tses[:, masked_stim_mask == 0] = 0
    vox_dat = np.dot(cat_clean_tses.T, seedmap_ts) / seedmap_ts.shape[0]
    ref_vox_img = nl.masking.unmask(vox_dat.T, subj_mask)
    if out_path is not None:
        ref_vox_img.to_filename(out_path)

    return ref_vox_img


def get_com_in_mm(clust_img):
    clust_idxs = np.array(
        [list(rr) + [1] for rr in zip(*np.where(clust_img.get_fdata() != 0))]
    )
    clust_locs = np.matmul(clust_img.affine, clust_idxs.T).T[:, :3]
    return clust_locs.mean(0)
