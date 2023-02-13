from pathlib import Path
from pkg_resources import resource_filename
import click
from contarg.seedmap import get_ref_vox_con
import nilearn as nl
from nilearn import image, masking, maskers


@click.group()
def contarg():
    pass


@contarg.group()
def seedmap():
    pass


@seedmap.command()
@click.option(
    "--bold-path",
    type=click.Path(exists=True),
    multiple=True,
    help="Path to preprocessed bold timeseries. May be passed multiple times "
    "if multiple runs should be concatenated.",
)
@click.option(
    "--mask-path",
    type=click.Path(exists=True),
    help="Path to whole brain mask to apply",
)
@click.option(
    "--derivatives-dir",
    type=click.Path(exists=True),
    help="Path to derivatives directory in which outputs will be placed in a bids-ish style",
)
@click.option(
    "--gm-path",
    type=click.Path(exists=True),
    help="Path to grey matter mask or probabilistic map to apply",
)
@click.option(
    "--refroi-name",
    type=str,
    default="SGCsphere",
    help="Name of roi to whose connectivity is being used to pick a stimulation site. "
    "Should be one of ['SGCsphere', 'bilateralSGCSpheres'], "
    "or provide the path to the roi in MNI152NLin6Asym space with the refroi-path option.",
)
@click.option(
    "--refroi-path",
    type=str,
    default=None,
    help="If providing a custom ref roi, give the path to that ROI here.",
)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="Name of seedmap run. If provided,"
    "output will be placed in derivatives_dir/contarg/seedmap/run_name. Otherwise,"
    "output will be in derivatives_dir/contarg/seedmap",
)
@click.option(
    "--smoothing-fwhm",
    type=float,
    help="FWHM in mm of Gaussian blur to apply to functional",
)
@click.option(
    "--tr", "t_r", type=float, help="Repetition time of the functional time series"
)
@click.option(
    "--subject",
    type=str,
    default=None,
    help="Subject identifier to construct output file names",
)
@click.option(
    "--session",
    type=str,
    help="Session identifier to construct output file names",
)
@click.option(
    "--run",
    type=str,
    default=None,
    help="Run identifier to construct output file names. Only pass if you are not concatenating runs.",
)
def subjectmap(
    bold_path,
    mask_path,
    derivatives_dir,
    run_name,
    gm_path,
    refroi_name,
    refroi_path,
    smoothing_fwhm,
    t_r,
    subject,
    session,
    run,
):
    """
    Get the voxel wise connectivity map of a passed bold image with the reference roi.
    If multiple bold_paths is passed,they'll be concatenated. Runs global signal regression.
    Output will be masked by grey matter mask and stimroi.
    """
    bold_paths = bold_path
    derivatives_dir = Path(derivatives_dir)
    roi_dir = Path(resource_filename("contarg", "data/rois"))

    if refroi_name in ["SGCsphere", "bilateralSGCspheres"]:
        ref_roi_2mm_path = (
            roi_dir / f"{refroi_name}_space-MNI152NLin6Asym_res-02.nii.gz"
        )
    elif refroi_path is None:
        raise ValueError(
            f"Custom roi name passed refroi, {refroi_name}, but no path to that roi was provided."
        )
    else:
        ref_roi_2mm_path = refroi_path

    if run_name is not None:
        out_dir = derivatives_dir / "contarg" / "seedmap" / run_name
    else:
        out_dir = derivatives_dir / "contarg" / "seedmap"

    if subject[:4] == "sub-":
        subject = subject[4:]
    out_dir = out_dir / f"sub-{subject}"
    out_name = f"sub-{subject}"
    if session is not None:
        if session[:4] == "ses-":
            session = session[4:]
        out_dir = out_dir / f"ses-{session}"
        out_name += f"_ses-{session}"
    if run is not None:
        if run[:4] == "run-":
            run = session[4:]
        out_name += f"_run-{run}"
    out_dir = out_dir / "func"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_vox_con_path = out_dir / f"{out_name}_desc-RefCon_stat.nii.gz"
    masked_ref_vox_con_path = out_dir / f"{out_name}_desc-MaskedRefCon_stat.nii.gz"

    ref_vox_img = get_ref_vox_con(
        bold_paths,
        mask_path,
        ref_vox_con_path,
        ref_roi_2mm_path,
        t_r,
        smoothing_fwhm=smoothing_fwhm,
    )
    # mask ref_vox_img
    subj_mask = nl.image.load_img(mask_path)
    gm_img = nl.image.load_img(gm_path)

    subj_gm_dat = nl.masking.apply_mask(ref_vox_img, gm_img)
    ref_vox_dat = nl.masking.apply_mask(ref_vox_img, subj_mask)
    gm_ref_vox_dat = ref_vox_dat * subj_gm_dat
    gm_ref_vox_img = nl.masking.unmask(gm_ref_vox_dat, subj_mask)
    gm_ref_vox_img.to_filename(masked_ref_vox_con_path)


# TODO: command to combine masks into an average
