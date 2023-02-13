from pathlib import Path
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import click
from contarg.seedmap import get_ref_vox_con
import nilearn as nl
from nilearn import image, masking, maskers
from bids import BIDSLayout  # pip version = 0.15.2
from contarg.utils import make_path, transform_mask_to_t1w, transform_stat_to_t1w
from contarg.seedmap import get_seedmap_vox_con


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
            run = run[4:]
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

    subj_gm_dat = nl.masking.apply_mask(gm_img, subj_mask)
    ref_vox_dat = nl.masking.apply_mask(ref_vox_img, subj_mask)
    gm_ref_vox_dat = ref_vox_dat * subj_gm_dat
    gm_ref_vox_img = nl.masking.unmask(gm_ref_vox_dat, subj_mask)
    gm_ref_vox_img.to_filename(masked_ref_vox_con_path)

@seedmap.command()
@click.option("--contarg-dir", type=click.Path(exists=True),
              help="Path to contarg output directory containing subject maps to average."
                   "Example: [datadirectory]/derivatives/contarg/seedmap/working")
@click.option(
    "--session",
    type=str,
    default=None,
    help="Session from dataset to group map for.",
)
@click.option(
    "--run", type=str, default=None, help="Run from dataset to generate group map for."
)
def groupmap(
    contarg_dir,
    session,
    run
):
    """
    Average subject level connectivity maps to make group seedmap.
    """
    contarg_dir = Path(contarg_dir)
    roi_dir = Path(resource_filename("contarg", "data/rois"))
    if stimroi_name in ["DLPFCspheres", "BA46sphere"]:
        stim_roi_2mm_path = (
            roi_dir / f"{stimroi_name}_space-MNI152NLin6Asym_res-02.nii.gz"
        )
    elif stimroi_path is None:
        raise ValueError(
            f"Custom roi name passed for stimroi, {stimroi_name}, but no path to that roi was provided."
        )
    else:
        stim_roi_2mm_path = stimroi_path
    glob_str = "*"

    out_name_parts = []
    if session is not None:
        if session[:4] == "ses-":
            session = session[4:]
        glob_str += f"ses-{session}"
        out_name_parts.append(f"ses-{session}")
    if run is not None:
        if run[:4] == "run-":
            run = run[4:]
        glob_str += f"*run-{run}"
        out_name_parts.append(f"run-{run}")
    glob_str += '_desc-MaskedRefCon_stat.nii.gz'
    out_name_parts.append("desc-groupmask.nii.gz")
    out_name = "_".join(out_name_parts)

    subjmaps = sorted(contarg_dir.rglob(glob_str))
    stimroi = nl.image.load_img(stim_roi_2mm_path)
    stimroi_dat = stimroi.get_fdata()

    mapsum = np.zeros_like(stimroi_dat, dtype=float)

    for subjmap in subjmaps:
        subjimg = nl.image.load_img(subjmap)
        mapsum += subjimg.get_fdata().squeeze()
        del subjimg
    mapave = mapsum / len(subjmaps)

    mapave_img = nl.image.new_img_like(tmp, mapave, affine=tmp.affine, copy_header=True)
    mapave_img.to_filename(contarg_dir / out_name)


@seedmap.command()
@click.option("--bids-dir", type=click.Path(exists=True), help="Path to bids root.")
@click.option(
    "--derivatives-dir",
    type=click.Path(exists=True),
    help="Path to derivatives directory with fMRIPrep output.",
)
@click.option(
    "--database-file",
    type=click.Path(),
    help="Path to pybids database file (expects version 0.15.2), "
    "if one does not exist here, it will be created.",
)
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="Name of clustering run. If provided,"
    "output will be placed in derivatives_dir/contarg/hierarchical/run_name. Otherwise,"
    "output will be in derivatives_dir/contarg/hierarchical",
)
@click.option(
    "--seedmap-path",
    type=click.Path(exists=True),
    help="Path to seedmap image in MNI152NLin6Asym space",
)
@click.option(
    "--stimroi-name",
    type=click.Choice(["DLPFCspheres", "BA46sphere"]),
    default="DLPFCspheres",
    help="Name of roi to which stimulation will be delivered. "
    "Should be one of ['DLPFCspheres', 'BA46sphere'], "
    "or provide the path to the roi in MNI152NLin6Asym space with the stimroi-path option.",
)
@click.option(
    "--stimroi-path",
    type=str,
    default=None,
    help="If providing a custom stim roi, give the path to that ROI here.",
)
@click.option(
    "--space",
    type=click.Choice(["T1w", "MNI152NLin6Asym"]),
    default="T1w",
    show_default=True,
    help="Which space should results be output in. "
    "T1w for directing treatment; "
    "MNI152NLin6Asym for comparing reliabilities across individuals.",
)
@click.option(
    "--smoothing-fwhm",
    type=float,
    help="FWHM in mm of Gaussian blur to apply to functional",
)
@click.option(
    "--ndummy",
    "n_dummy",
    type=int,
    help="Number of dummy scans at the beginning of the functional time series",
)
@click.option(
    "--tr", "t_r", type=float, help="Repetition time of the functional time series"
)
@click.option(
    "--target-method",
    type=click.Choice(["classic", "cluster"]),
    default="cluster",
    show_default=True,
    help="How to pick a target coordinate from the seedmap weighted connectivity."
)
@click.option(
    "--connectivity",
    type=click.Choice(["NN1", "NN2", "NN3"]),
    default="average",
    show_default=True,
    help="Linkage type for clustering. Defined as in AFNI: NN1 is faces, NN2 is edges, NN3 is verticies.",
)
@click.option(
    "--percentile", type=float, help="All values more extreme than percentile will be kept for clustering"
)
@click.option(
    "--subject",
    type=str,
    default=None,
    help="Subject from dataset to generate target(s) for.",
)
@click.option(
    "--session",
    type=str,
    default=None,
    help="Session from dataset to generate target(s) for.",
)
@click.option(
    "--run", type=str, default=None, help="Run from dataset to generate target(s) for."
)
@click.option(
    "--echo",
    type=str,
    default=None,
    help="Echo from dataset to generate target(s) for.",
)
@click.option(
    "--njobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of jobs to run in parallel to find targets",
)
def run(
    bids_dir,
    derivatives_dir,
    database_file,
    run_name,
    stimroi_name,
    stimroi_path,
    seedmap_path,
    space,
    smoothing_fwhm,
    n_dummy,
    t_r,
    target_method,
    connectivity,
    percentile,
    subject,
    session,
    run,
    echo,
    njobs,
):
    bids_dir = Path(bids_dir)
    derivatives_dir = Path(derivatives_dir)
    database_path = Path(database_file)
    roi_dir = Path(resource_filename("contarg", "data/rois"))
    layout = BIDSLayout(
        bids_dir,
        database_path=database_path,
        derivatives=derivatives_dir / "fmriprep",
    )
    if run_name is not None:
        targeting_dir = derivatives_dir / "contarg" / "seedmap" / run_name
    else:
        targeting_dir = derivatives_dir / "contarg" / "seedmap"
    targeting_dir.mkdir(parents=True, exist_ok=True)

    if stimroi_name in ["DLPFCspheres", "BA46sphere"]:
        stim_roi_2mm_path = (
            roi_dir / f"{stimroi_name}_space-MNI152NLin6Asym_res-02.nii.gz"
        )
    elif stimroi_path is None:
        raise ValueError(
            f"Custom roi name passed for stimroi, {stimroi_name}, but no path to that roi was provided."
        )
    else:
        stim_roi_2mm_path = stimroi_path

    assert stim_roi_2mm_path.exists()

    # Getting all the needed input paths
    # build paths df off of bolds info
    get_kwargs = {
        "return_type": "object",
        "task": "rest",
        "desc": "preproc",
        "suffix": "bold",
        "space": "T1w",
        "extension": ".nii.gz",
    }
    if subject is not None:
        get_kwargs["subject"] = subject
    if session is not None:
        get_kwargs["session"] = session
    if run is not None:
        get_kwargs["run"] = run
    if echo is not None:
        get_kwargs["echo"] = echo
    bolds = layout.get(**get_kwargs)
    rest_paths = pd.DataFrame([bb.get_entities() for bb in bolds])
    rest_paths["entities"] = [bb.get_entities() for bb in bolds]
    rest_paths["bold_obj"] = bolds
    rest_paths["bold_path"] = [bb.path for bb in bolds]
    if "session" not in rest_paths.columns:
        rest_paths["session"] = None
    # add boldref
    rest_paths["boldref"] = rest_paths.entities.apply(
        lambda ee: layout.get(
            return_type="file",
            subject=ee["subject"],
            task=ee["task"],
            run=ee["run"],
            datatype="func",
            space="T1w",
            extension=".nii.gz",
            suffix="boldref",
        )
    )
    assert rest_paths.boldref.apply(lambda x: len(x) == 1).all()
    rest_paths["boldref"] = rest_paths.boldref.apply(lambda x: x[0])
    # add brain mask
    rest_paths["brain_mask"] = rest_paths.boldref.str.replace(
        "_boldref", "_desc-brain_mask"
    )
    assert rest_paths.brain_mask.apply(lambda x: Path(x).exists()).all()

    # add t1w path
    rest_paths["T1w"] = rest_paths.entities.apply(
        lambda ee: layout.get(
            return_type="file",
            subject=ee["subject"],
            desc="preproc",
            datatype="anat",
            extension=".nii.gz",
            suffix="T1w",
            space=None,
        )
    )
    assert rest_paths.T1w.apply(lambda x: len(x) == 1).all()
    rest_paths["T1w"] = rest_paths.T1w.apply(lambda x: x[0])

    # add mnito t1w path
    rest_paths["mnitoT1w"] = rest_paths.entities.apply(
        lambda ee: layout.get(
            return_type="file",
            subject=ee["subject"],
            datatype="anat",
            extension=".h5",
            suffix="xfm",
            to="T1w",
            **{"from": "MNI152NLin6Asym"},
        )
    )
    assert rest_paths.mnitoT1w.apply(lambda x: len(x) == 1).all()
    rest_paths["mnitoT1w"] = rest_paths.mnitoT1w.apply(lambda x: x[0])

    # add confounds path
    rest_paths["confounds"] = rest_paths.entities.apply(
        lambda ee: layout.get(
            return_type="file",
            subject=ee["subject"],
            task=ee["task"],
            run=ee["run"],
            datatype="func",
            extension=".tsv",
            suffix="timeseries",
            desc="confounds",
        )
    )
    assert rest_paths.confounds.apply(lambda x: len(x) == 1).all()
    rest_paths["confounds"] = rest_paths.confounds.apply(lambda x: x[0])

    if space == "T1w":
        # set up paths to transform stim and target rois back to subject space
        boldmask_pattern = "sub-{subject}/[ses-{session}/]func/sub-{subject}_[ses-{session}_]task-{task}_run-{run}_[echo-{echo}_]atlas-{atlas}_space-{space}_desc-{desc}_{suffix}.{extension}"
        stimmask_updates = {
            "desc": stimroi_name,
            "suffix": "mask",
            "space": space,
            "atlas": "Coords",
        }
        rest_paths[f"stim_mask"] = rest_paths.entities.apply(
            lambda x: make_path(
                x,
                stimmask_updates,
                boldmask_pattern,
                targeting_dir,
                layout.build_path,
                check_exist=False,
                check_parent=False,
                make_parent=True,
            )
        )
        rest_paths["stim_mask_exists"] = rest_paths.stim_mask.apply(
            lambda x: x.exists()
        )

        refmask_updates = {
            "desc": "seedmap",
            "suffix": "stat",
            "space": space,
            "atlas": "Coords",
        }
        rest_paths[f"seedmap"] = rest_paths.entities.apply(
            lambda x: make_path(
                x,
                refmask_updates,
                boldmask_pattern,
                targeting_dir,
                layout.build_path,
                check_exist=False,
                check_parent=False,
                make_parent=True,
            )
        )
        rest_paths["seedmap_exists"] = rest_paths.seedmap.apply(lambda x: x.exists())

    else:
        # TODO: clean this up for custom ROIs

        rest_paths[f"stim_mask"] = stim_roi_2mm_path
        rest_paths["stim_mask_exists"] = True
        rest_paths["seedmap"] = seedmap_path
        rest_paths["seedmap_exists"] = True

    if target_method == 'cluster':
        desc = f"{connectivity}.{target_method}.p{percentile}"
    else:
        desc = f"{connectivity}.{target_method}"

    clust_updates = {
        "desc": desc,
        "suffix": "targetclust",
        "space": space,
        "atlas": "Coords",
        "extension": ".nii.gz",
    }
    rest_paths[f"{desc}_targout_cluster"] = rest_paths.entities.apply(
        lambda x: make_path(
            x,
            clust_updates,
            boldmask_pattern,
            targeting_dir,
            layout.build_path,
            check_exist=False,
            check_parent=False,
            make_parent=True,
        )
    )
    rest_paths[f"{desc}_targout_cluster_exists"] = rest_paths[
        f"{desc}_targout_cluster"
    ].apply(lambda x: x.exists())

    clust_updates = {
        "desc": desc,
        "suffix": "targetcoords",
        "space": space,
        "atlas": "Coords",
        "extension": ".tsv",
    }
    rest_paths[f"{desc}_targout_coordinates"] = rest_paths.entities.apply(
        lambda x: make_path(
            x,
            clust_updates,
            boldmask_pattern,
            targeting_dir,
            layout.build_path,
            check_exist=False,
            check_parent=False,
            make_parent=True,
        )
    )
    rest_paths[f"{desc}_targout_coordinates_exists"] = rest_paths[
        f"{desc}_targout_coordinates"
    ].apply(lambda x: x.exists())

    # transform all the masks that don't exist
    rest_paths.loc[~rest_paths.stim_mask_exists].apply(
        lambda x: transform_mask_to_t1w(
            x, inmask=stim_roi_2mm_path.as_posix(), outmask_col="stim_mask"
        ),
        axis=1,
    )
    rest_paths["stim_mask_exists"] = rest_paths.stim_mask.apply(lambda x: x.exists())
    assert rest_paths.stim_mask_exists.all()

    rest_paths.loc[~rest_paths.seedmap_exists].apply(
        lambda x: transform_stat_to_t1w(
            x, inmask=seedmap_path.as_posix(), outmask_col="seedmap"
        ),
        axis=1,
    )
    rest_paths["seedmap_exists"] = rest_paths.seedmap.apply(lambda x: x.exists())
    assert rest_paths.seedmap_exists.all()

    # now that all of the paths are set
    # ref_vox_img = get_seedmap_vox_con()
