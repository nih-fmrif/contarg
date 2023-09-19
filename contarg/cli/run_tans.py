import os
import subprocess
from pkg_resources import resource_filename
import configparser
from pathlib import Path
import click
import numpy as np
from nibabel import cifti2 as ci

from contarg.tans import (
    write_headmodel_script,
    write_optimization_script,
    tans_inputs_from_fmriprep,
    write_sim_script,
    get_spatial_correlation,
    get_correlation_map,
    mask_to_cifti
)
from contarg.pfm import pfm_inputs_from_tedana, pfm_inputs_from_fmriprep
from contarg.utils import (
    find_bids_files,
    update_bidspath,
    get_stimroi_path,
    get_refroi_path,
    parse_bidsname,
    REFROIS,
    STIMROIS,
    t1w_mask_to_mni
)


@click.group()
def contarg():
    pass


@contarg.group()
def tans():
    pass


@tans.command()
@click.option(
    "--midnight-scan-club-path",
    "msc_path",
    type=click.Path(exists=True),
    prompt=True,
    help="path to utilities directory from midnight scan club repo (https://github.com/MidnightScanClub/MSCcodebase)",
)
@click.option(
    "--SimNIBS-path",
    "simnibs_path",
    type=click.Path(exists=True),
    prompt=True,
    help="path to installation directory for SimNIBS-4.0 (download from https://simnibs.github.io/simnibs/build/html/index.html)",
)
@click.option(
    "--TANS-path",
    "tans_path",
    type=click.Path(exists=True),
    prompt=True,
    default=resource_filename(
        "contarg", "resources/Targeted-Functional-Network-Stimulation"
    ),
    help="path to TANS repo included in contarg, but if you would like to use a different directory, specify it here.",
)
@click.option(
    "--overwrite", is_flag=True, help="overwrite a contarg config file if it exists."
)
def config(msc_path, simnibs_path, tans_path, overwrite):
    config_path = Path.home() / ".contarg"
    if config_path.exists() and not overwrite:
        raise ValueError(
            f"Existing contarg config found at {config_path}."
            f" If you would like to overwrite it, rerun with --overwrite"
        )
    config = configparser.ConfigParser()
    config["TANS"] = {
        "MidnightScanClubPath": msc_path,
        "SimNIBSPath": simnibs_path,
        "TANSPath": tans_path,
    }
    with config_path.open("w") as h:
        config.write(h)


@tans.command()
@click.option(
    "--t1",
    "t1w_path",
    type=click.Path(exists=True),
    help="Path to T1w image.",
)
@click.option(
    "--t2",
    "t2w_path",
    type=click.Path(),
    help="Path to T2w image.",
)
@click.option(
    "--out-dir",
    type=click.Path(),
    help="Path for outputs.",
)
@click.option(
    "--subject",
    type=str,
    default=None,
    help="Subject id.",
)
def headmodel(t1w_path, t2w_path, out_dir, subject):
    config_path = Path.home() / ".contarg"
    if not config_path.exists():
        raise ValueError(
            f"No contarg config found at {config_path}. Please run contarg tans config first."
        )
    config = configparser.ConfigParser()
    config.read(config_path)
    msc_path = config["TANS"]["MidnightScanClubPath"]
    simnibs_path = config["TANS"]["SimNIBSPath"]
    tans_path = config["TANS"]["TANSPath"]

    headmodels_script_path = write_headmodel_script(
        t1w_path, t2w_path, out_dir, subject, msc_path, simnibs_path, tans_path
    )
    headmodels_script_path = f"'{headmodels_script_path}'"
    cmd = [
        "matlab",
        "-nodisplay",
        "-nosplash",
        "-r",
        f"run({headmodels_script_path}); exit",
    ]
    print(f"running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


@tans.command()
@click.option(
    "--fmriprep-dir",
    type=click.Path(exists=True),
    help="Path to fmriprep outputs",
    required=True,
)
@click.option("--out-dir", type=click.Path(), help="Path for outputs.", required=True)
@click.option("--subject", type=str, default=None, help="Subject id.", required=True)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing outputs",
)
def prepanat(fmriprep_dir, out_dir, subject, overwrite):
    fmriprep_dir = Path(fmriprep_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    # prep for tans
    tans_inputs_from_fmriprep(
        fmriprep_dir, out_dir, subject=subject, overwrite=overwrite
    )


@tans.command()
@click.option(
    "--fmriprep-dir",
    type=click.Path(exists=True),
    help="Path to fmriprep outputs",
    required=True,
)
@click.option("--out-dir", type=click.Path(), help="Path for outputs.", required=True)
@click.option(
    "--tedana-dir",
    type=click.Path(),
    default=None,
    help="Path to TEDANA outputs",
)
@click.option(
    "--subjects-dir",
    type=click.Path(exists=True),
    default=None,
    help="Path freesurfer directory, defaults to {fmriprep_dir}/sourcedata/freesurfer. "
    "Only pass this if you need to point somewhere else.",
)
@click.option(
    "--target-method",
    type=click.Choice(["DCSS", "reference", "precomputed"]),
    default="DCSS",
    help="Method for selecting a target,depression circuit spatial similarity (DCSS), "
    "correlation with a reference seed (reference), "
    "and use of a precomputed mask (precomputed) are implemented.",
)
@click.option(
    "--precomputed-path",
    type=click.Path(exists=True),
    default=None,
    help="If method is 'precomputed', you must pass a path to the precomputed mask. "
         "If it's in T1w space, will attempt to transform it to MNI152NLin6Asym "
         "so that it can be converted to a surface.",
)
@click.option(
    "--stimroi-name",
    type=click.Choice(STIMROIS),
    default="DLPFCspheres",
    help="Name of roi to which stimulation will be delivered."
    f"Should be one of {STIMROIS}, "
    "or pass some other name and provide the path to the roi in MNI152NLin6Asym space with the stimroi-path option.",
)
@click.option(
    "--stimroi-path",
    type=str,
    default=None,
    help="If providing a custom stim roi, give the path to that ROI here.",
)
@click.option(
    "--refroi-name",
    type=click.Choice(REFROIS),
    default="SGCsphere",
    help="Name of roi to whose connectivity is being used to pick a stimulation site. "
    f"Should be one of {REFROIS}, "
    "or pass some other name and provide the path to the roi in MNI152NLin6Asym space with the refroi-path option.",
)
@click.option(
    "--refroi-path",
    type=str,
    default=None,
    help="If providing a custom ref roi, give the path to that ROI here.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.05,
    help="Minimum similarity (spatial correlation or correlation, depending on target-method)"
    " to be considered part of target.",
)
@click.option(
    "--aroma",
    is_flag=True,
    help="Include aroma regressors when cleaning functional time series.",
)
@click.option(
    "--ndummy",
    "n_dummy",
    type=int,
    help="Number of dummy scans at the beginning of the functional time series",
    required=True,
)
@click.option(
    "--tr",
    "t_r",
    type=float,
    help="Repetition time of the functional time series",
    required=True,
)
@click.option(
    "--cortical-smoothing",
    type=float,
    default=2.5,
    help="Geodesic smoothing to apply to surface data",
)
@click.option(
    "--subcortical-smoothing",
    type=float,
    default=2.5,
    help="Gaussian smoothing to apply to volumetric data",
)
@click.option(
    "--regress-gm",
    is_flag=True,
    help="Regress mean grey matter time series.",
)
@click.option(
    "--regress-globalsignal",
    is_flag=True,
    help="Regress mean global signal.",
)
@click.option(
    "--max-outfrac",
    type=float,
    default=None,
    show_default=True,
    help="Maximum allowed fraction of outlier voxels in a frame",
)
@click.option(
    "--max-fd",
    type=float,
    default=None,
    show_default=True,
    help="Maximum allowed framewise displacement.",
)
@click.option(
    "--frames-before",
    type=int,
    default=0,
    show_default=True,
    help="How many frames to exclude prior to a frame excluded because of framewise displacement.",
)
@click.option(
    "--frames-after",
    type=int,
    default=0,
    show_default=True,
    help="How many frames to exclude after a frame excluded because of framewise displacement.",
)
@click.option(
    "--minimum-segment-length",
    type=int,
    default=None,
    show_default=True,
    help="Minimum number of consecutive non-censored frames allowed.",
)
@click.option(
    "--minimum-total-length",
    type=int,
    default=None,
    show_default=True,
    help="Minimum number of consecutive non-censored frames allowed. "
         "Note this is an integer number of frames, not minutes.",
)
@click.option("--subject", type=str, help="Subject id.", required=True)
@click.option(
    "--session",
    type=str,
    default=None,
    help="Session id.",
)
@click.option(
    "--acquisition", type=str, default=None, help="Acquisition tag to filter scans by."
)
@click.option(
    "--run",
    type=str,
    default=[],
    help="Runs to use, pass once for each run. They will be concatenated.",
    multiple=True,
)
@click.option(
    "--echo",
    type=str,
    default=None,
    help="Echo from dataset to generate target(s) for.",
)
@click.option(
    "--nthreads",
    type=int,
    required=True,
    help="Number of threads to run in parallel, you really don't want to wait on this to finish single threaded",
)
@click.option(
    "--noanatinputprep",
    is_flag=True,
    help="skip inputprep step, outputs will be assumed to be in the correct place.",
)
@click.option(
    "--nofuncinputprep",
    is_flag=True,
    help="skip inputprep step, outputs will be assumed to be in the correct place.",
)
@click.option(
    "--noheadmodel",
    is_flag=True,
    help="skip headmodel step, outputs will be assumed to be in the correct place.",
)
# TODO: add a headmodel-path option
@click.option(
    "--nosimulation",
    is_flag=True,
    help="skip the simulation step, outputs will be assumed to be in the correct place.",
)
@click.option(
    "--nooptimization",
    is_flag=True,
    help="Skip the optimization step, output will be assumed to be in the correct place.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing outputs",
)
def run(
    fmriprep_dir,
    out_dir,
    tedana_dir,
    subjects_dir,
    target_method,
    precomputed_path,
    stimroi_name,
    stimroi_path,
    refroi_name,
    refroi_path,
    threshold,
    aroma,
    n_dummy,
    t_r,
    cortical_smoothing,
    subcortical_smoothing,
    regress_gm,
    regress_globalsignal,
    max_outfrac,
    max_fd,
    frames_before,
    frames_after,
    minimum_segment_length,
    minimum_total_length,
    subject,
    session,
    acquisition,
    run,
    echo,
    nthreads,
    noanatinputprep,
    nofuncinputprep,
    noheadmodel,
    nosimulation,
    nooptimization,
    overwrite,
):
    fmriprep_dir = Path(fmriprep_dir)
    tedana = False
    if tedana_dir is not None:
        tedana = True
        tedana_dir = Path(tedana_dir)
        if not tedana_dir.exists():
            raise FileNotFoundError(tedana_dir)
    if target_method not in ["DCSS", "reference", "precomputed"]:
        raise NotImplementedError(
            "I've only implemented depression circuit spatial similarity (DCSS) and reference correlation for TANS."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    # prep for tans
    if not noanatinputprep:
        tans_inputs_from_fmriprep(
            fmriprep_dir, out_dir, subject=subject, overwrite=overwrite
        )
    if precomputed_path is not None:
        precomputed_path = Path(precomputed_path)

    # get stimroi and ref roi
    stimroi_path = get_stimroi_path(stimroi_name, stimroi_path, cifti=True)
    refroi_path = get_refroi_path(refroi_name, refroi_path, cifti=True)
    # get config paths
    config_path = Path.home() / ".contarg"
    if not config_path.exists():
        raise ValueError(
            f"No contarg config found at {config_path}. Please run contarg tans config first."
        )
    config = configparser.ConfigParser()
    config.read(config_path)
    msc_path = config["TANS"]["MidnightScanClubPath"]
    simnibs_path = config["TANS"]["SimNIBSPath"]
    tans_path = config["TANS"]["TANSPath"]

    bold_paths = []
    cleaned_paths = []
    # find the runs to use and create necessary inputs
    print("preprocessing functional", flush=True)
    if tedana:
        # TODO: make confounds a passable option
        if len(run) > 0:
            for rr in run:
                bold_paths.extend(
                    find_bids_files(
                        tedana_dir,
                        type="func",
                        sub=subject,
                        ses=session,
                        acq=acquisition,
                        run=run,
                        desc="optcomDenoised",
                        suffix="bold",
                        extension=".nii.gz",
                    )
                )
        else:
            bold_paths.extend(
                find_bids_files(
                    tedana_dir,
                    type="func",
                    sub=subject,
                    ses=session,
                    acq=acquisition,
                    desc="optcomDenoised",
                    suffix="bold",
                    extension=".nii.gz",
                )
            )
        for boldtd_path in bold_paths:
            # TODO: make this less hacky and site specific
            if 'old' in boldtd_path.parts[-2]:
                continue
            cleaned_paths.append(
                pfm_inputs_from_tedana(
                    boldtd_path,
                    fmriprep_dir,
                    tedana_dir,
                    out_dir,
                    aroma,
                    n_dummy,
                    t_r,
                    nthreads,
                    regress_globalsignal=regress_globalsignal,
                    regress_gm=regress_gm,
                    noinputprep=nofuncinputprep,
                    max_outfrac=max_outfrac,
                    max_fd=max_fd,
                    frames_before=frames_before,
                    frames_after=frames_after,
                    minimum_segment_length=minimum_segment_length,
                    minimum_total_length=minimum_total_length,
                )
            )
    else:
        if len(run) > 0:
            for rr in run:
                bold_paths.extend(
                    find_bids_files(
                        fmriprep_dir,
                        type="func",
                        sub=subject,
                        ses=session,
                        acq=acquisition,
                        space="T1w",
                        run=run,
                        desc="preproc",
                        suffix="bold",
                        extension=".nii.gz",
                    )
                )
        else:
            bold_paths.extend(
                find_bids_files(
                    fmriprep_dir,
                    type="func",
                    sub=subject,
                    ses=session,
                    acq=acquisition,
                    space="T1w",
                    desc="preproc",
                    suffix="bold",
                    extension=".nii.gz",
                )
            )

        # set the subjects dir
        if subjects_dir is None:
            subjects_dir = fmriprep_dir / "sourcedata/freesurfer"
        subjects_dir = Path(subjects_dir)
        if not subjects_dir.exists():
            raise FileNotFoundError("Could not find subjects dir at {subjects_dir}")

        for bold_path in bold_paths:
            cleaned_paths.append(
                pfm_inputs_from_fmriprep(
                    subject,
                    subjects_dir,
                    fmriprep_dir,
                    out_dir,
                    bold_path,
                    t_r,
                    n_dummy,
                    overwrite=overwrite,
                    aroma=aroma,
                    regress_globalsignal=regress_globalsignal,
                    regress_gm=regress_gm,
                    max_outfrac=max_outfrac,
                    max_fd=max_fd,
                    frames_before=frames_before,
                    frames_after=frames_after,
                    minimum_segment_length=minimum_segment_length,
                    minimum_total_length=minimum_total_length,
                )
            )

    # TODO: maybe find a less hacky way to decide where to put the matlab script
    subses_out_dir = cleaned_paths[0].parent.parent

    subses_anat_dir = subses_out_dir / "anat/"
    if not subses_anat_dir.exists():
        if (subses_anat_dir / "../../anat").resolve().exists():
            subses_anat_dir.symlink_to("../anat")
        else:
            raise ValueError("Something's screwy with the paths. Dylan needs to fix it. "
                             f"{subses_anat_dir} doesn't exist. "
                             f"{(subses_anat_dir / '../../anat').resolve()} does not exist.")

    # smooth ciftis
    smoothed_paths = []
    for clean_path in cleaned_paths:
        if (cortical_smoothing != 0) or (subcortical_smoothing != 0):
            midthickL = subses_anat_dir / f"sub-{subject}.L.midthickness.32k_fs_LR.surf.gii"
            midthickR = subses_anat_dir / f"sub-{subject}.R.midthickness.32k_fs_LR.surf.gii"
            clean_ents = parse_bidsname(clean_path)
            smoothed_updates = dict(desc=f"{clean_ents['desc']+f'smoothed'}")
            smoothed_path = update_bidspath(clean_path, out_dir, smoothed_updates)
            cmd = [
                "wb_command",
                "-cifti-smoothing",
                clean_path.as_posix(),
                f"{cortical_smoothing}",
                f"{subcortical_smoothing}",
                "COLUMN",
                smoothed_path.as_posix(),
                "-left-surface",
                midthickL.as_posix(),
                "-right-surface",
                midthickR.as_posix(),
                "-merged-volume",
            ]
            subprocess.run(cmd, check=True)
            smoothed_paths.append(smoothed_path)
        else:
            smoothed_paths.append(clean_path)

    target_mask_updates = dict(
        desc=f"{target_method}Target", suffix="mask", space="fsLR", den="91k"
    )
    target_mask_path = update_bidspath(
        smoothed_paths[0], out_dir, target_mask_updates, exclude=["run"]
    )

    print("Finding Targets", flush=True)
    if target_method == "DCSS":
        # get spatial similarity to depression circuit
        similarity_updates = dict(
            desc="DepCircSimMap", suffix="stat", space="fsLR", den="91k"
        )
        similarity_path = update_bidspath(
            smoothed_paths[0], out_dir, similarity_updates, exclude=["run"]
        )
        dilstim_roi_2mm_path = get_stimroi_path("dilatedDLPFCspheres", cifti=True)
        similarity_img = get_spatial_correlation(
            smoothed_paths, dilstim_roi_2mm_path, out_path=similarity_path
        )

        similarity_data = similarity_img.get_fdata(dtype=np.float32)
        if threshold == 0:
            target_data = (similarity_data > threshold).astype(np.float32)
        else:
            target_data = (similarity_data >= threshold).astype(np.float32)
        target_img = ci.Cifti2Image(target_data, similarity_img.header)
        target_img.to_filename(target_mask_path)
    elif target_method == "reference":
        correlation_updates = dict(
            desc=f"{refroi_name}Corr", suffix="stat", space="fsLR", den="91k"
        )
        correlation_path = update_bidspath(
            smoothed_paths[0], out_dir, correlation_updates, exclude=["run"]
        )
        if stimroi_name in STIMROIS:
            dilstim_roi_2mm_path = get_stimroi_path("dilatedDLPFCspheres", cifti=True)
        else:
            dilstim_roi_2mm_path = stimroi_path
        correlation_img = get_correlation_map(
            smoothed_paths,
            dilstim_roi_2mm_path,
            refroi_path,
            out_path=correlation_path,
            invert_reference=True,
        )
        correlation_data = correlation_img.get_fdata(dtype=np.float32)
        if threshold == 0:
            target_data = (correlation_data > threshold).astype(np.float32)
        else:
            target_data = (correlation_data >= threshold).astype(np.float32)
        target_img = ci.Cifti2Image(target_data, correlation_img.header)
        target_img.to_filename(target_mask_path)
    elif target_method == "precomputed":
        if parse_bidsname(precomputed_path)['space'] == 'T1w':
            precomputed_mni_path = t1w_mask_to_mni(precomputed_path, fmriprep_dir, out_dir)
        # TODO: revisit this after fixing the naming bug in hierarchical to check the res of the precomputed path with
        # and (parse_bidsname(precomputed_path)['res'] == '2')
        elif (parse_bidsname(precomputed_path)['space'] == 'MNI152NLin6Asym') :
            precomputed_mni_path = precomputed_path
        else:
            raise ValueError(f"precomputed_path must have a parsebly bidsish name and be in T1w or MNI152NLin6Asym "
                             f"res-2 space. {precomputed_path} was received instead")
        target_mask_path = mask_to_cifti(precomputed_mni_path, precomputed_mni_path.parent)


    # build headmodel
    t1_paths = find_bids_files(
        fmriprep_dir / f"sub-{subject}", type="anat", suffix="T1w", extension=".nii.gz"
    )
    t1_paths = [tp for tp in t1_paths if "space" not in tp.parts[-1]]
    if len(t1_paths) > 1:
        raise ValueError(f"Looking for a single T1, found {len(t1_paths)}: {t1_paths}")
    elif len(t1_paths) == 0:
        t1_paths = find_bids_files(
            fmriprep_dir / f"sub-{subject}", ses="*", type="anat", suffix="T1w", extension=".nii.gz"
        )
        t1_paths = [tp for tp in t1_paths if "space" not in tp.parts[-1]]
        if len(t1_paths) > 1:
            raise ValueError(f"Looking for a single T1, found {len(t1_paths)}: {t1_paths}")
        elif len(t1_paths) == 0:
            find_bids_files(
                fmriprep_dir / f"sub-{subject}", debug=True, ses="*", type="anat", suffix="T1w", extension=".nii.gz"
            )
            raise ValueError(f"Couldn't find a T1.")
    t1w_path = t1_paths[0]

    t2_paths = find_bids_files(
        out_dir / f"sub-{subject}", type="anat", suffix="T2w", extension=".nii.gz"
    )
    t2_paths = [tp for tp in t2_paths if "space" not in tp.parts[-1]]
    if len(t2_paths) > 1:
        raise ValueError(f"Looking for a single T2, found {len(t2_paths)}: {t2_paths}")
    elif len(t2_paths) == 0:
        t2w_path=None
    else:
        t2w_path = t2_paths[0]

    os.environ["MPLBACKEND"] = "PDF"
    os.environ["OMP_NUM_THREADS"] = "2"
    if not noheadmodel:
        headmodels_script_path = write_headmodel_script(
            t1w_path,
            t2w_path,
            subses_out_dir,
            subject,
            msc_path,
            simnibs_path,
            tans_path,
        )
        headmodels_script_path = f"'{headmodels_script_path}'"

        cmd = [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-r",
            f"run({headmodels_script_path}); exit",
        ]
        print(f"running {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    if not nosimulation:
        # Write sim script and run it

        sim_script_path = write_sim_script(
            subses_out_dir,
            target_mask_path,
            stimroi_path,
            subses_out_dir,
            subject,
            msc_path=config["TANS"]["midnightscanclubpath"],
            simnibs_path=config["TANS"]["simnibspath"],
            tans_path=config["TANS"]["tanspath"],
            nthreads=nthreads,
        )
        sim_script_path = f"'{sim_script_path}'"
        cmd = [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-r",
            f"run({sim_script_path}); exit",
        ]
        print(f"running {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        # check to see if sim outputs are there
        sim_out_dir = subses_out_dir / 'SearchGrid'
        sim_res = sorted(sim_out_dir.glob('magnE_*.dtseries.nii'))
        if len(sim_res) == 0:
            raise FileNotFoundError(f"Couldn't find any simulation outputs of form magnE_*.dtseries.nii in {sim_out_dir}")

    if not nooptimization:
        # write optimization script and run it
        searchgrid_path = subses_out_dir / "SearchGrid/SubSampledSearchGrid.shape.gii"
        # just symlink the target mask path in the anat directory
        optimization_script_path = write_optimization_script(
            subses_out_dir,
            target_mask_path,
            searchgrid_path,
            subses_out_dir,
            subject,
            msc_path=config["TANS"]["midnightscanclubpath"],
            simnibs_path=config["TANS"]["simnibspath"],
            tans_path=config["TANS"]["tanspath"],
        )
        optimization_script_path = f"'{optimization_script_path}'"
        cmd = [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-r",
            f"run({optimization_script_path}); exit",
        ]
        print(f"running {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        # check to see if it worked
        expected_files = [
            'CoilCenter.foci',
            'CoilOrientationCoordinates.txt',
            'CoilCenter_OnTarget_s0.85.shape.gii',
            'CoilOrientation_OnTarget.shape.gii',
            'CoilOrientation_OnTarget_s0.85.shape.gii',
            'CoilCenter_OnTarget.shape.gii',
            'CoilOrientation.foci',
            'CoilCenter_OnTarget.mat',
            'CoilCenterCoordinates.txt'
        ]
        opt_out_dir = subses_out_dir / 'Optimize'
        for ef_name in expected_files:
            ef_path = opt_out_dir / ef_name
            if not ef_path.exists():
                raise FileNotFoundError("Optimize looks like it failed. Could not find {ef_path}.")
