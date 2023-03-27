import click
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import nilearn as nl
from nilearn import image
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from fmriprep.workflows.bold.resampling import init_bold_surf_wf, init_bold_grayords_wf
from contarg.utils import (
    parse_bidsname,
    build_bidsname,
    update_bidspath,
    find_bids_files,
)
from contarg.tans import tans_inputs_from_fmriprep
from contarg.pfm import (
    build_pfm_inputdir,
    write_pfm_script,
    pfm_inputs_from_tedana,
    pfm_inputs_from_fmriprep,
)
import subprocess
from pkg_resources import resource_filename
import configparser


@click.group()
def contarg():
    pass


@contarg.group()
def pfm():
    pass


@pfm.command()
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
@click.option("--subject", type=str, default=None, help="Subject id.", required=True)
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
    "--overwrite",
    is_flag=True,
    help="Overwrite existing outputs",
)
def run(
    fmriprep_dir,
    tedana_dir,
    subjects_dir,
    out_dir,
    aroma,
    n_dummy,
    t_r,
    subject,
    session,
    acquisition,
    run,
    echo,
    nthreads,
    overwrite,
):
    fmriprep_dir = Path(fmriprep_dir)
    tedana = False
    if tedana_dir is not None:
        tedana = True
        tedana_dir = Path(tedana_dir)
        if not tedana_dir.exists():
            raise FileNotFoundError(tedana_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # transform anatomicals
    tans_inputs_from_fmriprep(
        fmriprep_dir, out_dir, subject=subject, overwrite=overwrite
    )

    bold_paths = []
    # find the runs to use and create necessary inputs
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
            pfm_inputs_from_tedana(
                boldtd_path,
                fmriprep_dir,
                tedana_dir,
                out_dir,
                aroma,
                n_dummy,
                t_r,
                nthreads,
                overwrite=overwrite,
            )
            # TODO: finish wiring up overwrite for everything below here
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
                        desc="optcomDenoised",
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
                    desc="optcomDenoised",
                    suffix="bold",
                    extension=".nii.gz",
                )
            )

        # set the subjects dir
        if subjects_dir is None:
            subjects_dir = fmriprep_dir / "sourcedata/freesurfer"

        if not subjects_dir.exists():
            raise FileNotFoundError("Could not find subjects dir at {subjects_dir}")

        for bold_path in bold_paths:
            # TODO: fix pfm_inputs_from_fmriprep so that it uses the grey matter signal and not the global
            confounds = [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "global_signal",
            ]
            pfm_inputs_from_fmriprep(
                subject,
                subjects_dir,
                out_dir,
                bold_path,
                t_r,
                n_dummy,
                confounds=confounds,
                aroma=aroma,
            )

    if session is not None:
        pfm_indir_parent = out_dir / f"sub-{subject}/ses-{session}"
    else:
        pfm_indir_parent = out_dir / f"sub-{subject}/"

    pfm_indir = build_pfm_inputdir(pfm_indir_parent, t_r, tedana=tedana)

    pfm_out = pfm_indir_parent / "pfm_output"
    pfm_out.mkdir(exist_ok=True, parents=True)

    # get paths from config
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

    pfm_script_path = write_pfm_script(
        pfm_indir,
        out_dir / f"sub-{subject}/",
        pfm_out,
        subject,
        msc_path=msc_path,
        tans_path=tans_path,
        nthreads=nthreads,
    )
    pfm_script_path = f"'{pfm_script_path}'"

    cmd = [
        "matlab",
        "-nodisplay",
        "-nosplash",
        "-r",
        f"run({pfm_script_path}); exit",
    ]
    print(f"running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
