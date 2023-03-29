import os
import shutil
from pathlib import Path
from pkg_resources import resource_filename
import pandas as pd
import numpy as np
import nilearn as nl
from nilearn import image

from fmriprep.workflows.bold.resampling import init_bold_surf_wf, init_bold_grayords_wf
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from contarg.tans import mcf_from_fmriprep_confounds, clean_bold, get_tans_inputs
from contarg.utils import (
    parse_bidsname,
    build_bidsname,
    make_rel_symlink,
    update_bidspath,
)


def pfm_inputs_from_fmriprep(
    subject,
    subjects_dir,
    out_dir,
    bold_path,
    t_r,
    n_dummy,
    confounds=None,
    aroma=False,
    overwrite=False,
):
    bold_path = Path(bold_path)

    # make standard space path
    mni_bold_path = bold_path.parent / bold_path.parts[-1].replace(
        "_space-T1w_", "_space-MNI152NLin6Asym_res-2_"
    )
    if not mni_bold_path.exists():
        raise ValueError(f"No MNI152NLin6Asym_res-2 image found for {bold_path}")

    # make MCF.par
    bold_parts = parse_bidsname(bold_path.parts[-1])
    for k in ["space", "res", "hemi"]:
        _ = bold_parts.pop(k, None)
    bold_parts["desc"] = "confounds"
    bold_parts["suffix"] = "timeseries"
    bold_parts["extension"] = "tsv"
    confounds_path = bold_path.parent / build_bidsname(bold_parts)

    bold_parts["desc"] = "MCF"
    bold_parts["suffix"] = "timeseries"
    bold_parts["extension"] = "par"
    mcf_path = out_dir / build_bidsname(bold_parts)

    mcf_from_fmriprep_confounds(confounds_path, mcf_path, n_dummy=n_dummy)

    # make path to t1w2fsnative_xfm
    t1w2fsnative_xfm = (
        bold_path.parent.parent
        / f"anat/sub-{subject}_from-T1w_to-fsnative_mode-image_xfm.txt"
    )
    if not t1w2fsnative_xfm.exists():
        t1w2fsnative_xfm = (
            bold_path.parent.parent.parent
            / f"anat/sub-{subject}_from-T1w_to-fsnative_mode-image_xfm.txt"
        )
    if not t1w2fsnative_xfm.exists():
        raise ValueError("Couldn't find the t1w to fsnative transformation.")

    # create cleaned bold series in T1w and MNI space, both are needed to make fsLR cifti
    clean_t1_path = clean_bold(
        bold_path,
        out_dir,
        n_dummy,
        t_r,
        confounds=confounds,
        aroma=aroma,
        overwrite=overwrite,
    )
    clean_mni_path = clean_bold(
        mni_bold_path,
        out_dir,
        n_dummy,
        t_r,
        confounds=confounds,
        aroma=aroma,
        overwrite=overwrite,
    )

    # set subjects dir
    os.environ["SUBJECTS_DIR"] = subjects_dir

    # generate output paths so you can check to see if they exist
    clean_parts = parse_bidsname(clean_t1_path.parts[-1])
    clean_parts["extension"] = "dtseries.nii"
    clean_parts["space"] = "32k_fs_LR"
    clean_cifti_path = out_dir / build_bidsname(clean_parts)
    clean_parts["extension"] = "dtseries.json"
    clean_json_path = out_dir / build_bidsname(clean_parts)

    if not overwrite and clean_cifti_path.exists():
        return clean_cifti_path

    # create fsaverage giftis from T1w space bold
    bold_surf_wf = init_bold_surf_wf(
        mem_gb=20, surface_spaces=["fsaverage"], medial_surface_nan=False
    )

    bold_surf_wf.inputs.inputnode.source_file = out_dir / clean_t1_path
    bold_surf_wf.inputs.inputnode.subject_id = f"sub-{subject}"
    bold_surf_wf.inputs.inputnode.subjects_dir = subjects_dir
    bold_surf_wf.inputs.inputnode.t1w2fsnative_xfm = t1w2fsnative_xfm.as_posix()
    if "echo" in bold_parts:
        wf_basedir = (
            out_dir
            / f"sub-{bold_parts['sub']}_task-{bold_parts['task']}_run-{bold_parts['run']}_echo-{bold_parts['echo']}"
        )
    else:
        wf_basedir = (
            out_dir
            / f"sub-{bold_parts['sub']}_task-{bold_parts['task']}_run-{bold_parts['run']}"
        )

    bold_surf_wf.base_dir = wf_basedir
    bold_surf_wf.run()

    lh = (
        wf_basedir
        / "bold_surf_wf/_target_fsaverage/update_metadata/mapflow/_update_metadata0/lh.fsaverage.gii"
    )
    rh = (
        wf_basedir
        / "bold_surf_wf/_target_fsaverage/update_metadata/mapflow/_update_metadata1/rh.fsaverage.gii"
    )

    # create fsLR ciftis
    bold_grayords_wf = init_bold_grayords_wf(
        grayord_density="91k",
        mem_gb=20,
        repetition_time=t_r,
    )

    bold_grayords_wf.inputs.inputnode.spatial_reference = ["MNI152NLin6Asym_res-2"]
    bold_grayords_wf.inputs.inputnode.bold_std = [clean_mni_path]
    bold_grayords_wf.inputs.inputnode.surf_refs = ["fsaverage"]
    bold_grayords_wf.inputs.inputnode.surf_files = [[lh, rh]]
    bold_grayords_wf.base_dir = wf_basedir

    bold_grayords_wf.run()

    # TODO: replace with a proper io node

    grayords_nii_out = list(
        (wf_basedir / "bold_grayords_wf/gen_cifti").glob("*.dtseries.nii")
    )[0]
    grayords_json_out = list(
        (wf_basedir / "bold_grayords_wf/gen_cifti").glob("*.dtseries.json")
    )[0]

    shutil.copyfile(grayords_nii_out, clean_cifti_path)
    shutil.copyfile(grayords_json_out, clean_json_path)

    return clean_cifti_path


def pfm_inputs_from_tedana(
    boldtd_path,
    fmriprep_dir,
    tedana_dir,
    out_dir,
    aroma,
    n_dummy,
    t_r,
    nthreads,
    noinputprep=False,
    overwrite=False,
    drop_rundir=True,
    ciftis_out=True,
    regress_gm=True,
    # if the following aren't defined, they'll be assumed to be in default locations
    subjects_dir=None,
    scanner_to_t1w_path=None,
    t1w_to_scanner_path=None,
    t1w_to_MNI152NLin6Asym_path=None,
    t1w_to_fsnative_path=None,
    gmseg_path=None,
    confounds_path=None,
    boldmask_path=None,
    t2starmap_path=None,
    boldref_t1_path=None,
    boldref_MNI152NLin6Asym_path=None,
):
    boldtd_path = Path(boldtd_path)

    if drop_rundir:
        if 'run' in boldtd_path.parts[-1]:
            boldtd_path_for_building = boldtd_path.parent.parent / boldtd_path.parts[-1]
            ents = parse_bidsname(boldtd_path_for_building)
    else:
        ents = parse_bidsname(boldtd_path)

    # find the root of the tedana dir
    tedana_root = []
    for pp in boldtd_path.parts[:-1]:
        if not pp.split("-")[-1] in ents.values():
            tedana_root.append(pp)
    tedana_root = Path(*tedana_root)

    # set the subjects dir
    if subjects_dir is None:
        subjects_dir = fmriprep_dir / "sourcedata/freesurfer"

    if not subjects_dir.exists():
        raise FileNotFoundError("Could not find subjects dir at {subjects_dir}")

    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()

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

    if t1w_to_scanner_path is None:
        t1w_to_scanner_ents = {
            "suffix": "xfm",
            "extension": "txt",
            "from": "T1w",
            "to": "scanner",
            "mode": "image",
        }
        t1w_to_scanner_path = update_bidspath(
            boldtd_path_for_building,
            fmriprep_dir,
            t1w_to_scanner_ents,
            exclude=["desc"],
            exists=True,
        )

    if t1w_to_fsnative_path is None:
        t1w_to_fsnative_ents = {
            "suffix": "xfm",
            "extension": "txt",
            "from": "T1w",
            "to": "fsnative",
            "mode": "image",
            "type": "anat",
        }
        t1w_to_fsnative_path = update_bidspath(
            boldtd_path_for_building,
            fmriprep_dir,
            t1w_to_fsnative_ents,
            exclude=["desc", "ses", "task", "acq", "run"],
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
            exists=True,
        )

    if gmseg_path is None:
        gmseg_ents = dict(
            label="GM", suffix="probseg", type="anat", extension=".nii.gz"
        )
        gmseg_path = update_bidspath(
            boldtd_path_for_building,
            fmriprep_dir,
            gmseg_ents,
            exclude=["ses", "task", "acq", "run", "desc"],
            exists=True,
        )

    if boldmask_path is None:
        boldmask_ents = dict(
            desc="brain",
            suffix="mask",
        )
        boldmask_path = update_bidspath(
            boldtd_path_for_building, fmriprep_dir, boldmask_ents, exists=True
        )

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

    if confounds_path is None:
        confounds_ents = dict(desc="confounds", suffix="timeseries", extension="tsv")
        confounds_path = update_bidspath(
            boldtd_path_for_building, fmriprep_dir, confounds_ents, exists=True
        )

    if t2starmap_path is None:
        t2starmap_ents = dict(suffix="T2starmap", extension=".nii.gz")
        t2starmap_path = update_bidspath(
            boldtd_path, tedana_root, t2starmap_ents, exclude="desc", exists=True, keep_rundir=True
        )

    # output paths
    cleaned_boldtd_ents = dict(desc=ents["desc"] + "MGTR")
    cleaned_boldtd_path = update_bidspath(boldtd_path_for_building, out_dir, cleaned_boldtd_ents)
    cleaned_boldtd_path.parent.mkdir(exist_ok=True, parents=True)

    scanner_gmseg_ents = dict(space="scanner", label="GM", suffix="probseg")
    scanner_gmseg_path = update_bidspath(
        boldtd_path_for_building, out_dir, scanner_gmseg_ents, exclude="desc"
    )
    scanner_gmseg_path.parent.mkdir(exist_ok=True, parents=True)

    cleaned_T1wboldtd_ents = dict(space="T1w")
    cleaned_T1wboldtd_path = update_bidspath(
        cleaned_boldtd_path, out_dir, cleaned_T1wboldtd_ents
    )

    cleaned_MNIboldtd_ents = dict(space="MNI152NLin6Asym", res="2")
    cleaned_MNIboldtd_path = update_bidspath(
        cleaned_boldtd_path, out_dir, cleaned_MNIboldtd_ents
    )

    wf_basedir_path = Path(
        update_bidspath(
            boldtd_path_for_building,
            out_dir,
            dict(desc="workflows"),
            exclude=["suffix", "extension"],
        ).as_posix()[:-1]
    )

    cleaned_boldtdnii_ents = dict(extension="dtseries.nii", space="fsLR", den="91k")
    cleaned_boldtdnii_path = update_bidspath(
        cleaned_boldtd_path, out_dir, cleaned_boldtdnii_ents
    )
    if noinputprep:
        return cleaned_boldtdnii_path

    if not overwrite and cleaned_boldtdnii_path.exists():
        return cleaned_boldtdnii_path

    # write MCF file
    mcf_ents = dict(extension="par", suffix="timeseries", desc="MCF")
    mcf_path = update_bidspath(cleaned_boldtd_path, out_dir, mcf_ents)
    mcf_from_fmriprep_confounds(confounds_path, mcf_path, n_dummy=n_dummy)

    # transform gmseg to native space
    at = ApplyTransforms()
    at.inputs.input_image = gmseg_path
    at.inputs.reference_image = t2starmap_path
    at.inputs.transforms = [t1w_to_scanner_path]
    at.inputs.interpolation = "LanczosWindowedSinc"
    at.inputs.output_image = scanner_gmseg_path.as_posix()
    at.inputs.float = True
    _ = at.run()

    # get mean gm timeseries
    scanner_gmseg = nl.image.load_img(scanner_gmseg_path)
    boldtd = nl.image.load_img(boldtd_path)
    gmseg_dat = np.expand_dims(scanner_gmseg.get_fdata(), -1)
    gm_data = boldtd.get_fdata() * gmseg_dat
    gm_timeseries = gm_data[(gmseg_dat != 0).squeeze()].mean(0)

    # TODO: add drift terms
    # add mean gm to timeseries
    confounds = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "global_signal",
    ]

    cfds = pd.read_csv(confounds_path, sep="\t")
    confound_names = confounds.copy()
    if aroma:
        confound_names += [nn for nn in cfds.columns if "aroma" in nn]
    cfds_to_use = cfds.loc[n_dummy:, confound_names].copy()

    # if boldtd has already had dummy scans dropped, this will produce a ValueError, set n_dummy to 0 going forward
    try:
        cfds_to_use["gm"] = gm_timeseries[n_dummy:]
    except ValueError:
        cfds_to_use["gm"] = gm_timeseries
        n_dummy = 0

    if not regress_gm:
        cfds_to_use = cfds_to_use.drop('gm', axis=1)

    # clean boldtd
    cleaned = nl.image.clean_img(
        nl.image.load_img(boldtd_path).slicer[:, :, :, n_dummy:],
        confounds=cfds_to_use,
        high_pass=0.01,
        low_pass=0.1,
        mask_img=nl.image.load_img(boldmask_path),
        t_r=t_r,
    )

    cleaned.to_filename(cleaned_boldtd_path)

    # transform cleaned from native space to T1w space
    at = ApplyTransforms(interpolation="LanczosWindowedSinc", float=True)
    at.inputs.num_threads = nthreads
    at.inputs.input_image = cleaned_boldtd_path
    at.inputs.output_image = cleaned_T1wboldtd_path.as_posix()
    at.inputs.reference_image = boldref_t1_path
    at.inputs.input_image_type = 3
    at.inputs.transforms = [scanner_to_t1w_path]
    _ = at.run()

    # transform cleaned from native space to MNI152NLin6Asym space
    at = ApplyTransforms(interpolation="LanczosWindowedSinc", float=True)
    at.inputs.num_threads = nthreads
    at.inputs.input_image = cleaned_boldtd_path
    at.inputs.output_image = cleaned_MNIboldtd_path.as_posix()
    at.inputs.reference_image = boldref_MNI152NLin6Asym_path
    at.inputs.input_image_type = 3
    at.inputs.transforms = [t1w_to_MNI152NLin6Asym_path, scanner_to_t1w_path]
    _ = at.run()

    if ciftis_out:
        # create fsaverage giftis from T1w space bold
        bold_surf_wf = init_bold_surf_wf(
            mem_gb=20, surface_spaces=["fsaverage"], medial_surface_nan=False
        )

        bold_surf_wf.inputs.inputnode.source_file = cleaned_T1wboldtd_path
        bold_surf_wf.inputs.inputnode.subject_id = f'sub-{ents["sub"]}'
        bold_surf_wf.inputs.inputnode.subjects_dir = subjects_dir
        bold_surf_wf.inputs.inputnode.t1w2fsnative_xfm = t1w_to_fsnative_path.as_posix()

        bold_surf_wf.base_dir = wf_basedir_path
        bold_surf_wf.run()

        lh = (
            wf_basedir_path
            / "bold_surf_wf/_target_fsaverage/update_metadata/mapflow/_update_metadata0/lh.fsaverage.gii"
        )
        rh = (
            wf_basedir_path
            / "bold_surf_wf/_target_fsaverage/update_metadata/mapflow/_update_metadata1/rh.fsaverage.gii"
        )

        # create fsLR ciftis
        bold_grayords_wf = init_bold_grayords_wf(
            grayord_density="91k",
            mem_gb=20,
            repetition_time=t_r,
        )

        bold_grayords_wf.inputs.inputnode.spatial_reference = ["MNI152NLin6Asym_res-2"]
        bold_grayords_wf.inputs.inputnode.bold_std = [cleaned_MNIboldtd_path]
        bold_grayords_wf.inputs.inputnode.surf_refs = ["fsaverage"]
        bold_grayords_wf.inputs.inputnode.surf_files = [[lh, rh]]
        bold_grayords_wf.base_dir = wf_basedir_path

        bold_grayords_wf.run()

        # TODO: replace with a proper io node
        grayords_nii_out = list(
            (wf_basedir_path / "bold_grayords_wf/gen_cifti").glob("*.dtseries.nii")
        )[0]
        grayords_json_out = list(
            (wf_basedir_path / "bold_grayords_wf/gen_cifti").glob("*.dtseries.json")
        )[0]

        shutil.copyfile(grayords_nii_out, cleaned_boldtdnii_path)
        shutil.copyfile(
            grayords_json_out, cleaned_boldtdnii_path.as_posix().replace(".nii", ".json")
        )

        return cleaned_boldtdnii_path
    else:
        return cleaned_T1wboldtd_path, cleaned_MNIboldtd_path


def build_pfm_inputdir(subdir, t_r, tedana=False):
    subdir = Path(subdir)
    sfdir = subdir / "func"
    if tedana:
        desc = "optcomDenoisedMGTR"
    else:
        desc = "cleaned"
    # we're only going to deal with single sessions at the moment
    # so we just need all the runs for a session
    # and the pfm Subdir will just be nested within sessions if they exist
    run_ciftis = sorted(sfdir.glob(f"*_desc-{desc}_bold.dtseries.nii"))
    if len(run_ciftis) == 0:
        raise ValueError(
            f"Didn't find any cleaned series in with this glob '*_desc-{desc}_bold.dtseries.nii' in  {sfdir}"
        )
    run_jsons = sorted(sfdir.glob(f"*_desc-{desc}_bold.dtseries.json"))
    run_mcfs = sorted(sfdir.glob(f"*_desc-MCF_timeseries.par"))
    # make sure we've got the same number for all of them
    if len(run_ciftis) != len(run_jsons):
        raise ValueError(
            f"Found {len(run_ciftis)} ciftis but only {len(run_jsons)} jsons in {sfdir}"
        )
    if len(run_ciftis) != len(run_mcfs):
        raise ValueError(
            f"Found {len(run_ciftis)} ciftis but only {len(run_mcfs)} mcf.par files in {sfdir}"
        )

    # check that run_ciftis and run_mcfds go together:
    run_ids = []
    for cifti, mcf, meta in zip(run_ciftis, run_mcfs, run_jsons):
        cifti_ents = parse_bidsname(cifti.parts[-1])
        mcf_ents = parse_bidsname(mcf.parts[-1])
        meta_ents = parse_bidsname(meta.parts[-1])

        for mcfe in mcf_ents.keys():
            if not mcfe in ["desc", "extension", "suffix"]:
                if mcf_ents[mcfe] != cifti_ents[mcfe]:
                    raise ValueError(
                        f"{mcf} has {mcfe} = {mcf_ents[mcfe]}, but apparently corresponding cifti {cifti} has {mcfe} = {cifti_ents[mcfe]}"
                    )
                if mcf_ents[mcfe] != meta_ents[mcfe]:
                    raise ValueError(
                        f"{mcf} has {mcfe} = {mcf_ents[mcfe]}, but apparently corresponding json {meta} has {mcfe} = {meta_ents[mcfe]}"
                    )
        run_ids.append(cifti_ents["run"])
    nruns = len(run_ciftis)

    # make session dir and create symlinks
    pfm_indir = subdir / "pfm_subdir/func/rest"
    session_dir = pfm_indir / "session_1"
    session_dir.mkdir(exist_ok=True, parents=True)
    for cifti, mcf, meta in zip(run_ciftis, run_mcfs, run_jsons):
        cifti_ents = parse_bidsname(cifti.parts[-1])
        run_id = cifti_ents["run"]
        run_dir = session_dir / f"run_{int(run_id)}"
        run_dir.mkdir(exist_ok=True)

        pfm_cifti_path = run_dir / "Rest_clean.dtseries.nii"
        pfm_mcf_path = run_dir / "MCF.par"
        pfm_meta_path = run_dir / "Rest_clean.dtseries.json"
        pfm_tr_path = run_dir / "TR.txt"

        make_rel_symlink(pfm_cifti_path, cifti)
        make_rel_symlink(pfm_mcf_path, mcf)
        make_rel_symlink(pfm_meta_path, meta)

        pfm_tr_path.write_text(f"{t_r}")

    return pfm_indir.parent.parent


def write_pfm_script(
    pfm_input_dir,
    tans_input_dir,
    out_dir,
    subject,
    msc_path=None,
    tans_path=None,
    nthreads=20,
):
    """
    Generate a Matlab script that runs the PFM (precision functional mapping) pipeline for a given set of input data.

    Parameters
    ----------
    pfm_input_dir : str
        Path to the directory containing the input resting-state fMRI data.
    tans_input_dir : str
        Path to the directory containing the input TANS data.
    out_dir : str
        Path to the output directory for the PFM pipeline.
    msc_path : str, optional
        Path to the MidnightScanClub MSCcodebase folder containing ft_read / gifti functions for reading and writing cifti files (default: None).
    tans_path : str, optional
        Path to the TANS package containing the resources for running the PFM pipeline (default: None).
    nthreads : int, optional
        The number of threads to use when running the PFM pipeline (default: 20).

    Returns
    -------
    pathlib.Path
        Path to the MATLAB script generated by this function.

    Notes
    -----
    This function generates a Matlab script that can be run using the Matlab software to perform the PFM pipeline on a given set of input data. The generated script assumes that the necessary software and dependencies are installed and available on the system. The script performs the following steps:
    - Reads in the input resting-state fMRI data
    - Concatenates and smooths the resting-state fMRI data
    - Makes a distance matrix and then regresses adjacent cortical signals from subcortical voxels
    - Sweeps a range of smoothing kernels
    - Runs precision mapping
    """

    out_dir = Path(out_dir)
    m_dir = Path(tans_input_dir / "matlab_scripts")
    m_dir.mkdir(exist_ok=True)

    lmidthicksurf = (
        f"{tans_input_dir}/anat/sub-{subject}.L.midthickness.32k_fs_LR.surf.gii"
    )
    if not Path(lmidthicksurf).exists():
        raise ValueError(f"{lmidthicksurf} does not exist")
    rmidthicksurf = (
        f"{tans_input_dir}/anat/sub-{subject}.R.midthickness.32k_fs_LR.surf.gii"
    )
    if not Path(rmidthicksurf).exists():
        raise ValueError(f"{rmidthicksurf} does not exist")

    pfm_script_path = m_dir / "run_pfm.m"
    if msc_path is None:
        msc_path = os.getenv("CONTARG_MSC_PATH")
    if tans_path is None:
        tans_path = resource_filename(
            "contarg", "resources/Targeted-Functional-Network-Stimulation"
        )

    pfm_script = f"""
    %% Example use of Targeted Functional Network Stimulation ("TANS")
    % "Automated optimization of TMS coil placement for personalized functional
    % network engagement" - Lynch et al., 2022 (Neuron)

    % define some paths
    Paths{{1}} = '{msc_path}'; % this folder contains ft_read / gifti functions for reading and writing cifti files (e.g., https://github.com/MidnightScanClub/MSCcodebase).
    Paths{{2}} = '{tans_path}'; %

    % add folders
    % to search path;
    for i = 1:length(Paths)
        addpath(genpath(Paths{{i}}));
    end

    % clear matlab's ldlibrary path
    tmp_path = getenv('LD_LIBRARY_PATH');
    setenv('LD_LIBRARY_PATH', '');

    % If successful, each of the commands below should return status as 0.

    % Confirm various software is available
    [status,~] = system('mris_convert -version'); % freesurfer
    assert(status == 0, 'Freesurfer not available')
    [status,~] = system('wb_command -version'); % connectome workbench
    assert(status == 0, 'connectome workbench not available')
    [status,~] = system('flirt -version'); % fsl
    assert(status == 0, 'FSL not available')
    [status,~] = system('infomap -h'); % infomap
    assert(status == 0, 'Infomap not available')

    % If successful, each of the commands below should return status as 2.

    % Confirm that functions for reading and writing
    % CIFTI and GIFTI files are also available
    status = exist('ft_read_cifti_mod','file');
    assert(status == 2, 'ft_read_cifti_mod not available')
    status = exist('gifti','file');
    assert(status == 2, 'gifti not available')

    Subdir = '{pfm_input_dir}';

    MidthickSurfs{{1}} = '{lmidthicksurf}';
    MidthickSurfs{{2}} = '{rmidthicksurf}';

    % denoising QC;
    % grayplot_qa_func(Subdir);

    % concatenate and smooth resting-state fMRI datasets;
    nSessions = length(dir([Subdir '/func/rest/session_*']));
    [C,ScanIdx,FD] = concatenate_scans(Subdir,'Rest_clean',1:nSessions);
    mkdir([Subdir '/ConcatenatedCiftis']);
    cd([Subdir '/ConcatenatedCiftis']);

    % make distance matrix and then regress
    % adjacent cortical signals from subcortical voxels;
    make_distance_matrix(C,MidthickSurfs,'{out_dir}',{nthreads});
    [C] = regress_cortical_signals(C,['{out_dir}', '/DistanceMatrix.mat'],{nthreads});
    ft_write_cifti_mod([Subdir '/ConcatenatedCiftis/Rest_clean_Concatenated+SubcortRegression.dtseries.nii'],C);
    save([Subdir '/ConcatenatedCiftis/ScanIdx'],'ScanIdx');
    save([Subdir '/ConcatenatedCiftis/FD'],'FD');
    clear C % clear intermediate file

    % sweep a range of
    % smoothing kernels;
    for k = [0.85 1.7 2.55]
        system(['wb_command -cifti-smoothing ' [Subdir '/ConcatenatedCiftis/Rest_clean_Concatenated+SubcortRegression.dtseries.nii'] ' ' num2str(k) ' ' num2str(k) ' COLUMN ' [Subdir '/ConcatenatedCiftis/Rest_clean_Concatenated+SubcortRegression+SpatialSmoothing' num2str(k) '.dtseries.nii'] ' -left-surface ' MidthickSurfs{{1}} ' -right-surface ' MidthickSurfs{{2}} ' -merged-volume']);
    end

    % load your concatenated resting-state dataset, here we selected the highest level of spatial smoothing
    C = ft_read_cifti_mod([Subdir '/ConcatenatedCiftis/Rest_clean_Concatenated+SubcortRegression+SpatialSmoothing2.55.dtseries.nii']);
    C.data(:,FD<0.3)=[]; % remove high motion volumes

    % run precision mapping;
    pfm(C,['{out_dir}' '/DistanceMatrix.mat'],['{out_dir}' '/pfm/'],flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]),[1 5 10 50 50 50 50 50 50],10,[],{{'CORTEX_LEFT','CEREBELLUM_LEFT','ACCUMBENS_LEFT','CAUDATE_LEFT','PALLIDUM_LEFT','PUTAMEN_LEFT','THALAMUS_LEFT','HIPPOCAMPUS_LEFT','AMYGDALA_LEFT','ACCUMBENS_LEFT','CORTEX_RIGHT','CEREBELLUM_RIGHT','ACCUMBENS_RIGHT','CAUDATE_RIGHT','PALLIDUM_RIGHT','PUTAMEN_RIGHT','THALAMUS_RIGHT','HIPPOCAMPUS_RIGHT','AMYGDALA_RIGHT','ACCUMBENS_RIGHT'}},{nthreads},Paths);
    spatial_filtering(['{out_dir}' '/pfm/Bipartite_PhysicalCommunities.dtseries.nii'],['{out_dir}' '/pfm/'],['Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii'],MidthickSurfs, 50, 50);

    % remove some intermediate files;
    system(['rm {out_dir}/pfm/*.net']);
    system(['rm {out_dir}/pfm/*.clu']);
    system(['rm {out_dir}/pfm/*Log*']);   
    setenv('LD_LIBRARY_PATH', tmp_path);
    exit
    """

    pfm_script_path.write_text(pfm_script)

    # copy functional network excel sheet to pfm output directory
    fnxl = Path(resource_filename("contarg", "data/FunctionalNetworks_template.xlsx"))
    out_dir_pfm = out_dir / "pfm"
    out_dir_pfm.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(fnxl, out_dir_pfm / "FunctionalNetworks_template.xlsx")

    return pfm_script_path
