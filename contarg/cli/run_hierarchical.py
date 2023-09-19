from pathlib import Path
from bids import BIDSLayout  # pip version = 0.15.2
import numpy as np
import pandas as pd
import nilearn as nl
from nilearn import image, masking
from joblib import Parallel, delayed
import click
from contarg.utils import (
    make_path,
    transform_mask_to_t1w,
    clean_mask,
    get_stimroi_path,
    get_refroi_path,
    update_bidspath,
    parse_bidsname,
    select_confounds,
    add_censor_columns
)
from contarg.hierarchical import (
    find_target,
    custom_metric,
    rank_clusters,
    get_com_in_mm,
    get_clust_image,
    get_vox_ref_corr,
    prep_tedana_for_hierarchical,
    block_bootstrap
)


@click.group()
def contarg():
    pass


@contarg.group()
def hierarchical():
    pass


@hierarchical.command()
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
    "--fmriprep-dir",
    type=click.Path(exists=True),
    help="Path to fmriprep outputs, if not given, they are assumed to be in derivatives_dir / fmriprep",
    default=None
)
@click.option(
    "--tedana-dir",
    type=click.Path(),
    default=None,
    help="Path to TEDANA outputs",
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
    "--stimroi-name",
    type=str,
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
    "--linkage",
    type=click.Choice(["single", "complete", "average"]),
    default="average",
    show_default=True,
    help="Linkage type for clustering",
)
@click.option(
    "--adjacency/--no-adjacency",
    default=False,
    show_default=True,
    help="Whether or not to constrain clustering by face adjacency",
)
@click.option(
    "--distance-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum spearman correlation required to cluster",
)
@click.option(
    "--concat",
    is_flag=True,
    help="If true, concatenate all the runs found with the subject, session, and acquistion values passed",
)
@click.option(
    "--nvox_weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight given to number of voxels when selecting a target cluster",
)
@click.option(
    "--concentration-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight given to concentration when selecting a target cluster",
)
@click.option(
    "--nrc-weight",
    "net_reference_correlation_weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight given to net reference correlation when selecting a target cluster",
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
    "--acquisition", type=str, default=None, help="Acquisition from dataset to generate target(s) for."
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
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing outputs",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Don't use joblib to facilitate debugging",
)
def run(
    bids_dir,
    derivatives_dir,
    database_file,
    fmriprep_dir,
    tedana_dir,
    run_name,
    stimroi_name,
    stimroi_path,
    refroi_name,
    refroi_path,
    space,
    smoothing_fwhm,
    n_dummy,
    t_r,
    linkage,
    adjacency,
    distance_threshold,
    concat,
    nvox_weight,
    concentration_weight,
    net_reference_correlation_weight,
    max_outfrac,
    max_fd,
    frames_before,
    frames_after,
    minimum_segment_length,
    minimum_total_length,
    subject,
    session,
    run,
    acquisition,
    echo,
    njobs,
    overwrite,
    debug
):
    bids_dir = Path(bids_dir)
    derivatives_dir = Path(derivatives_dir)
    database_path = Path(database_file)
    if fmriprep_dir is None:
        fmriprep_dir = derivatives_dir / 'fmriprep'
    fmriprep_dir = Path(fmriprep_dir)
    if not fmriprep_dir.exists():
        raise FileNotFoundError(fmriprep_dir)
    if tedana_dir is not None:
        tedana = True
        tedana_dir = Path(tedana_dir)
        if not tedana_dir.exists():
            raise FileNotFoundError(tedana_dir)
        derivatives = [fmriprep_dir, tedana_dir]
    else:
        derivatives = fmriprep_dir
    layout = BIDSLayout(
        bids_dir,
        database_path=database_path,
        derivatives=derivatives,
    )
    if run_name is not None:
        targeting_dir = derivatives_dir / "contarg" / "hierarchical" / run_name
    else:
        targeting_dir = derivatives_dir / "contarg" / "hierarchical"
    targeting_dir.mkdir(parents=True, exist_ok=True)

    stim_roi_2mm_path = get_stimroi_path(stimroi_name, stimroi_path)
    ref_roi_2mm_path = get_refroi_path(refroi_name, refroi_path)

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
    if acquisition is not None:
        get_kwargs["acquisition"] = acquisition
    if tedana:
        get_kwargs["desc"] = "optcomDenoised"
        get_kwargs.pop("space")
    bolds = layout.get(**get_kwargs)

    # get a run number to use for things like the bold ref
    dummy_run_number = parse_bidsname(bolds[0])['run']

    counfound_paths = None
    if tedana:
        boldtd_paths = [bb.path for bb in bolds]
        bold_paths = []
        confound_paths = []
        for boldtd_path in boldtd_paths:
            T1wboldtd_path, MNIboldtd_path, trunc_confounds_path = prep_tedana_for_hierarchical(
                boldtd_path,
                fmriprep_dir,
                tedana_dir,
                targeting_dir,
                t_r,
                n_dummy,
                njobs,
                overwrite=overwrite,
                max_outfrac=max_outfrac, max_fd=max_fd,
                frames_before=frames_before, frames_after=frames_after,
                minimum_segment_length=minimum_segment_length, minimum_total_length=minimum_total_length,
            )
            if space == 'T1w':
                bold_paths.append(T1wboldtd_path)
            elif space == 'MNI152NLin6Asym':
                bold_paths.append(MNIboldtd_path)
            confound_paths.append(trunc_confounds_path)
        # set n_dummy to 0 as these have been dropped now for the tedana images and confounds
        n_dummy = 0
    else:
        bold_paths = [bb.path for bb in bolds]
        for bold_path in bold_paths:
            confounds_ents = dict(desc="confounds", suffix="timeseries", extension="tsv")
            confounds_path = update_bidspath(
                bold_path, fmriprep_dir, confounds_ents, exists=True
            )


    if concat:
        bolds = []
        clean_concat_path = update_bidspath(bold_paths[0], targeting_dir, updates={}, exclude='run')
        if not overwrite and clean_concat_path.exists():
            pass
        else:
            clean_bold_imgs = [nl.image.load_img(cbi) for cbi in bold_paths]
            frames = []
            for img in clean_bold_imgs:
                frames.extend([i for i in nl.image.iter_img(img)])
            clean_concat_img = nl.image.concat_imgs(frames, ensure_ndim=4)
            clean_concat_img.to_filename(clean_concat_path)
        bolds.append(clean_concat_path)
    else:
        bolds = bold_paths

    rest_paths = pd.DataFrame([layout.parse_file_entities(bb) for bb in bolds])
    rest_paths["entities"] = [layout.parse_file_entities(bb) for bb in bolds]
    rest_paths["bold_path"] = bolds
    rest_paths["confounds"] = None
    if "session" not in rest_paths.columns:
        rest_paths["session"] = None
    # add boldref
    rest_paths["boldref"] = rest_paths.entities.apply(
        lambda ee: layout.get(
            return_type="file",
            subject=ee["subject"],
            session=ee["session"],
            task=ee["task"],
            run=ee.get("run", dummy_run_number),
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

    # define a pattern for building paths
    boldmask_pattern = "sub-{subject}/[ses-{session}/]func/sub-{subject}_[ses-{session}_]task-{task}_[run-{run}_][echo-{echo}_]atlas-{atlas}_space-{space}_desc-{desc}_{suffix}.{extension}"

    if space == "T1w":
        # set up paths to transform stim and target rois back to subject space

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
            "desc": refroi_name,
            "suffix": "mask",
            "space": space,
            "atlas": "Coords",
        }
        rest_paths[f"ref_mask"] = rest_paths.entities.apply(
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
        rest_paths["ref_mask_exists"] = rest_paths.ref_mask.apply(lambda x: x.exists())

        # set up paths for masks with small connected components dropped
        clnstimmask_updates = {
            "desc": stimroi_name + "Clean",
            "suffix": "mask",
            "space": space,
            "atlas": "Coords",
        }
        rest_paths[f"clnstim_mask"] = rest_paths.entities.apply(
            lambda x: make_path(
                x,
                clnstimmask_updates,
                boldmask_pattern,
                targeting_dir,
                layout.build_path,
                check_exist=False,
                check_parent=False,
                make_parent=True,
            )
        )
        rest_paths["clnstim_mask_exists"] = rest_paths.clnstim_mask.apply(
            lambda x: x.exists()
        )

        clnrefmask_updates = {
            "desc": refroi_name + "Clean",
            "suffix": "mask",
            "space": space,
            "atlas": "Coords",
        }
        rest_paths[f"clnref_mask"] = rest_paths.entities.apply(
            lambda x: make_path(
                x,
                clnrefmask_updates,
                boldmask_pattern,
                targeting_dir,
                layout.build_path,
                check_exist=False,
                check_parent=False,
                make_parent=True,
            )
        )
        rest_paths["clnref_mask_exists"] = rest_paths.clnref_mask.apply(
            lambda x: x.exists()
        )

    else:
        # TODO: clean this up for custom ROIs
        clnstm_roi_2mm_path = get_stimroi_path(stimroi_name, stimroi_path=stimroi_path, masked=True)
        rest_paths[f"stim_mask"] = stim_roi_2mm_path
        rest_paths["stim_mask_exists"] = True
        rest_paths["ref_mask"] = ref_roi_2mm_path
        rest_paths["ref_mask_exists"] = True
        rest_paths[f"clnstim_mask"] = clnstm_roi_2mm_path
        rest_paths["clnstim_mask_exists"] = True
        rest_paths["clnref_mask"] = ref_roi_2mm_path
        rest_paths["clnref_mask_exists"] = True

    if adjacency:
        desc = f"dt{distance_threshold}.ad.custaff"
    else:
        desc = f"dt{distance_threshold}.custaff"

    clust_updates = {
        "desc": desc,
        "suffix": "targetclust",
        "space": space,
        "atlas": "Coords",
        "extension": ".pkl.gz",
    }
    rest_paths[f"{desc}_targout"] = rest_paths.entities.apply(
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
    rest_paths[f"{desc}_targout_exists"] = rest_paths[f"{desc}_targout"].apply(
        lambda x: x.exists()
    )

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
        "suffix": "targetclustinfo",
        "space": space,
        "atlas": "Coords",
        "extension": ".tsv",
    }
    rest_paths[f"{desc}_targout_cluster_info"] = rest_paths.entities.apply(
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
    rest_paths[f"{desc}_targout_cluster_info_exists"] = rest_paths[
        f"{desc}_targout_cluster_info"
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

    clust_updates = {
        "desc": desc,
        "suffix": "net_reference_correlation",
        "space": space,
        "atlas": "Coords",
        "extension": ".nii.gz",
    }
    rest_paths[f"{desc}_net_reference_correlation"] = rest_paths.entities.apply(
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
    rest_paths[f"{desc}_net_reference_correlation_exists"] = rest_paths[
        f"{desc}_net_reference_correlation"
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

    rest_paths.loc[~rest_paths.ref_mask_exists].apply(
        lambda x: transform_mask_to_t1w(
            x, inmask=ref_roi_2mm_path.as_posix(), outmask_col="ref_mask"
        ),
        axis=1,
    )
    rest_paths["ref_mask_exists"] = rest_paths.ref_mask.apply(lambda x: x.exists())
    assert rest_paths.ref_mask_exists.all()

    # clean up transformed masks by dropping disconnected bits
    rest_paths.loc[~rest_paths.clnstim_mask_exists].apply(
        lambda row: clean_mask(
            nl.image.load_img(row.stim_mask),
            nl.image.load_img(row.brain_mask),
            max_drop_frac=0.1,
            clean_mask_path=row.clnstim_mask,
            error="skip",
        ),
        axis=1,
    )
    rest_paths["clnstim_mask_exists"] = rest_paths.clnstim_mask.apply(
        lambda x: x.exists()
    )
    if not rest_paths.clnstim_mask_exists.all():
        raise ValueError("Could not create a clean mask")

    rest_paths.loc[~rest_paths.clnref_mask_exists].apply(
        lambda row: clean_mask(
            nl.image.load_img(row.ref_mask),
            nl.image.load_img(row.brain_mask),
            max_drop_frac=0.1,
            clean_mask_path=row.clnref_mask,
            error="skip",
        ),
        axis=1,
    )
    rest_paths["clnref_mask_exists"] = rest_paths.clnref_mask.apply(
        lambda x: x.exists()
    )
    if not rest_paths.clnref_mask_exists.all():
        raise ValueError("Could not create a clean mask")

    if debug:
        # now that we've got all the paths set, lets start finding targets
        r = []
        for ix, row in rest_paths.iterrows():
            r.append(find_target(
                    row,
                    "bold_path",
                    f"{desc}_targout",
                    n_dummy=n_dummy,
                    t_r=t_r,
                    distance_threshold=distance_threshold,
                    adjacency=adjacency,
                    smoothing_fwhm=smoothing_fwhm,
                    confound_selectors=None,
                    metric=custom_metric,
                    linkage=linkage,
                    write_pkl=True,
                    return_df=True,
                )
            )
        targouts = pd.concat(r)
    else:
        # now that we've got all the paths set, lets start finding targets
        jobs = []
        for ix, row in rest_paths.iterrows():
            jobs.append(
                delayed(find_target)(
                    row,
                    "bold_path",
                    f"{desc}_targout",
                    n_dummy=n_dummy,
                    t_r=t_r,
                    distance_threshold=distance_threshold,
                    adjacency=adjacency,
                    smoothing_fwhm=smoothing_fwhm,
                    confound_selectors=None,
                    metric=custom_metric,
                    linkage=linkage,
                    write_pkl=True,
                    return_df=True,
                )
            )
        r = Parallel(n_jobs=njobs, verbose=10)(jobs)
        targouts = pd.concat(r)

    # merge some paths in to the targets dataframe
    rp_cols = [
        "subject",
        "session",
        "run",
        "clnstim_mask",
        "clnref_mask",
        "bold_path",
        "brain_mask",
        "confounds",
        f"{desc}_targout",
        f"{desc}_targout_exists",
        f"{desc}_targout_cluster",
        f"{desc}_targout_cluster_exists",
        f"{desc}_targout_cluster_info",
        f"{desc}_targout_cluster_info_exists",
        f"{desc}_targout_coordinates",
        f"{desc}_targout_coordinates_exists",
        f"{desc}_net_reference_correlation",
        f"{desc}_net_reference_correlation_exists",
    ]
    kept_cols = [rpc for rpc in rp_cols if rpc in rest_paths.columns]
    tmp_merge_on = ["subject", "session", "run"]
    tmp_merge_on = [tt for tt in tmp_merge_on if tt in targouts.columns]
    tmp = targouts.merge(
        rest_paths.loc[:, kept_cols],
        how="left",
        on=tmp_merge_on,
        indicator=True,
    )
    assert (tmp._merge != "both").sum() == 0
    targouts_mp = targouts.merge(
        rest_paths.loc[:, kept_cols], how="left", on=tmp_merge_on
    )
    if not run in tmp_merge_on:
        targouts_mp['run'] = 'merged'
    # rank clusters
    targouts_sel = rank_clusters(
        targouts_mp, nvox_weight, concentration_weight, net_reference_correlation_weight
    )

    # write out results
    for ixs, df in targouts_sel.groupby(tmp_merge_on[::-1], dropna=False):
        target_row = df.query("overall_rank == 1").iloc[0]
        target_cluster = get_clust_image(target_row)
        target_cluster.to_filename(target_row[f"{desc}_targout_cluster"])
        df.to_csv(target_row[f"{desc}_targout_cluster_info"], index=None, sep="\t")
        target_coords = get_com_in_mm(target_row)
        target_coords_df = pd.DataFrame(data=[target_coords], columns=["x", "y", "z"])
        target_coords_df.to_csv(
            target_row[f"{desc}_targout_cluster_info"], index=None, sep="\t"
        )

        vox_ref_corr_img = get_vox_ref_corr(
            target_row,
            distance_threshold,
            adjacency,
            smoothing_fwhm,
            ["-gs", "-motion", "-dummy"],
            custom_metric,
            linkage,
            t_r,
            n_dummy,
        )
        vox_ref_corr_img.to_filename(target_row[f"{desc}_net_reference_correlation"])


def test_block_bootstrap():
    # Test with random data.
    np.random.seed(0)
    ts = np.random.randn(500, 430)
    nsamples = 10
    block_length = 10
    samples = block_bootstrap(ts, nsamples, block_length)

    # Check that the output has the correct shape.
    assert samples.shape == (nsamples, ts.shape[0], ts.shape[1])

    # check that all the values in the output came from the correct row in the input
    assert np.all([np.isin(ss, tt) for sample in samples for tt, ss in zip(ts, sample)])

    block_length = 55
    samples = block_bootstrap(ts, nsamples, block_length)

    # Check that the output has the correct shape.
    assert samples.shape == (nsamples, ts.shape[0], ts.shape[1])

    # check that all the values in the output came from the correct row in the input
    assert np.all([np.isin(ss, tt) for sample in samples for tt, ss in zip(ts, sample)])

    # Test with a different seed.
    seed = 1
    samples1 = block_bootstrap(ts, nsamples, block_length, seed)
    samples2 = block_bootstrap(ts, nsamples, block_length, seed)

    # Check that the two sets of samples are the same.
    assert np.allclose(samples1, samples2)