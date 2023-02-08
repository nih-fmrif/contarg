from pathlib import Path
from bids import BIDSLayout  # pip version = 0.15.2
import pandas as pd
import nilearn as nl
from nilearn import image, masking
from joblib import Parallel, delayed
import click
from pkg_resources import resource_filename
from contarg.utils import make_path, transform_mask_to_t1w, clean_mask
from contarg.hierarchical import (
    find_target,
    custom_metric,
    rank_clusters,
    get_com_in_mm,
    get_clust_image,
    get_vox_ref_corr,
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
    refroi_name,
    refroi_path,
    space,
    smoothing_fwhm,
    n_dummy,
    t_r,
    linkage,
    adjacency,
    distance_threshold,
    nvox_weight,
    concentration_weight,
    net_reference_correlation_weight,
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
        targeting_dir = derivatives_dir / "contarg" / "hierarchical" / run_name
    else:
        targeting_dir = derivatives_dir / "contarg" / "hierarchical"
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

    assert stim_roi_2mm_path.exists()
    assert ref_roi_2mm_path.exists()

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
            **{"from": "MNI152NLin2009cAsym"},
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
        boldmask_pattern = "sub-{subject}/[ses-{session}/]func/sub-{subject}_[ses-{session}_]task-{task}_run-{run}_[echo-{echo}_]atlas-{atlas}_space-{space}_desc-{desc}_{suffix}.{extension}"
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
        rest_paths["clnref_mask_exists"] = False

    else:
        # TODO: clean this up for custom ROIs
        clnref_roi_2mm_path = (
            roi_dir / f"{ref_roi_name + 'masked'}_space-MNI152NLin6Asym_res-02.nii.gz"
        )
        rest_paths[f"stim_mask"] = stim_roi_2mm_path
        rest_paths["stim_mask_exists"] = True
        rest_paths["ref_mask"] = ref_roi_2mm_path
        rest_paths["ref_mask_exists"] = True
        rest_paths[f"clnstim_mask"] = stim_roi_2mm_path
        rest_paths["clnstim_mask_exists"] = True
        rest_paths["clnref_mask"] = clnref_roi_2mm_path
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
                confound_selectors=["-gs", "-motion", "-dummy"],
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
    tmp = targouts.merge(
        rest_paths.loc[:, rp_cols],
        how="left",
        on=["subject", "session", "run"],
        indicator=True,
    )
    assert (tmp._merge != "both").sum() == 0
    targouts_mp = targouts.merge(
        rest_paths.loc[:, rp_cols], how="left", on=["subject", "session", "run"]
    )

    # rank clusters
    targouts_sel = rank_clusters(
        targouts_mp, nvox_weight, concentration_weight, net_reference_correlation_weight
    )

    # write out results
    for ixs, df in targouts_sel.groupby(["run", "session", "subject"], dropna=False):
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
