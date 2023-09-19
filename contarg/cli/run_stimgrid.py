from pathlib import Path
import click
import pandas as pd
import nibabel as nb
from nibabel import cifti2 as ci


from contarg.utils import get_refroi_path, get_stimroi_path, SurfROI, load_timeseries, parse_bidsname, build_bidsname
from contarg.stimgrid import load_stimgrid, find_timeseries, run_stimgrid_eval


@click.group()
def contarg():
    pass


@contarg.group()
def stimgrid():
    pass

@stimgrid.command()
@click.option("--anat-input-dir", type=click.Path(exists=True), help="Path to directory with anatomical preprocessed "
                                                                     "things (HeadModel, SearchGrid Simulation).")
@click.option("--func-input-dir", type=click.Path(exists=True), help="Path to directory with functional preprocessed"
                                                                     " things (timeseries).")
@click.option("--out-dir", type=click.Path(), help="Directory where outputs will be written.")
@click.option(
    "--subject",
    type=str,
    required=True,
    help="Subject number",
)
@click.option(
    "--run-ind", type=int, default=None, help="Which run to use, not based on run number, but order of runs."
                                              "0 will get the first run regardless of run number,"
                                              "1 the second run, and so on."
)
@click.option(
    "--nbootstraps",
    type=int,
    default=100,
    show_default=True,
    help="Number of bootstraps to run.",
)
@click.option(
    "--block-length",
    type=int,
    default=45,
    show_default=True,
    help="Blocklength to use in bloock bootstrap",
)
@click.option(
    "--stimroi-name",
    type=str,
    default="expandedcoleBA46",
    help="Name of roi to which stimulation will be delivered. "
    "Should be one of ['DLPFCspheres', 'BA46sphere'], "
    "or provide the path to the roi in MNI152NLin6Asym space with the stimroi-path option.",
)
@click.option(
    "--stimroi-path",
    type=str,
    default=None,
    help="If providing a custom stim roi, give the path to the fslr 32k ROI here.",
)
@click.option(
    "--refroi-name",
    type=str,
    default="bilateralfullSGCsphere",
    help="Name of roi to whose connectivity is being used to pick a stimulation site. "
    "Should be one of ['SGCsphere', 'bilateralSGCSpheres'], "
    "or provide the path to the roi in MNI152NLin6Asym space with the refroi-path option.",
)
@click.option(
    "--refroi-path",
    type=str,
    default=None,
    help="If providing a custom ref roi, give the path to the fslr 32k ROI here.",
)
@click.option(
    "--njobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of jobs to run in parallel to find targets",
)
def eval(anat_input_dir, func_input_dir, out_dir, subject, run_ind,
                  nbootstraps, block_length,
                  stimroi_name, stimroi_path, refroi_name, refroi_path, njobs):
    anat_input_dir = Path(anat_input_dir)
    anat_dir = anat_input_dir / 'anat'
    simulation_dir = anat_input_dir / 'Simulation/'
    func_dir = Path(func_input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    stimroi_path = get_stimroi_path(stimroi_name, stimroi_path, cifti=True)
    stimroi = ci.load(stimroi_path)

    refroi_path = get_refroi_path(refroi_name, refroi_path, cifti=True)
    refroi = ci.load(refroi_path)

    r_surf_path = anat_dir / f'sub-{subject}.R.midthickness.32k_fs_LR.surf.gii'
    r_surf = nb.load(r_surf_path)

    l_surf_path = anat_dir / f'sub-{subject}.L.midthickness.32k_fs_LR.surf.gii'
    l_surf = nb.load(l_surf_path)

    magne_paths = sorted(simulation_dir.glob('*_angle-*.nii'))
    poslist = pd.read_csv(simulation_dir / f'sub-{subject}_simulations.tsv', sep='\t')
    poslist['pos_ix'] = poslist.index.values
    pos_idxs = load_stimgrid(magne_paths, poslist)

    l_mtva_path = anat_dir / f'sub-{subject}.L.midthickness_va.32k_fs_LR.shape.gii'
    l_mtva = nb.load(l_mtva_path).agg_data()
    pos_idxs['surface_area'] = l_mtva[pos_idxs.idx.values]

    timeseries = find_timeseries(func_dir, run_ind=run_ind)

    lts_data, rts_data, ts_data = load_timeseries(timeseries)
    ts_ents = parse_bidsname(timeseries[0])
    if run_ind is not None:
        prefix = build_bidsname(ts_ents, exclude=['type', 'desc', 'suffix', 'extension'])
    else:
        prefix = build_bidsname(ts_ents, exclude=['type', 'desc', 'suffix', 'extension', 'run'])

    dstim_roi = SurfROI(l_surf, 'left', lts_data, dilate=10, roi=stimroi)
    rref_roi = SurfROI(r_surf, 'right', rts_data, take_largest_cc=True, roi=refroi)
    lref_roi = SurfROI(l_surf, 'left', lts_data, take_largest_cc=True, roi=refroi)

    run_stimgrid_eval(out_dir, lref_roi, rref_roi, dstim_roi, poslist, pos_idxs, nbootstraps=nbootstraps,
                      block_length=block_length, nthreads=njobs, prefix=prefix)