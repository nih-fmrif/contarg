import subprocess
from pkg_resources import resource_filename
from pathlib import Path
from bids import BIDSLayout
import pandas as pd
import nilearn as nl
from nilearn import image
import nibabel as nb
from nibabel import cifti2 as ci
from simnibs import sim_struct, opt_struct
from simnibs.utils.transformations import interpolate_to_volume
from simnibs.simulation.petsc_solver import SolverError
from joblib import Parallel, delayed
import numpy as np
import click
from contarg.normgrid import load_liston_surfs, load_surfaces
from contarg.stimgrid import run_opt_and_save_outputs, angle_between
from contarg.utils import surf_data_from_cifti
import templateflow
from scipy.spatial.distance import cdist


@click.group()
def contarg():
    pass


@contarg.group()
def normgrid():
    pass

@normgrid.command()
@click.option("--headmodel-dir", type=click.Path(exists=True), help="Path to HeadModel directory.", required=True)
@click.option("--searchgrid-dir", type=click.Path(exists=True), help="Path to SearchGrid directory.", required=True)
@click.option("--out-dir", type=click.Path(), help="Path to write to.", required=True)
@click.option("--src-surf-dir", type=click.Path(exists=True), help="Path in which to find subject surfaces.", required=True)
@click.option("--coil",  default='MagVenture_MCF-B65.ccd', show_default=True,
              help="Name of the coil if it's one of the ones that comes with simnibs, othewise the path to it.")
@click.option(
    "--distancetoscalp",
    type=float,
    default=2,
    show_default=True,
    help="Distance in mm from the coil to the scalp. Default=2.",
)
@click.option(
    "--njobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of jobs to run in parallel to find targets",
)
@click.option(
    "--surf_src",
    type=str,
    default='liston',
    show_default=True,
    help='flag to indicate where data is coming from, options are liston or fmriprep')
@click.option("--bids-dir", type=click.Path(), help="Bids directory", required=False)
@click.option("--fmriprep-dir", type=click.Path(), help="FMRIPREP directory", required=False)
@click.option("--anat-dir", type=click.Path(), help="contarg anat outputs directory", required=False)
def sim_gyral_lip(headmodel_dir, searchgrid_dir, out_dir, src_surf_dir,
                  coil='MagVenture_MCF-B65.ccd', distancetoscalp=2,
                  surf_src='liston', bids_dir=None, fmriprep_dir=None, anat_dir=None, njobs=1):
    HeadModel_dir = Path(headmodel_dir)
    SearchGrid_dir = Path(searchgrid_dir)
    src_surf_dir = Path(src_surf_dir)
    out_dir = Path(out_dir)

    try:
        m2m_dir = sorted(HeadModel_dir.glob('m2m*'))[0]

    except IndexError:
        raise FileNotFoundError(f"No m2m directory found in {HeadModel_dir}")
    skinsurf_path = m2m_dir / 'Skin.surf.gii'
    subject = '_'.join(m2m_dir.parts[-1].split('_')[1:])
    headmesh_path = m2m_dir / f'{subject}.msh'

    # load surfaces
    if surf_src == 'liston':
        surfaces = load_liston_surfs(subject, src_surf_dir)
    elif surf_src =='fmriprep':
        if bids_dir is None or fmriprep_dir is None or anat_dir is None:
            raise ValueError("Must specify bids_dir, fmriprep_dir, and anat_dir if surf_src is fmriprep")
        layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir)
        surfaces = load_surfaces(subject, layout, anat_dir, overwrite=False)
    else:
        raise NotImplementedError

    cortical_points, normals = np.load((SearchGrid_dir / 'SearchGrid.npy').as_posix())

    if not np.isclose(np.linalg.norm(normals, axis=1), 1).all():
        raise ValueError(
            'The second element of the search grid should be normal vectors, but some vectors had norm != 1')

    # get the skin surface vertices
    skinsurf = nb.load(skinsurf_path)
    skinsurf_verts = skinsurf.get_arrays_from_intent('pointset')[0].data

    if not Path(coil).exists():
        simnibs_coil_dir = Path(resource_filename('simnibs', 'resources/coil_models'))
        coil_path = simnibs_coil_dir / f'Drakaki_BrainStim_2022/{coil}'
        if not coil_path.exists():
            coil_path = simnibs_coil_dir / f'legacy_and_other/{coil}'
            if not coil_path.exists():
                raise FileNotFoundError(f"Could not find coil ({coil}) in {simnibs_coil_dir}.")
    else:
        coil_path = Path(coil)

    jobs = []
    sim_run_n = 0
    sim_dir = out_dir / f'simulation-{sim_run_n:02d}'
    while sim_dir.exists():
        sim_run_n += 1
        sim_dir = out_dir / f'simulation-{sim_run_n:02d}'

    sim_dir.mkdir(parents=True)
    for ii, (cp, nn) in enumerate(zip(cortical_points, normals)):
        poso_outdir = sim_dir / f'opt-{ii:04d}'
        poss_outdir = poso_outdir / 'sim'

        assert poso_outdir.parent.exists()
        if poso_outdir.exists():
            raise ValueError(f"Output directory {poso_outdir} exists, Simnibs won't run.")
        o = opt_struct.TMSoptimize()
        o.fnamehead = headmesh_path.as_posix()
        o.pathfem = poso_outdir.as_posix()
        o.fnamecoil = coil_path.as_posix()
        o.target = cp
        o.target_direction = nn
        o.target_size = 2
        o.distance = distancetoscalp
        o.method = 'ADM'
        # o.map_to_vol = True
        # o.map_to_surf = True
        # o.fields = 'eE'
        o.angle_resolution = 5
        o.spatial_resolution = 2
        o.solver_options = 'paradiso'
        o.open_in_gmsh = False
        jobs.append(delayed(run_opt_and_save_outputs)(o))
        # run_opt_and_save_outputs(o)

    print(f"Running {len(jobs)} optimizations", flush=True)
    Parallel(n_jobs=njobs, verbose=10)(jobs)
    print('###################')
    print('Finished optimizations, consolidating outputs')

    # concat magnE
    catted = sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-magnE_stat.nii.gz"
    if not catted.exists():
        cmd = [
            "3dTcat",
            "-overwrite",
            "-prefix",
            catted,
            f"{sim_dir.as_posix()}/opt-*/subject_volumes/*_magnE.nii.gz"
        ]
        subprocess.run(cmd, check=True)

    # concat target
    catted = sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-target_stat.nii.gz"
    if not catted.exists():
        cmd = [
            "3dTcat",
            "-DAFNI_GLOB_SELECTORS=YES",
            "-overwrite",
            "-prefix",
            catted,
            f"{sim_dir.as_posix()}/opt-*/subject_volumes/*_Target.nii.gz[0]"
        ]
        subprocess.run(cmd, check=True)

    # concat evec
    catted = sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-Ei_stat.nii.gz"
    if not catted.exists():
        cmd = [
            "3dTcat",
            "-DAFNI_GLOB_SELECTORS=YES",
            "-overwrite",
            "-prefix",
            catted,
            f"{sim_dir.as_posix()}/opt-*/subject_volumes/*_E.nii.gz[0]"
        ]
        subprocess.run(cmd, check=True)

    catted = sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-Ej_stat.nii.gz"
    if not catted.exists():
        cmd = [
            "3dTcat",
            "-DAFNI_GLOB_SELECTORS=YES",
            "-overwrite",
            "-prefix",
            catted,
            f"{sim_dir.as_posix()}/opt-*/subject_volumes/*_E.nii.gz[1]"
        ]
        subprocess.run(cmd, check=True)

    catted = sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-Ek_stat.nii.gz"
    if not catted.exists():
        cmd = [
            "3dTcat",
            "-DAFNI_GLOB_SELECTORS=YES",
            "-overwrite",
            "-prefix",
            catted,
            f"{sim_dir.as_posix()}/opt-*/subject_volumes/*_E.nii.gz[2]"
        ]
        subprocess.run(cmd, check=True)

    consolidated_paths = [sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-{metric}_stat.nii.gz" for metric in
                          ['magnE', 'Ei', 'Ej', 'Ek', 'target']]

    medial_wall = {}
    medial_wall['l'] = templateflow.api.get(template='fsLR', density='32k', desc='nomedialwall', hemi='L')
    medial_wall['r'] = templateflow.api.get(template='fsLR', density='32k', desc='nomedialwall', hemi='R')

    consolidated_paths = [sim_dir / f"sub-{subject}_simulation-{sim_run_n:02d}_desc-{metric}_stat.nii.gz" for metric in
                          ['magnE', 'Ei', 'Ej', 'Ek', 'target']]
    cifti_outs = []
    for metric_path in consolidated_paths:
        cifti_out = metric_path.as_posix().replace(".nii.gz", ".dtseries.nii").replace("_desc-",
                                                                                       "_hemi-L_space-fsLR_den-32k_desc-")
        if not Path(cifti_out).exists():
            l_out_gifti = metric_path.as_posix().replace(".nii.gz", ".shape.gii").replace("_desc-",
                                                                                          "_hemi-L_space-fsLR_den-32k_desc-")
            l_v2s_cmd = [
                'wb_command',
                '-volume-to-surface-mapping',
                metric_path.as_posix(),
                surfaces.l.midthickness.path,
                l_out_gifti,
                '-ribbon-constrained',
                surfaces.l.white.path,
                surfaces.l.pial.path,
            ]
            subprocess.run(l_v2s_cmd, check=True)

            l_metricmask_cmd = [
                'wb_command',
                '-metric-mask',
                l_out_gifti,
                medial_wall['l'],
                l_out_gifti
            ]
            subprocess.run(l_metricmask_cmd, check=True)

            r_out_gifti = metric_path.as_posix().replace(".nii.gz", ".shape.gii").replace("_desc-",
                                                                                          "_hemi-R_space-fsLR_den-32k_desc-")
            r_v2s_cmd = [
                'wb_command',
                '-volume-to-surface-mapping',
                metric_path.as_posix(),
                surfaces.r.midthickness.path,
                r_out_gifti,
                '-ribbon-constrained',
                surfaces.r.white.path,
                surfaces.r.pial.path,
            ]
            subprocess.run(r_v2s_cmd, check=True)

            r_metricmask_cmd = [
                'wb_command',
                '-metric-mask',
                r_out_gifti,
                medial_wall['r'],
                r_out_gifti
            ]
            subprocess.run(r_metricmask_cmd, check=True)

            create_cifti_cmd = [
                'wb_command',
                '-cifti-create-dense-timeseries',
                cifti_out,
                '-left-metric',
                l_out_gifti,
                '-roi-left',
                medial_wall['l'],
                '-right-metric',
                r_out_gifti,
                '-roi-right',
                medial_wall['r'],
            ]
            subprocess.run(create_cifti_cmd, check=True)
        cifti_outs.append(cifti_out)

    # gather optimal point info
    coil_name = coil.split('.')[0]
    outputs = []
    for ii, (cp, nn) in enumerate(zip(cortical_points, normals)):
        poso_outdir = sim_dir / f'opt-{ii:04d}'
        vol_outdir = poso_outdir / 'subject_volumes'
        target_path = vol_outdir / f'{subject}_TMS_1-0001_{coil_name}_Target.nii.gz'
        magn_path = vol_outdir / f'{subject}_TMS_1-0001_{coil_name}_magnE.nii.gz'
        norm_path = vol_outdir / f'{subject}_TMS_1-0001_{coil_name}_E.nii.gz'
        optmat_path = poso_outdir / f'{subject}_TMS_optimize_{coil_name}_optmat.npy'
        row = dict(
            cx=cp[0],
            cy=cp[1],
            cz=cp[2],
            nx=nn[0],
            ny=nn[1],
            nz=nn[2],
            target_path=target_path,
            target_path_exist=target_path.exists(),
            magn_path=magn_path,
            magn_path_exist=magn_path.exists(),
            norm_path=norm_path,
            norm_path_exist=norm_path.exists(),
            optmat_path=optmat_path,
            optmat_path_exist=optmat_path.exists()
        )
        if optmat_path.exists():
            optmat = np.load(optmat_path)
            row['bx'] = optmat[0, 3]
            row['by'] = optmat[1, 3]
            row['bz'] = optmat[2, 3]
            row['bxv'] = optmat[:-1, 0]
            row['byv'] = optmat[:-1, 1]
            row['bzv'] = optmat[:-1, 2]
        outputs.append(row)
    outputs = pd.DataFrame(outputs)

    # check that targets are where they should be
    targets_path = cifti_outs[-1]
    targets_img = ci.load(targets_path)
    targets = targets_img.get_fdata()
    l_targets = surf_data_from_cifti(targets, targets_img.header.get_axis(1), 'CIFTI_STRUCTURE_CORTEX_LEFT').T
    sn_idxs = np.array([np.nonzero([lt == lt.max()])[1] for lt in l_targets]).squeeze()
    sn_targets = np.array([surfaces.l.midthickness.points[lt == lt.max()] for lt in l_targets]).squeeze()
    misses = ~np.isclose(sn_targets, cortical_points)
    if misses.sum() != 0:
        print(f"Subject {subject} has {misses.sum()} instances where the simnibs target"
              f" doesn't match the intended target. Inspect their results carefully.")
        outputs['simnibs_to_intended_target'] = [cdist([snt], [cp])[0][0] for snt, cp in zip(sn_targets, cortical_points)]
    else:
        outputs['simnibs_to_intended_target'] = 0

    # update poslist info based on outputs
    outputs['z_angle'] = np.array([angle_between(sy, [0, 1]) for sy in outputs.byv.str[1:]])
    outputs.loc[outputs.byv.str[1] < 0, 'z_angle'] = 360 - outputs.loc[outputs.byv.str[1] < 0, 'z_angle']

    pos_list = outputs.copy()
    pos_list['pos_ix'] = pos_list.index.values
    # add target coordinates
    pos_list['s_idx'] = sn_idxs
    pos_list['s_x'] = sn_targets[:, 0]
    pos_list['s_y'] = sn_targets[:, 1]
    pos_list['s_z'] = sn_targets[:, 2]
    pos_list.to_pickle(sim_dir / f'sub-{subject}_simulations.pkl.gz')


def run_sim_and_clean(fnamehead, pathfem, coil_path, matsimnibs):
    s = sim_struct.SESSION()
    s.fnamehead = fnamehead
    s.pathfem = pathfem
    s.open_in_gmsh = False
    s.map_to_vol = False
    s.fields = 'e'
    tms_list = s.add_tmslist()
    tms_list.fnamecoil = coil_path

    # 1 indexing to match what simnibs does
    pos = tms_list.add_position()
    pos.matsimnibs = matsimnibs
    try:
        s.run()
    except Exception as e:
        print(e, flush=True)
    del(s)

@normgrid.command()
@click.option("--headmesh-path", type=click.Path(exists=True), help="Path to mesh file.")
@click.option("--settings-path", type=click.Path(exists=True), help="Path to simulation settings file.")
@click.option(
    "--ix",
    type=int,
    help="Index for which set of simulations from the simulation settings file to run.",
)
@click.option("--coil-path", type=click.Path(exists=True), help="Path to coil file.")

@click.option("--tmp-dir", type=click.Path(), help="Path to write temporary sims to.")
@click.option("--out-dir", type=click.Path(), help="Path to write summary stats to.")
@click.option(
    "--njobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of jobs to run in parallel to find targets",
)

@click.option(
    "--max-mt",
    type=int,
    default=80,
    show_default=True,
    help="Maximum motor threshold. Stimulation values will be scaled based on this.",
)
@click.option(
    "--thresh-type",
    type=click.Choice(['mt', 'motor-threshold', 'fi', 'field-intensity'], case_sensitive=False),
    default='mt',
    show_default=True,
    help="Should the threhold be interpretted as a motor threshold or a E-field threshold in V/m."
)
@click.option(
    "--min-thresh",
    type=int,
    default=60,
    show_default=True,
    help="Minimum threshold value to call something an activation. Must be less than max-thresh",
)
@click.option(
    "--max-thresh",
    type=int,
    default=120,
    show_default=True,
    help="Maximum threshold. Vertices receiving this amount or greater"
         " will be given an activation probability of 100%.",
)
def sim_uncert(headmesh_path, settings_path, ix, coil_path, tmp_dir, out_dir, njobs=1, max_mt=80,
               thresh_type='mt', min_thresh=60, max_thresh=120
               ):
    # deal with thesh_type
    if thresh_type.lower() in ['mt', 'motor-threshold']:
        thresh_type = 'mt'
    elif thresh_type.lower() in ['fi', 'field-intensity']:
        thresh_type = 'fi'
    else:
        raise NotImplementedError
    Tmp_dir = Path(tmp_dir)
    Tmp_dir.mkdir(exist_ok=True, parents=True)
    out_dir = Path(out_dir)
    settings = pd.read_pickle(settings_path)
    settings = settings.loc[settings.oix == ix].copy()

    # check that min_mt_thresh is below max_mt
    if min_thresh >= max_thresh:
        raise ValueError(f"min_mt_thresh must be below max_mt, you passed min_thresh={min_thresh} "
                         f"and max_thresh={max_thresh}.")

    # get max magne
    magne_path = list(Path(settings_path).parent.parent.glob('*magnE_stat.dtseries.nii'))[0]
    magne_img = ci.load(magne_path)
    all_magne = magne_img.get_fdata()
    all_magne = all_magne.T
    max_magnE = all_magne.max(0)[ix]

    # Get threshold in terms of MT, even if it was given in terms of field
    # this way we can do targeting before we know the MT
    if thresh_type == 'fi':
        min_mt_thresh = (min_thresh / max_magnE) * max_mt
        max_mt_thresh = (max_thresh / max_magnE) * max_mt
    else:
        min_mt_thresh = min_thresh
        max_mt_thresh = max_thresh

    # variables for building outmesh names
    subject = Path(headmesh_path).parts[-1].split(".")[0]
    coil_name = Path(coil_path).parts[-1].split(".")[0]

    out_meshes = []
    out_vols = []
    jobs = []
    omix = 1
    for ssix, srow in settings.iterrows():
        out_meshes.append(Tmp_dir / f'sim_{omix:04d}/{subject}_TMS_1-0001_{coil_name}_scalar.msh')
        out_vols.append(Tmp_dir / f'subject_volumes/{subject}_TMS_1-{omix:04d}_{coil_name}_scalar_magnE.nii.gz')
        jobs.append(delayed(run_sim_and_clean)(headmesh_path,
                                       (Tmp_dir / f'sim_{omix:04d}').as_posix(),
                                       coil_path,
                                       srow.matsimnibs))
        omix += 1
    _ = Parallel(n_jobs=njobs, verbose=10)(jobs)
    settings['out_mesh'] = out_meshes
    settings['out_vol'] = out_vols

    settings['out_mesh_exist'] = settings.out_mesh.apply(lambda x: x.exists())
    settings['out_vol_exist'] = settings.out_vol.apply(lambda x: x.exists())
    # see if any out_vols corresponding to existing masks are missing and attempt to make them if so
    (Path(tmp_dir) / f'subject_volumes').mkdir(exist_ok=True, parents=True)
    jobs = []
    for _, row in settings.loc[settings.out_mesh_exist & ~settings.out_vol_exist].iterrows():
        jobs.append(delayed(interpolate_to_volume)(row.out_mesh.as_posix(),
                              Path(headmesh_path).parent.as_posix(),
                              row.out_vol.as_posix().replace('_magnE.nii.gz', ''),
                              keep_tissues=[2]))
    print(f"Converting {len(jobs)} meshes to volumes with {njobs} jobs.")
    _ = Parallel(n_jobs=njobs, verbose=10)(jobs)
    settings['out_vol_exist'] = settings.out_vol.apply(lambda x: x.exists())

    unc_maps = settings.loc[settings.out_vol_exist, 'out_vol'].values
    weights = settings.loc[settings.out_vol_exist, 'prob'].values
    weights = weights / weights.sum()
    assert len(unc_maps) == len(weights)
    print(f'Accumulating values from {len(weights)} maps that ran successfully.')
    meanfile = out_dir / f'oix-{ix:04d}_stat-mean_magnE.nii.gz'
    stdfile = out_dir / f'oix-{ix:04d}_stat-std_magnE.nii.gz'
    atpfile = out_dir / f'oix-{ix:04d}_stat-abovethreshactprobs_magnE.nii.gz'

    def get_maps_stats(files, weights, max_magnE, min_mt_thresh, max_mt_thresh, max_mt=80):
        shape = nb.load(files[0]).shape
        vals = np.zeros((shape[0], shape[1], shape[2], len(files)))
        for fix, (unc_map, w) in enumerate(zip(files, weights)):
            vals[:, :, :, fix] = nb.load(unc_map).get_fdata()
        mean = (vals * weights).sum(-1) / weights.sum()
        S = ((vals - mean.reshape(shape[0], shape[1], shape[2], 1)) ** 2 * weights).sum(-1)
        mts = (vals / max_magnE) * max_mt
        t = min_mt_thresh
        m = -1 / (t - max_mt_thresh)
        b = 1 - (m * max_mt_thresh)
        act_weighted_mts = mts * m + b
        act_weighted_mts[act_weighted_mts < 0] = 0
        act_weighted_mts[act_weighted_mts > 1] = 1
        above_thresh_probs = ((mts >= min_mt_thresh) * weights * act_weighted_mts).sum(-1)
        del vals
        return mean, S, weights.sum(), above_thresh_probs

    n_per_job = 10
    i = 0
    files = []
    run_weights = []
    jobs = []
    tmp2 = []
    for file, weight in zip(unc_maps, weights):
        files.append(file)
        run_weights.append(weight)
        i += 1
        if i == n_per_job:
            jobs.append(delayed(get_maps_stats)(files, np.array(run_weights), max_magnE,
                                                min_mt_thresh, max_mt_thresh, max_mt))
            files = []
            run_weights = []
            i = 0
    # deal with leftovers
    if len(files) > 0:
        jobs.append(delayed(get_maps_stats)(files, np.array(run_weights), max_magnE,
                                            min_mt_thresh, max_mt_thresh, max_mt))
    maps = Parallel(n_jobs=njobs, verbose=10)(jobs)

    # aggregate mean images
    meana = None
    for meanb, Sb, w_sumb, atpb in maps:
        if meana is None:
            meana = meanb
            Sa = Sb
            w_suma = w_sumb
            atpa = atpb

        else:
            w_sumab = w_suma + w_sumb

            meanab = ((w_suma * meana) + (w_sumb * meanb)) / w_sumab
            sab_rat = (w_suma * w_sumb) / w_sumab
            sab_term1 = (meanb ** 2) * sab_rat
            sab_term2 = -2 * (meana * meanb) * sab_rat
            sab_term3 = (meana ** 2) * sab_rat
            Sab = Sa + Sb + sab_term1 + sab_term2 + sab_term3
            atpab = atpa + atpb

            w_suma = w_sumab
            meana = meanab
            Sa = Sab
            atpa = atpab

    tmpimg = nl.image.load_img(unc_maps[0])
    meanimg = nl.image.new_img_like(tmpimg, meana, affine=tmpimg.affine, copy_header=True)
    meanimg.to_filename(meanfile)

    stdimg = nl.image.new_img_like(tmpimg, np.sqrt(Sa / w_suma), affine=tmpimg.affine, copy_header=True)
    stdimg.to_filename(stdfile)

    atpimg = nl.image.new_img_like(tmpimg, atpa, affine=tmpimg.affine, copy_header=True)
    atpimg.to_filename(atpfile)


@normgrid.command()
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
    help="Name of functional preprocessing run. If provided,"
    "output will be placed in derivatives_dir/contarg/func_preproc. Otherwise,"
    "output will be in derivatives_dir/{run-name}/func_preproc",
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
    "--ndummy",
    "n_dummy",
    type=int,
    help="Number of dummy scans at the beginning of the functional time series",
)
@click.option(
    "--tr", "t_r", type=float, help="Repetition time of the functional time series"
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
def preproc_func(
    bids_dir,
    derivatives_dir,
    database_file,
    fmriprep_dir,
    tedana_dir,
    run_name,
    cortical_smoothiing,
    subcortical_smoothing,
    n_dummy,
    t_r,
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
        func_dir = derivatives_dir / run_name / "func_preproc"
    else:
        func_dir = derivatives_dir / "contarg" / "func_preproc"
    func_dir.mkdir(parents=True, exist_ok=True)

    # Getting all the needed input paths
    # build paths df off of bolds info
    get_kwargs = {
        "return_type": "object",
        "task": "rest",
        "desc": 'optcomDenoised',
        "suffix": "bold",
        "extension": ".nii.gz",
        "datatype": 'func'
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