import os
import shutil
import logging
import time
from pathlib import Path
from contarg.tans import get_tans_inputs
from contarg.utils import surf_data_from_cifti, SurfROI
from contarg.hierarchical import block_bootstrap, find_cluster_threshold, bootstrap_clustering, get_stim_stats_with_uncertainty
from pkg_resources import resource_filename
import scipy.io as sio
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from nibabel import cifti2 as ci
import nilearn as nl
from nilearn import image
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import stats
import networkx as nx

from simnibs import sim_struct, opt_struct, run_simnibs
from simnibs.utils.transformations import interpolate_to_volume, middle_gm_interpolation


def write_searchgrid_script(
        tans_input_dir,
        target_mask_path,
        out_dir,
        subject,
        msc_path=None,
        simnibs_path=None,
        tans_path=None,
):
    """This writes a script that prepares the searchgrid based on an roi.

    Parameters
    ----------
    tans_input_dir : str or Path
        Path to the folder containing TANS inputs.
    target_mask_path : str or Path
        Path to the CIFTI file containing the target network patch (structure array). Non-zero values in TargetPatch.data are considered target network patch vertices.
    out_dir : str or Path
        Path to the output folder.
    subject : str
        Subject identifier.
    msc_path : str, optional
        Path to the folder containing ft_read / gifti functions for reading and writing cifti files (e.g., https://github.com/MidnightScanClub/MSCcodebase).
    simnibs_path : str, optional
        Path to the SimNIBS installation folder (e.g., /path/to/SimNIBS). If None, the CONTARG_SIMNIBS_PATH environment variable is used.
    tans_path : str, optional
        Path to the TANS installation folder. If None, the TANS installation folder is set to 'contarg/resources/Targeted-Functional-Network-Stimulation'.

    Returns
    -------
    sim_script_path : Path
        Path to the
    """
    if subject[:4] == "sub-":
        subject = subject[4:]

    out_dir = Path(out_dir)
    m_dir = Path(out_dir / "matlab_scripts")
    m_dir.mkdir(exist_ok=True)

    tans_inputs = get_tans_inputs(tans_input_dir, subject)

    script_path = m_dir / "run_simprep.m"
    if msc_path is None:
        msc_path = os.getenv("CONTARG_MSC_PATH")
    if simnibs_path is None:
        simnibs_path = os.getenv("CONTARG_SIMNIBS_PATH")
    if tans_path is None:
        tans_path = resource_filename(
            "contarg", "resources/Targeted-Functional-Network-Stimulation"
        )

    simnibs_path = Path(simnibs_path)
    # some paths that simnibs will need
    simnibs_python = simnibs_path / "../conda/envs/4.0/bin/python"
    simnibs_dir = simnibs_path / "simnibs/simnibs"

    script = f"""
    %% Example use of Targeted Functional Network Stimulation ("TANS")
    % "Automated optimization of TMS coil placement for personalized functional
    % network engagement" - Lynch et al., 2022 (Neuron)

    % define some paths
    Paths{{1}} = '{Path(simnibs_path) / 'simnibs/simnibs/matlab_tools'}'; % download from https://simnibs.github.io/simnibs/build/html/index.html
    Paths{{2}} = '{msc_path}'; % this folder contains ft_read / gifti functions for reading and writing cifti files (e.g., https://github.com/MidnightScanClub/MSCcodebase).
    Paths{{3}} = '{tans_path}'; %

    % add folders
    % to search path;
    for i = 1:length(Paths)
        addpath(genpath(Paths{{i}}));
    end

    % define some global variables for simnibs
    SIMNIBSPYTHON = '{simnibs_python}';
    SIMNIBSDIR = '{simnibs_dir}';

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

    % If successful, each of the commands below should return status as 2.

    % Confirm that functions for reading and writing
    % CIFTI and GIFTI files are also available
    status = exist('ft_read_cifti_mod','file');
    assert(status == 2, 'ft_read_cifti_mod not available')
    status = exist('gifti','file');
    assert(status == 2, 'gifti not available')

    % Note: The TANS approach is intended to target surface-registered functional brain networks mapped in individual subjects.
    % In principle, however, any spatial map, such as an ICA component, functional connectivity or task activation map can be used instead.
    % Weighted maps must be first thresholded and binarized before they can be used as inputs to the tans_roi function.

    % Make a search grid on the scalp above the target network patch centroid.
    % Timing: Typically 5-10 minutes.

    % Use the tans_searchgrid function to generate an array of three-dimensional coordinates
    % representing a search grid on the scalp directly above the centroid of the target network patch.

    % The inputs to the function are:
    % TargetPatch: A CIFTI file containing the target network patch (structure array). Non-zero values in TargetPatch.data are considered target network patch vertices.
    % PialSurfs: Paths to low (32k) dimensional FS_LR pial surfaces (Cell array of strings, PialSurfs{1} = path to LH, PialSurfs{2} = path to RH).
    % SkinSurf: Path to the skin surface geometry file (string).
    % SearchGridRadius: Radius (in mm) of the search grid on the scalp directly above the centroid of the target network patch.
    % GridSpacing: The (approximate) distance between pairs of vertices in the search grid, in mm.
    % OutDir: Path to the output folder (string).
    % Paths: Paths to folders that must be added to Matlab search path (cell array of strings).

    % define inputs
    OutDir = '{out_dir}';
    TargetNetworkPatch = ft_read_cifti_mod('{target_mask_path}');
    PialSurfs{{1}} = '{tans_inputs.lpial}';
    PialSurfs{{2}} = '{tans_inputs.rpial}';
    SkinSurf = '{tans_inputs.skinsurf}';
    SearchGridRadius = 20;
    GridSpacing = 2;
    % run the tans_searchgrid function
    [SubSampledSearchGrid,FullSearchGrid] = tans_searchgrid(TargetNetworkPatch,PialSurfs,SkinSurf,GridSpacing,SearchGridRadius,OutDir,Paths);

    setenv('LD_LIBRARY_PATH', tmp_path);
    exit
    """

    script_path.write_text(script)
    return script_path


def find_data(X):
    X = X.copy()
    while len(X) == 1:
        X = X[0]
    return X


def get_position_info(simmat_path, subject):
    simmat = sio.loadmat(simmat_path)
    poslist = pd.DataFrame(find_data(simmat['poslist'])[-1][0])
    poslist['cx'] = poslist.centre.str[0].str[0]
    poslist['cy'] = poslist.centre.str[0].str[1]
    poslist['cz'] = poslist.centre.str[0].str[2]
    poslist['px'] = poslist.pos_ydir.str[0].str[0]
    poslist['py'] = poslist.pos_ydir.str[0].str[1]
    poslist['pz'] = poslist.pos_ydir.str[0].str[2]
    poslist['simmat_path'] = simmat_path

    poslist['idx'] = poslist.index + 1
    vol_path = poslist.idx.apply(lambda
                                     x: simmat_path.parent / f'subject_volumes/{subject}_TMS_1-{x:04d}_MagVenture_MCF-B65_scalar_magnE.nii.gz')
    vol_path_exists = vol_path.apply(lambda x: x.exists())

    poslist['vol_path'] = vol_path
    poslist['vol_path_exists'] = vol_path_exists
    poslist['max_val'] = nl.image.load_img(vol_path).get_fdata().max()
    return poslist

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            90
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            180
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180/np.pi


def load_stimgrid(magne_paths, magne_min_percentile=None, min_motor_thresh=None, maxMT=80):
    """ Load stimulation grid simulations and find verticies above some threshold.
    If the min_motor_thresh is passed, assumes that the maximum E-field in any
    simulation will be scaled to some percent (default 80) of motor threshold.
    With that assumption, we can rescale the E-field magnitude in terms of
    precent of motor threshold and then threshold based on that.

    If magne_min_percentile is passed, threshold based on percentile out of all left hemisphere
    vertices accross all stimulations.
    """
    l_magne = []
    all_magne = []
    for mp in magne_paths:
        magne_img = ci.load(mp)
        magne = magne_img.get_fdata()
        all_magne.append(magne.T)
        l_magne.append(surf_data_from_cifti(magne, magne_img.header.get_axis(1), 'CIFTI_STRUCTURE_CORTEX_LEFT'))
    l_magne = np.hstack(l_magne)
    all_magne = np.hstack(all_magne)

    l_percentile = np.argsort(np.argsort(l_magne, axis=None), axis=None).reshape(l_magne.shape)
    l_percentile = (l_percentile / (np.product(l_magne.shape) - 1)) * 100

    l_mt = (l_magne / all_magne.max(0)) * maxMT
    if magne_min_percentile is not None:
        # this controls how tightly we consider the spread of the stimulation
        # going to 99.9 has us considering many vertices outside BA46 and tens of vertices per stimulation
        # 99.99 is on the order of single digits per stimulation
        thresh = np.percentile(l_magne, magne_min_percentile)
        print(f"Threshold for {magne_min_percentile} percentile = {thresh}")

        print(f"Percentile corresponding to 1/2 of maximal e field:{1 - (l_magne >= l_magne.max() / 2).mean()}")
        l_stat = l_percentile
    if min_motor_thresh is not None:
        l_stat = l_mt
        thresh = min_motor_thresh

    pos_idxs = []
    for ix, (angle_ix, coord_ix, pos_ix) in poslist.loc[:, ['angle_ix', 'coord_ix', 'pos_ix']].iterrows():
        idxs = np.nonzero(l_stat[:, ix] > thresh)[0]
        for idx in idxs:
            row = dict(idx=idx, angle_ix=angle_ix, coord_ix=coord_ix, pos_ix=pos_ix,
                       magne=l_magne[idx, ix], percentile=l_percentile[idx, ix], motor_threshold=l_mt[idx, ix])
            pos_idxs.append(row)
    pos_idxs = pd.DataFrame(pos_idxs)
    return pos_idxs


def find_timeseries(func_dir):
    func_descs = [
        'optcomDenoisedMGTRsmoo*',
        'optcomDenoisedMGTR',
        'optcomDenoised',
        'cleanedsmoo*',
        'cleaned',
    ]
    timeseries = []
    for func_desc in func_descs:
        timeseries = sorted(func_dir.glob(f"*_desc-{func_desc}_bold.dtseries.nii"))
        if len(timeseries) > 0:
            return timeseries

    if len(timeseries) == 0:
        raise ValueError(f"Couldn't find any functionals with the right name in {func_dir}")


def find_timeseries(func_dir, run_ind=None):
    func_descs = [
        'optcomDenoisedMGTRsmoo*',
        'optcomDenoisedMGTR',
        'optcomDenoised',
        'cleanedsmoo*',
        'cleaned',
    ]
    timeseries = []
    for func_desc in func_descs:
        timeseries = sorted(func_dir.glob(f"*_desc-{func_desc}_bold.dtseries.nii"))
        if len(timeseries) > 0:
            if run_ind is not None:
                return [timeseries[run_ind]]
            return timeseries

    if len(timeseries) == 0:
        raise ValueError(f"Couldn't find any functionals with the right name in {func_dir}")


def run_stimgrid_eval(out_dir, lref_roi, rref_roi, dstim_roi,
                      poslist, pos_idxs,
                      uncertainty_fwhm=2,
                      nthreads=20, linkage='complete', block_length=45,
                      seed=132923913298132, nbootstraps=100, magne_min_percentile=99.9, prefix=None):
    out_dir = Path(out_dir)
    lref_thresh, lref_entropy, lref_clusters_stats = find_cluster_threshold(lref_roi.ts, lref_roi.connectivity,
                                                                            seed=seed, nthreads=nthreads,
                                                                            linkage=linkage, block_length=block_length,
                                                                            nbootstraps=nbootstraps)
    # lref_labels, lref_goodts, lref_goodpoints, lref_goodcoords, lref_labelcolors = cluster_and_plot(lref_roi.ts, lref_thresh, lref_roi.idxs, lref_roi.coords, lref_entropy, entropy_thresh=1000, connectivity=lref_roi.connectivity, linkage="complete", plot=False)
    # lref_clusters = get_surface_cluster_stats(lref_labels, lref_goodts, lref_goodpoints, lref_goodcoords)

    rref_thresh, rref_entropy, rref_clusters_stats = find_cluster_threshold(rref_roi.ts, rref_roi.connectivity,
                                                                            seed=seed, nthreads=nthreads,
                                                                            linkage=linkage, block_length=block_length,
                                                                            nbootstraps=nbootstraps)
    # rref_labels, rref_goodts, rref_goodpoints, rref_goodcoords, rref_labelcolors = cluster_and_plot(rref_roi.ts, rref_thresh, rref_roi.idxs, rref_roi.coords, rref_entropy, entropy_thresh=1000, connectivity=rref_roi.connectivity, linkage="complete", plot=False)
    # rref_clusters = get_surface_cluster_stats(rref_labels, rref_goodts, rref_goodpoints, rref_goodcoords)

    dstim_thresh, dstim_entropy, dstim_clusters_stats = find_cluster_threshold(dstim_roi.ts, dstim_roi.connectivity,
                                                                               seed=seed, nthreads=nthreads,
                                                                               linkage=linkage,
                                                                               block_length=block_length,
                                                                               nbootstraps=nbootstraps)

    # rref_clusters['hemi'] = 'right'
    # lref_clusters['hemi'] = 'left'
    # ref_clusters = pd.concat([rref_clusters, lref_clusters])
    # ref_repts = np.array(list(ref_clusters.repts.apply(lambda x: list(x))))

    stimmed_idxs = np.unique(pos_idxs.idx)
    addtional_idxs = stimmed_idxs[~np.isin(stimmed_idxs, dstim_roi.idxs)]
    addtional_roi = SurfROI(lref_roi.surface, 'left', lref_roi._surf_ts_data, idxs=addtional_idxs)

    all_ts = np.vstack([dstim_roi.ts, rref_roi.ts, lref_roi.ts, addtional_roi.ts])
    bs_all_ts = block_bootstrap(all_ts, nsamples=nbootstraps, block_length=block_length, seed=132923913298132)

    jobs = []
    for bsi, bts in enumerate(bs_all_ts):
        jobs.append(delayed(bootstrap_clustering)(bsi, bts, dstim_roi, dstim_thresh,
                                                  rref_roi, rref_thresh,
                                                  lref_roi, lref_thresh, other_roi=addtional_roi))

    print(f"Running bootstrap clustering, {len(jobs)}", flush=True)
    bs_dstim_verts = Parallel(n_jobs=nthreads, verbose=5)(jobs)
    bs_dstim_verts_df = pd.concat(bs_dstim_verts)
    if prefix is not None:
        bs_dstim_verts_df_outpath = out_dir / f'{prefix}desc-bsdstimverts_stat.pkl.gz'
    else:
        bs_dstim_verts_df_outpath = out_dir / f'desc-bsdstimverts_stat.pkl.gz'
    bs_dstim_verts_df.to_pickle(bs_dstim_verts_df_outpath,
                                compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})

    merged_verts = pos_idxs.merge(bs_dstim_verts_df, how='left', on='idx', indicator=True)
    assert len(merged_verts.query("_merge != 'both'")) == 0
    merged_verts = merged_verts.drop('_merge', axis=1)
    merged_verts['in_clust'] = merged_verts.cluster.notnull()

    # set merged_verts index to make joinin
    merged_verts = merged_verts.set_index('pos_ix')
    merged_verts['pos_ix'] = merged_verts.index.values
    merged_verts = merged_verts.rename(columns={'pos_ix': 'pos_ix_col'})

    posdists = squareform(pdist(poslist.loc[:, ['cx', 'cy', 'cz']].values))
    uncert_dist = stats.norm(0, uncertainty_fwhm)
    uncert_weights = uncert_dist.sf(posdists)
    uncert_lut = {}
    for uw, (ix, row) in zip(uncert_weights, poslist.iterrows()):
        uncert_lut[row.pos_ix] = dict(zip(poslist.pos_ix, (poslist.angle_ix == row.angle_ix).values * uw))
    uncert_df = pd.DataFrame(uncert_lut)

    jobs = []
    for (angle_ix, coord_ix, pos_ix), df in merged_verts.groupby(['angle_ix', 'coord_ix', 'pos_ix_col']):
        jobs.append(delayed(get_stim_stats_with_uncertainty)(pos_ix, angle_ix, coord_ix, uncert_df, merged_verts,
                                                             magne_min_percentile, nboots=nbootstraps))
    print(len(jobs))
    stim_stats = Parallel(n_jobs=nthreads, verbose=10)(jobs)
    stim_stats = pd.DataFrame(stim_stats)

    nverts = pos_idxs.groupby('pos_ix').idx.nunique().rename('n_verts')
    mtvas = pos_idxs.groupby('pos_ix').surface_area.sum().rename('surface_area')
    coords = poslist.loc[:, ['pos_ix', 'cx', 'cy', 'cz']].set_index('pos_ix')

    stim_stats_rev = stim_stats.join(nverts).join(mtvas).join(coords)
    if prefix is not None:
        stim_stats_outpath = out_dir / f'{prefix}desc-bsstim_stat.pkl.gz'
    else:
        stim_stats_outpath = out_dir / f'desc-bsstim_stat.pkl.gz'
    stim_stats_rev.to_pickle(stim_stats_outpath,
                             compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})
    return stim_stats_rev


def load_stimgrid(magne_paths, poslist, magne_min_percentile=None, min_motor_thresh=None, maxMT=80, uncertainty_fwhm=2):
    """ Load stimulation grid simulations and find verticies above some threshold.
    If the min_motor_thresh is passed, assumes that the maximum E-field in any
    simulation will be scaled to some percent (default 80) of motor threshold.
    With that assumption, we can rescale the E-field magnitude in terms of
    precent of motor threshold and then threshold based on that.

    If magne_min_percentile is passed, threshold based on percentile out of all left hemisphere
    vertices accross all stimulations.
    """
    l_magne = []
    all_magne = []
    for mp in magne_paths:
        magne_img = ci.load(mp)
        magne = magne_img.get_fdata()
        all_magne.append(magne.T)
        l_magne.append(surf_data_from_cifti(magne, magne_img.header.get_axis(1), 'CIFTI_STRUCTURE_CORTEX_LEFT'))
    l_magne = np.hstack(l_magne)
    all_magne = np.hstack(all_magne)

    l_percentile = np.argsort(np.argsort(l_magne, axis=None), axis=None).reshape(l_magne.shape)
    l_percentile = (l_percentile / (np.product(l_magne.shape) - 1)) * 100

    l_mt = (l_magne / all_magne.max(0)) * maxMT
    if magne_min_percentile is not None:
        # this controls how tightly we consider the spread of the stimulation
        # going to 99.9 has us considering many vertices outside BA46 and tens of vertices per stimulation
        # 99.99 is on the order of single digits per stimulation
        thresh = np.percentile(l_magne, magne_min_percentile)
        print(f"Threshold for {magne_min_percentile} percentile = {thresh}")

        print(f"Percentile corresponding to 1/2 of maximal e field:{1 - (l_magne >= l_magne.max() / 2).mean()}")
        l_stat = l_percentile
    if min_motor_thresh is not None:
        l_stat = l_mt
        thresh = min_motor_thresh

    posdists = squareform(pdist(poslist.loc[:, ['cx', 'cy', 'cz']].values))
    uncert_dist = stats.norm(0, uncertainty_fwhm)
    # multiply weights by 2 so that ddof caculations work better
    uncert_weights = uncert_dist.sf(posdists) * 2

    # Incorporate weights on vertices so that they can be applied later
    pos_idxs = []
    for ix, (angle_ix, coord_ix, pos_ix) in poslist.loc[:, ['angle_ix', 'coord_ix', 'pos_ix']].iterrows():
        close_pos = ((uncert_weights[pos_ix] > 1e-5) & (poslist.angle_ix == angle_ix).values)
        vert, tmppos = np.nonzero(l_stat[:, close_pos] > thresh)
        pos = np.nonzero(close_pos)[0][tmppos]
        rows = [dict(idx=ivert,
                     angle_ix=angle_ix,
                     coord_ix=coord_ix,
                     pos_ix=pos_ix,
                     opos_ix=ipos,
                     uncert_weight=iweight,
                     magne=imagne,
                     percentile=ipct,
                     motor_threshold=imt
                     )
                for ivert, ipos, iweight, imagne, ipct, imt
                in zip(
                vert,
                pos,
                uncert_weights[ix, pos],
                l_magne[vert, pos],
                l_percentile[vert, pos],
                l_mt[vert, pos]
            )]
        pos_idxs.extend(rows)

    pos_idxs = pd.DataFrame(pos_idxs)
    # take the biggest weight for each vertex
    pos_idxs = pos_idxs.sort_values('uncert_weight', ascending=False).groupby(['pos_ix', 'idx']).first().reset_index()
    m = -0.9 / (min_motor_thresh - maxMT)
    b = 1 - m * maxMT
    pos_idxs['intensity_weight'] = pos_idxs.motor_threshold * m + b
    pos_idxs['weight'] = pos_idxs.uncert_weight * pos_idxs.intensity_weight
    return pos_idxs


def get_norm(idx, points, G, wm_points=None, pial_points=None, ):
    idx_coords = points[idx]
    neighbor_coords = points[list(G.neighbors(idx))]
    # loop through all pairs of neighboring vertices and find the distance from idx to each line
    a, b = zip(*[[inc, jnc] for ii, inc in enumerate(neighbor_coords) for jnc in neighbor_coords[ii + 1:]])
    a = np.array(a)
    b = np.array(b)
    p = idx_coords.reshape(1, -1)
    dists_from_line = squareform(np.linalg.norm(np.cross((p - a), (p - b)), axis=1) / np.linalg.norm(b - a, axis=1))

    # just to get the diagonal out of the way
    # find the pair with the shortest distance from idx to line
    dists_from_line[dists_from_line == 0] = np.nan
    p00, p01 = np.nonzero(dists_from_line == np.nanmin(dists_from_line))[0]
    c00 = neighbor_coords[p00]
    c01 = neighbor_coords[p01]
    v0 = c01 - c00

    # delete the shortest pair, find the second shortest pair
    tmp = np.delete(np.delete(dists_from_line, [p00, p01], axis=0), [p00, p01], axis=1)
    p10, p11 = np.nonzero(dists_from_line == np.nanmin(tmp))[0]
    c10 = neighbor_coords[p10]
    c11 = neighbor_coords[p11]
    v1 = c11 - c10

    norm_vec = np.cross(v1, v0)
    norm_vec /= np.linalg.norm(norm_vec)
    assert np.linalg.norm(np.cross((p - a), (p - b)), axis=1).all()

    # check that norm_vec is in the right direction by loading up the white and pial surfs
    # norm vec should be towards white surf and away from pial
    if (wm_points is not None) and (pial_points is not None):
        pial_idx_coords = pial_points[idx]
        wm_idx_coords = wm_points[idx]
        idx_plus_norm = idx_coords + norm_vec
        pial_dist = np.linalg.norm(pial_idx_coords - idx_plus_norm)
        wm_dist = np.linalg.norm(wm_idx_coords - idx_plus_norm)
        if wm_dist > pial_dist:
            norm_vec = - norm_vec
    return norm_vec

def patient_delete(directory, pause=60, nattempts=10):
    """Delete a directory that may have a process still writing to it by attempting to delete once
    every 60 seconds for 10 minutes."""

    directory = Path(directory)
    n = 0
    while directory.exists() and n < nattempts:
        try:
            shutil.rmtree(directory)
        except OSError:
            time.sleep(pause)
            n += 1
    # Try to rmtree one last time so any error gets triggered and can be dealt with elsewhere.
    shutil.rmtree(directory)

def run_opt_and_save_outputs(o):
    # clear any preexisting handlers
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    try:
        run_simnibs(o)
        outpath = o.pathfem
    except ValueError:
        try:
            outpath = o.pathfem + '_ts-4'
            no = opt_struct.TMSoptimize()
            no.fnamehead = o.fnamehead
            no.pathfem = outpath
            no.fnamecoil = o.fnamecoil
            no.target = o.target
            no.target_direction = o.target_direction
            no.distance = o.distance
            no.method = o.method
            # o.map_to_vol = True
            # o.map_to_surf = True
            # o.fields = 'eE'
            no.angle_resolution = o.angle_resolution
            no.spatial_resolution = o.spatial_resolution
            no.solver_options = 'paradiso'
            no.open_in_gmsh = False
            no.target_size = 4
            run_simnibs(no)
        except ValueError:
            outpath = o.pathfem + '_ts-6'
            no = opt_struct.TMSoptimize()
            no.fnamehead = o.fnamehead
            no.pathfem = outpath
            no.fnamecoil = o.fnamecoil
            no.target = o.target
            no.target_direction = o.target_direction
            no.distance = o.distance
            no.method = o.method
            # o.map_to_vol = True
            # o.map_to_surf = True
            # o.fields = 'eE'
            no.angle_resolution = o.angle_resolution
            no.spatial_resolution = o.spatial_resolution
            no.solver_options = 'paradiso'
            no.open_in_gmsh = False
            no.target_size = 6
            run_simnibs(no)

    # extract some variables
    reference = Path(o.fnamehead).parent
    coil_name = Path(o.fnamecoil).parts[-1].split('.')[0]
    subject = reference.parts[-1].split('m2m_')[-1]

    # get the best position from the log
    pos_log = sorted(Path(outpath).glob('simnibs_*.log'))[-1].read_text()

    bestmat = []
    for il, ll in enumerate(pos_log.split('Best coil position\n=============================\n')[-1].split('\n')):
        row = []
        for vv in ll.split():
            try:
                row.append(float(vv.strip('[] \t\n')))
            except ValueError:
                continue
        if len(row) > 0:
            bestmat.append(row)
        if il == 3:
            break
    bestmat = np.array(bestmat)

    # save the best mat
    matout = Path(outpath) / f"{subject}_TMS_optimize_{coil_name}_optmat.npy"
    np.save(matout, bestmat, allow_pickle=False)

    # save the volumes
    outmesh = Path(outpath) / f"{subject}_TMS_optimize_{coil_name}.msh"
    vol_out = Path(outpath) / f'subject_volumes/{subject}_TMS_1-0001_{coil_name}'
    vol_out.parent.mkdir(exist_ok=True)
    interpolate_to_volume(outmesh.as_posix(), reference.as_posix(), vol_out.as_posix(),
                          keep_tissues=[2])

    # save surfaces
    surf_out = vol_out.parent.parent / 'subject_overlays'
    surf_out.mkdir(exist_ok=True)
    middle_gm_interpolation(outmesh.as_posix(), reference.as_posix(), surf_out.as_posix(),
                            quantities=['magn', 'normal'], fields='E')


def get_steepest_ascent(start, G, metric, coords, min_rise=1, search_radius=5):
    path = [start]
    n_idxs = []
    # for nn in G.neighbors(start):
    #     n_idxs.extend(list(G.neighbors(nn)))
    # n_idxs = np.array(list(G.neighbors(start)))
    # n_idxs = np.unique(n_idxs)
    # n_idxs = n_idxs[n_idxs != start]
    dists = cdist([coords[start]], coords).squeeze()
    n_idxs = np.nonzero(dists < search_radius)[0]
    n_idxs = np.unique(n_idxs)
    nG = G.subgraph(n_idxs)
    n_idxs = n_idxs[np.array([nx.has_path(nG, start, nn) for nn in n_idxs])]
    good_path = []
    for nix in n_idxs:
        nsp = nx.shortest_path(G, start, nix)
        good_path.append((metric[nsp[1:]] < metric[nsp[:1]]).all())
    # skip inner ring of neighbors if we're in a local minima
    if np.sum(good_path) >= 1:
        good_path = []
        for nix in n_idxs:
            nsp = nx.shortest_path(G, start, nix)
            good_path.append((metric[nsp[2:]] < metric[nsp[:-2]]).all())
    n_idxs = n_idxs[good_path]
    n_idxs = n_idxs[n_idxs != start]
    n_sulcs = metric[n_idxs]
    i_sulc = metric[start]
    sulc_difs = i_sulc - n_sulcs
    greatest_rise = sulc_difs.max()
    gr_idx = n_idxs[sulc_difs == greatest_rise][0]
    gr_dir = coords[gr_idx] - coords[start]
    gr_dir /= np.linalg.norm(gr_dir)
    next_rise = min_rise + 1

    while next_rise > min_rise:
        idx = path[-1]
        n_idxs = np.array(list(G.neighbors(idx)))
        n_sulcs = metric[n_idxs]
        i_sulc = metric[idx]
        sulc_difs = i_sulc - n_sulcs
        n_dirs = coords[n_idxs] - coords[idx]
        n_dirs /= np.linalg.norm(n_dirs, axis=-1).reshape(-1, 1)
        n_cos = np.array([np.dot(gr_dir, nd) for nd in n_dirs])
        best_idx = n_idxs[n_cos == np.max(n_cos)][0]
        next_rise = metric[idx] - metric[best_idx]
        if next_rise > 0:
            path.append(best_idx)
    return np.array(path)