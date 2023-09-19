import typing
import shutil
from pathlib import Path
from pkg_resources import resource_filename
import os
import subprocess
from collections import namedtuple
import templateflow
from niworkflows.interfaces.cifti import _prepare_cifti, CIFTI_STRUCT_WITH_LABELS
import warnings
import json
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import nilearn as nl
from nilearn.image import resample_to_img
from niworkflows.interfaces.nibabel import reorient_image
from nibabel import cifti2 as ci
import nibabel as nb
import numpy as np
from contarg.utils import (
    make_fmriprep_t2,
    parse_bidsname,
    build_bidsname,
    get_stimroi_path,
    get_refroi_path,
    transform_mask_to_t1w,
    find_bids_files,
    update_bidspath
)
import configparser
import pandas as pd
from typing import Union, Optional


def write_headmodel_script(
    t1w_path,
    t2w_path,
    out_dir,
    subject,
    msc_path=None,
    simnibs_path=None,
    tans_path=None,
):
    if subject[:4] == "sub-":
        subject = subject[4:]

    out_dir = Path(out_dir)
    m_dir = Path(out_dir / "matlab_scripts")
    m_dir.mkdir(exist_ok=True)
    if t2w_path is None:
        t2w_path = "[]"
    else:
        t2w_path = f"'{t2w_path}'"

    headmodels_script_path = m_dir / "run_headmodels.m"
    if msc_path is None:
        msc_path = os.getenv("CONTARG_MSC_PATH")
    if simnibs_path is None:
        simnibs_path = os.getenv("CONTARG_SIMNIBS_PATH")
    if tans_path is None:
        tans_path = resource_filename(
            "contarg", "resources/Targeted-Functional-Network-Stimulation"
        )

    headmodels_script = f"""
    %% Example use of Targeted Functional Network Stimulation ("TANS")
    % "Automated optimization of TMS coil placement for personalized functional
    % network engagement" - Lynch et al., 2022 (Neuron)

    % define some paths
    Paths{{1}} = '{Path(simnibs_path)/'simnibs/simnibs/matlab_tools'}'; % download from https://simnibs.github.io/simnibs/build/html/index.html
    Paths{{2}} = '{msc_path}'; % this folder contains ft_read / gifti functions for reading and writing cifti files (e.g., https://github.com/MidnightScanClub/MSCcodebase).
    Paths{{3}} = '{tans_path}'; %


    % add folders
    % to search path;
    for i = 1:length(Paths)
        addpath(genpath(Paths{{i}}));
    end

    % clear matlab's ldlibrary path
    tmp_path = getenv('LD_LIBRARY_PATH');
    setenv('LD_LIBRARY_PATH', '');
    setenv('MPLBACKEND', '');
    % If successful, each of the commands below should return status as 0.

    % Confirm various software is available
    [status,~] = system('mris_convert -version'); % freesurfer
    assert(status == 0, 'Freesurfer not available')
    [status,~] = system('charm -v'); % simnibs: note charm program replaced headreco in SimNIBS v4.0.
    assert(status == 0, 'SimNIBS not available')
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

    % Create the head model
    % Timing: Typically 1-2 hours.

    % Use the tans_headmodels function to create a head model for electric field (E-field) modeling
    % using the CHARM method (Puonti et al. 2020). In addition, this function generates skin
    % surface geometry files (*.surf.gii) for visualizing certain results.

    % The inputs to the function are:
    % Subject: The subject ID (string).
    % T1w: Path to T1w-weighted anatomical image (string).
    % T2w: Path to T2w-weighted anatomical image (string).
    % OutDir : Path to the output folder (string).
    % Paths: Paths to folders that must be added to Matlab search path (cell array of strings).

    % run the tans_headmodels function;

    tans_headmodels('{subject}','{t1w_path}',{t2w_path},'{out_dir}',Paths);
    setenv('LD_LIBRARY_PATH', tmp_path);
    exit

    """
    headmodels_script_path.write_text(headmodels_script)
    return headmodels_script_path


TansInputs = namedtuple(
    "TansInputs",
    "sulc "
    "lmidthicksurf "
    "rmidthicksurf "
    "vertex_surface "
    "lpial "
    "rpial "
    "lwhite "
    "rwhite "
    "skinsurf "
    "lmedialwall "
    "rmedialwall "
    "headmesh "
    "coil",
)


def get_tans_inputs(tans_input_dir, subject):
    sulc = f"{tans_input_dir}/anat/sub-{subject}.sulc.32k_fs_LR.dscalar.nii"
    if not Path(sulc).exists():
        raise FileNotFoundError(f"{sulc} does not exist")
    lmidthicksurf = (
        f"{tans_input_dir}/anat/sub-{subject}.L.midthickness.32k_fs_LR.surf.gii"
    )
    if not Path(lmidthicksurf).exists():
        raise FileNotFoundError(f"{lmidthicksurf} does not exist")
    rmidthicksurf = (
        f"{tans_input_dir}/anat/sub-{subject}.R.midthickness.32k_fs_LR.surf.gii"
    )
    if not Path(rmidthicksurf).exists():
        raise FileNotFoundError(f"{rmidthicksurf} does not exist")
    vertex_surface = (
        f"{tans_input_dir}/anat/sub-{subject}.midthickness_va.32k_fs_LR.dscalar.nii"
    )
    if not Path(vertex_surface).exists():
        raise FileNotFoundError(f"{vertex_surface} does not exist")
    lpial = f"{tans_input_dir}/anat/sub-{subject}.L.pial.32k_fs_LR.surf.gii"
    if not Path(lpial).exists():
        raise FileNotFoundError(f"{lpial} does not exist")
    rpial = f"{tans_input_dir}/anat/sub-{subject}.R.pial.32k_fs_LR.surf.gii"
    if not Path(rpial).exists():
        raise FileNotFoundError(f"{rpial} does not exist")
    lwhite = f"{tans_input_dir}/anat/sub-{subject}.L.white.32k_fs_LR.surf.gii"
    if not Path(lwhite).exists():
        raise FileNotFoundError(f"{lwhite} does not exist")
    rwhite = f"{tans_input_dir}/anat/sub-{subject}.R.white.32k_fs_LR.surf.gii"
    if not Path(rwhite).exists():
        raise FileNotFoundError(f"{rwhite} does not exist")
    skinsurf = f"{tans_input_dir}/HeadModel/m2m_{subject}/Skin.surf.gii"
    if not Path(skinsurf).exists():
        raise FileNotFoundError(f"{skinsurf} does not exist")
    lmedialwall = templateflow.api.get(
        "fsLR",
        hemi="L",
        density="32k",
        desc="nomedialwall",
        suffix="dparc",
        extension="label.gii",
    )
    rmedialwall = templateflow.api.get(
        "fsLR",
        hemi="R",
        density="32k",
        desc="nomedialwall",
        suffix="dparc",
        extension="label.gii",
    )
    headmesh = f"{tans_input_dir}/HeadModel/m2m_{subject}/{subject}.msh"
    if not Path(headmesh).exists():
        raise FileNotFoundError(f"{headmesh} does not exist")
    config_path = Path.home() / ".contarg"
    config = configparser.ConfigParser()
    config.read(config_path)
    # TODO: consider making this work outside of biowulf
    coil = (
        Path(config["TANS"]["SimNIBSPath"])
        / "../conda/envs/4.0/lib/python3.9/site-packages/simnibs/resources/coil_models/Drakaki_BrainStim_2022/MagVenture_MCF-B65.ccd"
    )
    if not Path(coil).exists():
        raise ValueError(f"{coil} does not exist")
    result = TansInputs(
        sulc,
        lmidthicksurf,
        rmidthicksurf,
        vertex_surface,
        lpial,
        rpial,
        lwhite,
        rwhite,
        skinsurf,
        lmedialwall,
        rmedialwall,
        headmesh,
        coil,
    )
    return result


def write_sim_script(
    tans_input_dir,
    target_mask_path,
    stim_path,
    out_dir,
    subject,
    msc_path=None,
    simnibs_path=None,
    tans_path=None,
    nthreads=20,
):



    out_dir = Path(out_dir)
    m_dir = Path(out_dir / "matlab_scripts")
    m_dir.mkdir(exist_ok=True)

    tans_inputs = get_tans_inputs(tans_input_dir, subject)

    sim_script_path = m_dir / "run_sim.m"
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

    sim_script = f"""
    %% Example use of Targeted Functional Network Stimulation ("TANS")
    % "Automated optimization of TMS coil placement for personalized functional
    % network engagement" - Lynch et al., 2022 (Neuron)

    % define some paths
    Paths{{1}} = '{Path(simnibs_path)/'simnibs/simnibs/matlab_tools'}'; % download from https://simnibs.github.io/simnibs/build/html/index.html
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
    [status,~] = system('charm -v'); % simnibs: note charm program replaced headreco in SimNIBS v4.0.
    assert(status == 0, 'SimNIBS not available')
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
    
    TargetNetwork = ft_read_cifti_mod('{target_mask_path}');
    
    % load cortical mask representing the potential stimulation space;
    SearchSpace = ft_read_cifti_mod('{stim_path}');
    
    % load sulcal depth information;
    Sulc = ft_read_cifti_mod('{tans_inputs.sulc}');
    BrainStructure = SearchSpace.brainstructure; % extract the brain structure index
    Sulc.data(BrainStructure==-1) = []; % remove medial wall vertices present in this file.
    
    % define input variables;
    MidthickSurfs{{1}} = '{tans_inputs.lmidthicksurf}';
    MidthickSurfs{{2}} = '{tans_inputs.rmidthicksurf}';
    VertexSurfaceArea = ft_read_cifti_mod('{tans_inputs.vertex_surface}');
    OutDir = '{out_dir}';
    
    % run the tans_roi function
    tans_roi(TargetNetwork,MidthickSurfs,VertexSurfaceArea,Sulc,SearchSpace,OutDir,Paths);
    
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
    TargetNetworkPatch = ft_read_cifti_mod([OutDir '/ROI/TargetNetworkPatch.dtseries.nii']);
    PialSurfs{{1}} = '{tans_inputs.lpial}';
    PialSurfs{{2}} = '{tans_inputs.rpial}';
    SkinSurf = '{tans_inputs.skinsurf}';
    SearchGridRadius = 20;
    GridSpacing = 2;
    % run the tans_searchgrid function
    [SubSampledSearchGrid,FullSearchGrid] = tans_searchgrid(TargetNetworkPatch,PialSurfs,SkinSurf,GridSpacing,SearchGridRadius,OutDir,Paths);
    
    % Perform electric field modeling iteratively in the search grid.
    % Timing: Depends on the total number of simulations performed and available parallel pools in Matlab, typically several hours.
    
    % Use the tans_simnibs function to perform electric field simulations at each point in the search grid using
    % SimNibs (Thielscher et al., 2015). A number of coil orientations determined by an angle (in degrees)
    % specified by the user are performed. The strength of the electric field is mapped to the individual’s midthickness surface.
    
    % The inputs to the function are:
    % SearchGridCoords: Number of coil positions x 3 numeric array generated by tans_searchgrid function.
    % HeadMesh: Path to tetrahedral head mesh (string).
    % CoilModel: Name of the coil model (string). Must be available in SimNIBS library ( /SimNIBS-3.2/simnibs/ccd-files/).
    % AngleResolution: Inter angle resolution used in the search grid in mm (numeric).
    % SkinSurf: Path to the skin surface geometry file (string).
    % MidthickSurfs: Paths to low (32k) dimensional FS_LR midthickness surfaces (Cell array of strings, MidthickSurfs{1} = path to LH, MidthickSurfs{2} = path to RH).
    % WhiteSurfs: Paths to low (32k) dimensional FS_LR white surfaces (Cell array of strings, WhiteSurfs{1} = path to LH, WhiteSurfs{2} = path to RH).
    % PialSurfs: Paths to low (32k) dimensional FS_LR pial surfaces (Cell array of strings, PialSurfs{1} = path to LH, PialSurfs{2} = path to RH).
    % MedialWallMasks: Paths to low (32k) dimensional FS_LR medial wall masks (Cell array of strings, MedialWallMasks{1} = path to LH, MedialWallMasks{2} = path to RH).
    % nThreads: The number of parallel pools that will be used in Matlab (numeric).
    % OutDir: Path to the output folder (string).
    % Paths: Paths to folders that must be added to Matlab search path (cell array of strings).
    
    % define inputs
    SearchGridCoords = SubSampledSearchGrid;
    PialSurfs{{1}} = '{tans_inputs.lpial}';
    PialSurfs{{2}} = '{tans_inputs.rpial}';
    WhiteSurfs{{1}} = '{tans_inputs.lwhite}';
    WhiteSurfs{{2}} = '{tans_inputs.rwhite}';
    MidthickSurfs{{1}} = '{tans_inputs.lmidthicksurf}';
    MidthickSurfs{{2}} = '{tans_inputs.rmidthicksurf}';
    MedialWallMasks{{1}} = '{tans_inputs.lmedialwall}';
    MedialWallMasks{{2}} = '{tans_inputs.rmedialwall}';
    SkinSurf = '{tans_inputs.skinsurf}';
    HeadMesh = '{tans_inputs.headmesh}';
    CoilModel = '{tans_inputs.coil}';
    DistanceToScalp = 2;
    AngleResolution = 30;
    nThreads = {nthreads};
    
    % run the tans_simnibs function
    tans_simnibs(SearchGridCoords,HeadMesh,CoilModel,AngleResolution,DistanceToScalp,SkinSurf,MidthickSurfs,WhiteSurfs,PialSurfs,MedialWallMasks,nThreads,OutDir,Paths, SIMNIBSPYTHON, SIMNIBSDIR);
    
    % Note: A fixed stimulation intensity (𝑑𝐼/𝑑𝑡 = 1 A/µs) is used during this stage of TANS because the strength of the E-field
    % varies linearly with 𝑑𝐼/𝑑𝑡 (the speed of variation of the current throughout the coil) and, for this reason,
    % has no effect on its spatial distribution (including where it is maximal relative to the target).
    
    setenv('LD_LIBRARY_PATH', tmp_path);
    exit
    """

    sim_script_path.write_text(sim_script)
    return sim_script_path


def write_optimization_script(
    tans_input_dir,
    target_mask_path,
    searchgrid_path,
    out_dir,  # TODO: for right now this has to be the same as the sim out_dir
    subject,
    msc_path=None,
    simnibs_path=None,
    tans_path=None,
    avoidance_mask_path=None,
):
    if subject[:4] == "sub-":
        subject = subject[4:]

    out_dir = Path(out_dir)
    m_dir = Path(out_dir / "matlab_scripts")
    m_dir.mkdir(exist_ok=True)

    tans_inputs = get_tans_inputs(tans_input_dir, subject)

    optimization_script_path = m_dir / "run_optimization.m"
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

    optimization_script = f"""
    %% Adapted from: Example use of Targeted Functional Network Stimulation ("TANS")
    % "Automated optimization of TMS coil placement for personalized functional
    % network engagement" - Lynch et al., 2022 (Neuron)

    % define some paths
    Paths{{1}} = '{Path(simnibs_path)/'simnibs/simnibs/matlab_tools'}'; % download from https://simnibs.github.io/simnibs/build/html/index.html
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
    [status,~] = system('charm -v'); % simnibs: note charm program replaced headreco in SimNIBS v4.0.
    assert(status == 0, 'SimNIBS not available')
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
    
    % Find the coil placement that best aligns the electric field hotspot with the target network
    % Timing: Depends on the total number of simulations performed, typically 1-2 hours.
    
    % Use the tans_optimize function to find which of the coil placements submitted to the tans_simnibs function
    % produced the greatest on-target value (calculated as the proportion of the E-field hotspot inside the
    % target network). In other words, which coil placement was best for maximizing stimulation specificity.
    
    % The inputs to the function are:
    % Subject: The subject ID (string).
    % TargetNetwork: A CIFTI file containing the functional network of interest (structure array). Non-zero values in TargetNetwork.data are considered target network vertices.
    % AvoidanceRegion: A CIFTI file indexing the brain regions or networks of no interest (structure array). Non-zero values in AvoidanceRegion.data are considered non-target network vertices.
    % PercentileThresholds: A range of percentiles used for operationalizing the E-field hotspot (numeric). For example, linspace(99.9,99,10).
    % SkinSurf: Path to the skin surface geometry file (string).
    % VertexSurfaceArea: A CIFTI file containing the vertex surface areas (structure array).
    % MidthickSurfs: Paths to low (32k) dimensional FS_LR midthickness surfaces (Cell array of strings, MidthickSurfs{1} = path to LH, MidthickSurfs{2} = path to RH).
    % WhiteSurfs: Paths to low (32k) dimensional FS_LR white surfaces (Cell array of strings, WhiteSurfs{1} = path to LH, WhiteSurfs{2} = path to RH).
    % PialSurfs: Paths to low (32k) dimensional FS_LR pial surfaces (Cell array of strings, PialSurfs{1} = path to LH, PialSurfs{2} = path to RH).
    % MedialWallMasks: Paths to low (32k) dimensional FS_LR medial wall masks (Cell array of strings, MedialWallMasks{1} = path to LH, MedialWallMasks{2} = path to RH).
    % AngleResolution: Inter angle resolution used to fine tune coil orientation (numeric).
    % HeadMesh: Path to tetrahedral head mesh (string).
    % PositionUncertainity: Coil center positioning uncertainty (in mm, numeric). Optimal coil placement is defined as the position that maximizes the on target value on average within this distance.
    % CoilModel: Path to the coil model (string). Must be available in SimNIBS library (/SimNIBS-3.2/simnibs/ccd-files/).
    % OutDir: Path to the output folder (string).
    % Paths: Paths to folders that must be added to Matlab search path (cell array of strings).
    
    % define inputs
    PialSurfs{{1}} = '{tans_inputs.lpial}';
    PialSurfs{{2}} = '{tans_inputs.rpial}';
    WhiteSurfs{{1}} = '{tans_inputs.lwhite}';
    WhiteSurfs{{2}} = '{tans_inputs.rwhite}';
    MidthickSurfs{{1}} = '{tans_inputs.lmidthicksurf}';
    MidthickSurfs{{2}} = '{tans_inputs.rmidthicksurf}';
    VertexSurfaceArea = ft_read_cifti_mod('{tans_inputs.vertex_surface}');
    MedialWallMasks{{1}} = '{tans_inputs.lmedialwall}';
    MedialWallMasks{{2}} = '{tans_inputs.rmedialwall}';
    SearchGrid = '{searchgrid_path}';
    SkinSurf = '{tans_inputs.skinsurf}';
    HeadMesh = '{tans_inputs.headmesh}';
    OutDir = '{out_dir}';
    PercentileThresholds = linspace(99.9,99,10);
    CoilModel = '{tans_inputs.coil}';
    DistanceToScalp = 2;
    Uncertainty = 5;
    AngleResolution = 5;
    
    % isolate the target network again
    TargetNetwork = ft_read_cifti_mod('{target_mask_path}');
    
    % run the "tans_optimize.m" module;
    tans_optimize('{subject}',TargetNetwork,[],PercentileThresholds,SearchGrid,DistanceToScalp,SkinSurf,VertexSurfaceArea,MidthickSurfs,WhiteSurfs,PialSurfs,MedialWallMasks,HeadMesh,AngleResolution,Uncertainty,CoilModel,OutDir,Paths,SIMNIBSPYTHON,SIMNIBSDIR);

    setenv('LD_LIBRARY_PATH', tmp_path);
    exit
    """

    optimization_script_path.write_text(optimization_script)
    return optimization_script_path


def surface_resample(surface, subject, subjects_dir, out_dir, h, overwrite=False):
    if subject[:4] == "sub-":
        subject = subject[4:]

    H = h.upper()
    subj_dir = subjects_dir / f"sub-{subject}"
    surf_dir = subj_dir / "surf"
    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()

    sma_dir = Path(resource_filename("contarg", "data/standard_mesh_atlases"))

    new_sphere = sma_dir / f"fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii"
    current_sphere = surf_dir / f"{h}h.sphere.reg.surf.gii"
    surf_in = surf_dir / f"{h}h.{surface}.gii"
    surf_out = out_dir / f"sub-{subject}.{H}.{surface}.32k_fs_LR.surf.gii"
    if not surf_out.exists() or overwrite:
        cmd = [
            "wb_command",
            "-surface-resample",
            surf_in,
            current_sphere,
            new_sphere,
            "BARYCENTRIC",
            surf_out,
        ]
        subprocess.run(cmd, check=True)


def create_dense_scalar(metric, subject, subjects_dir, out_dir, overwrite=False):
    if subject[:4] == "sub-":
        subject = subject[4:]

    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()

    metric_Lin = out_dir / f"sub-{subject}.L.{metric}.32k_fs_LR.shape.gii"
    metric_Rin = out_dir / f"sub-{subject}.R.{metric}.32k_fs_LR.shape.gii"
    metric_out = out_dir / f"sub-{subject}.{metric}.32k_fs_LR.dscalar.nii"
    if not metric_out.exists() or overwrite:
        cmd = [
            "wb_command",
            "-cifti-create-dense-scalar",
            metric_out,
            "-left-metric",
            metric_Lin,
            "-right-metric",
            metric_Rin,
        ]
        subprocess.run(cmd, check=True)


def metric_resample(
    metric, subject, subjects_dir, h, out_dir, new_metric_name=None, overwrite=False
):
    if subject[:4] == "sub-":
        subject = subject[4:]

    if new_metric_name is None:
        new_metric_name = metric
    H = h.upper()
    subj_dir = subjects_dir / f"sub-{subject}"
    surf_dir = subj_dir / "surf"
    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()
    sma_dir = Path(resource_filename("contarg", "data/standard_mesh_atlases"))

    new_sphere = sma_dir / f"fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii"
    current_area = surf_dir / f"{h}h.midthickness.surf.gii"
    new_area = out_dir / f"sub-{subject}.{H}.midthickness.32k_fs_LR.surf.gii"
    current_sphere = surf_dir / f"{h}h.sphere.reg.surf.gii"
    metric_in = surf_dir / f"{h}h.{metric}.gii"
    metric_out = out_dir / f"sub-{subject}.{H}.{new_metric_name}.32k_fs_LR.shape.gii"
    if not metric_out.exists() or overwrite:
        cmd = [
            "wb_command",
            "-metric-resample",
            metric_in,
            current_sphere,
            new_sphere,
            "ADAP_BARY_AREA",
            metric_out,
            "-area-surfs",
            current_area,
            new_area,
        ]
        subprocess.run(cmd, check=True)


def make_metric_gifti(metric, subject, subjects_dir, h, overwrite=False):
    if subject[:4] == "sub-":
        subject = subject[4:]

    subjects_dir = Path(subjects_dir)
    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()
    subj_dir = subjects_dir / f"sub-{subject}"
    surf_dir = subj_dir / "surf"
    sval = surf_dir / f"{h}h.{metric}"
    if not sval.exists():
        raise ValueError(
            f"Couldn't find {sval}, make sure you've got your paths correct"
        )
    tval = sval.as_posix() + ".gii"
    if not Path(tval).exists() or overwrite:
        cmd = [
            "mri_surf2surf",
            "--srcsubject",
            f"sub-{subject}",
            "--trgsubject",
            f"sub-{subject}",
            "--hemi",
            f"{h}h",
            "--sval",
            sval,
            "--tval",
            tval,
        ]
        subprocess.run(cmd, check=True)


def make_surf_gifti(metric, subject, subjects_dir, h, overwrite):
    if subject[:4] == "sub-":
        subject = subject[4:]

    subjects_dir = Path(subjects_dir)
    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()
    subj_dir = subjects_dir / f"sub-{subject}"
    surf_dir = subj_dir / "surf"
    sval = surf_dir / f"{h}h.{metric}"
    if not sval.exists():
        raise ValueError(
            f"Couldn't find {sval}, make sure you've got your paths correct"
        )
    tval = sval.as_posix() + ".gii"
    if not Path(tval).exists() or overwrite:
        cmd = ["mris_convert", sval, tval]
        subprocess.run(cmd, check=True)


def freesurfer_resample_prep(subject, subjects_dir, h, out_dir, overwrite=False):
    if subject[:4] == "sub-":
        subject = subject[4:]

    H = h.upper()
    subj_dir = subjects_dir / f"sub-{subject}"
    surf_dir = subj_dir / "surf"
    os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()
    fs_white = surf_dir / f"{h}h.white"
    fs_pial = surf_dir / f"{h}h.pial"
    current_sphere = surf_dir / f"{h}h.sphere.reg"

    sma_dir = Path(resource_filename("contarg", "data/standard_mesh_atlases"))

    new_sphere = sma_dir / f"fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii"
    midthickness_current_out = surf_dir / f"{h}h.midthickness.surf.gii"
    midthickness_new_out_fs = (
        out_dir / f"sub-{subject}.{H}.midthickness.32k_fs_LR.surf.gii"
    )
    midthickness_new_out = (
        out_dir / f"sub-{subject}.{H}.midthickness.32k_fs_LR.surf.gii"
    )
    current_gifti_sphere_out = surf_dir / f"{h}h.sphere.reg.surf.gii"
    if overwrite or not midthickness_new_out_fs.exists():
        cmd = f"wb_shortcuts -freesurfer-resample-prep {fs_white} {fs_pial} {current_sphere} {new_sphere} {midthickness_current_out} {midthickness_new_out_fs} {current_gifti_sphere_out}"
        subprocess.run(cmd.split(), check=True)
    if not midthickness_new_out.exists():
        shutil.copy(midthickness_new_out_fs, midthickness_new_out)


def strip_roi_fn(roi_fn):
    return (
        roi_fn.replace(".nii.gz", "")
        .replace("_space-MNI152NLin6Asym", "")
        .replace("_res-02", "")
    )


def roi_to_surface(roi, H, out_file):
    # std_dir = Path(resource_filename("contarg", "data/standard_mesh_atlases"))
    std = templateflow.api.get(
        "fsLR", hemi=H, density="32k", suffix="midthickness", extension="surf.gii"
    )

    cmd = ["wb_command", "-volume-to-surface-mapping", roi, std, out_file, "-enclosing"]
    subprocess.run(cmd, check=True)


def mask_to_cifti(mask, out_dir, **kwargs):
    """
    Create a cifti from a volumetric image.

    Parameters
    ----------
    mask : str or Path
        A single volume image in space-MNI152NLin6Asym_res-02 that you want to project to space-fs_LR.L.midthickness_va_avg.32k_fs_LR.
    out_dir : str or Path
        The directory where gifti surfaces and cifti will be written. Will be created if it does not exist.
    kwargs are passed to _create_cifti_image
    """
    # deal with inputs
    mask = Path(mask)
    if not mask.exists():
        raise ValueError(f"{mask} does not exist")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # create output names
    mask_name = mask.parts[-1]
    stripped = strip_roi_fn(mask_name)
    l_mask_surf_path = out_dir / (stripped + "_hemi-L_space-fsLR_den-32k.func.gii")
    r_mask_surf_path = out_dir / (stripped + "_hemi-R_space-fsLR_den-32k.func.gii")
    cifti_path = out_dir / (stripped + "_space-fsLR_den-91k.dtseries.nii")

    # make surfaces
    roi_to_surface(mask, "L", l_mask_surf_path)
    roi_to_surface(mask, "R", r_mask_surf_path)

    # get info for cifti
    surface_labels, volume_labels, metadata = _prepare_cifti("91k")

    # write cifti
    _create_cifti_image(
        mask,
        volume_labels,
        [l_mask_surf_path, r_mask_surf_path],
        surface_labels,
        metadata,
        cifti_path,
        **kwargs,
    )
    return cifti_path

def mask_to_fsLR(out_dir, roi_name, subjects_dir, subject, session, fmriprep_dir, anat_dir, **kwargs):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    subjects_dir = Path(subjects_dir)
    fmriprep_dir = Path(fmriprep_dir)

    if subject[:4] == "sub-":
        subject = subject[4:]
    try:
        if session[:4] == "ses-":
            session = session[4:]
    except TypeError:
        pass
    # Try MNIvol to T1wvol to freesurfer surf to sub fsLR
    # MNIvol to T1wvolx
    try:
        inmask = get_stimroi_path(roi_name)
    except ValueError:
        inmask = get_refroi_path(roi_name)
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
                fmriprep_dir / f"sub-{subject}", debug=True, ses="*", type="anat", suffix="T1w", extension=".nii.gz",
            )
            raise ValueError(f"Couldn't find a T1.")
    reference_image = t1_paths[0]
    tfm_updates = {
        "from": "MNI152NLin6Asym",
        "to": "T1w",
        "mode": "image",
        "suffix": "xfm",
        "extension": "h5"
    }
    transforms = [update_bidspath(reference_image, fmriprep_dir, tfm_updates, exists=True, exclude='desc').as_posix()]

    t1w_mask = out_dir / f"sub-{subject}_ses-{session}_task-rest_atlas-Coords_space-T1w_desc-{roi_name}Clean_mask.nii.gz"

    transform_mask_to_t1w(inmask=inmask, reference=reference_image, transforms=transforms, output_image=t1w_mask.as_posix())
    l_metric_out = out_dir / f"sub-{subject}_ses-{session}.L.{roi_name}.32k_fs_LR.shape.gii"
    r_metric_out = out_dir / f"sub-{subject}_ses-{session}.R.{roi_name}.32k_fs_LR.shape.gii"
    cifti_path = out_dir / f"sub-{subject}_ses-{session}_space-fsLR_den-91k_desc-{roi_name}.dtseries.nii"

    # T1w vol mask to freesurfer surf
    for h, metric_out in [('l', l_metric_out), ('r', r_metric_out)]:
        H = h.upper()
        out_file = out_dir/f'sub-{subject}_ses-{session}_task-rest_atlas-Coords_hemi-{H}_space-fsnative_desc-{roi_name}.gii'
        hemi=f'{h}h'
        os.environ['SUBJECTS_DIR'] = (Path(subjects_dir)).as_posix()
        cmd = [
            "mri_vol2surf",
            "--regheader",
            f"sub-{subject}",
            "--src",
            t1w_mask,
            "--out",
            out_file,
            "--hemi",
            hemi,
            "--interp",
            "nearest",
            "--surf",
            "midthickness",
            "--cortex"
        ]
        subprocess.run(cmd, check=True)

        #freesurfersurf to fslr
        H = h.upper()
        subj_dir = Path(subjects_dir) / f"sub-{subject}"
        surf_dir = subj_dir / "surf"
        os.environ["SUBJECTS_DIR"] = subjects_dir.as_posix()

        sma_dir = Path(resource_filename("contarg", "data/standard_mesh_atlases"))

        new_sphere = sma_dir / f"fs_LR-deformed_to-fsaverage.{H}.sphere.32k_fs_LR.surf.gii"
        current_area = surf_dir / f"{h}h.midthickness.surf.gii"
        new_area = anat_dir / f"sub-{subject}.{H}.midthickness.32k_fs_LR.surf.gii"
        current_sphere = surf_dir / f"{h}h.sphere.reg.surf.gii"
        metric_in = out_file
        cmd = [
            "wb_command",
            "-metric-resample",
            metric_in,
            current_sphere,
            new_sphere,
            "ADAP_BARY_AREA",
            metric_out,
            "-area-surfs",
            current_area,
            new_area,
        ]
        subprocess.run(cmd, check=True)

        # Threshold the output to get a mask
        cmd = [
            "wb_command",
            "-metric-math",
            "x>0.5",
            metric_out,
            "-var",
            "x",
            metric_out,
        ]
        subprocess.run(cmd, check=True)

        surface_labels, volume_labels, metadata = _prepare_cifti("91k")

    # write cifti
    _create_cifti_image(
        inmask,
        volume_labels,
        [l_metric_out, r_metric_out],
        surface_labels,
        metadata,
        cifti_path,
        **kwargs,
    )
    return cifti_path


def _create_cifti_image(
    mask_file,
    volume_label,
    mask_surfs,
    surface_labels,
    metadata,
    out_file,
    binarize=True,
    dtype=None,
):
    """
    Generate CIFTI image in target space.

    Parameters
    ----------
    mask_file
        mask volume
    volume_label
        Subcortical label file
    mask_surfs
        mask surfaces (L,R)
    surface_labels
        Surface label files used to remove medial wall (L,R)
    metadata
        Metadata to include in CIFTI header
    out_file
        Path ot out output file
    binarize
        If true, round all non-zero surface values to zero

    Returns
    -------
    out :
        BOLD data saved as CIFTI dtseries
    """
    mask_img = nb.load(mask_file)
    label_img = nb.load(volume_label)
    if label_img.shape != mask_img.shape:
        warnings.warn("Resampling bold volume to match label dimensions")
        mask_img = resample_to_img(mask_img, label_img)

    # ensure images match HCP orientation (LAS)
    mask_img = reorient_image(mask_img, target_ornt="LAS")
    label_img = reorient_image(label_img, target_ornt="LAS")

    mask_data = mask_img.get_fdata(dtype="float32")
    timepoints = 1
    label_data = np.asanyarray(label_img.dataobj).astype("int16")

    # Create brain models
    idx_offset = 0
    brainmodels = []
    bm_ts = np.empty((timepoints, 0), dtype="float32")

    for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
        if labels is None:  # surface model
            model_type = "CIFTI_MODEL_TYPE_SURFACE"
            # use the corresponding annotation
            hemi = structure.split("_")[-1]
            # currently only supports L/R cortex
            surf_ts = nb.load(mask_surfs[hemi == "RIGHT"])
            surf_verts = len(surf_ts.darrays[0].data)
            labels = nb.load(surface_labels[hemi == "RIGHT"])
            medial = np.nonzero(labels.darrays[0].data)[0]
            # extract values across volumes
            ts = np.array([tsarr.data[medial] for tsarr in surf_ts.darrays])
            if binarize:
                # round everything up to 1
                ts[ts != 0] = 1

            vert_idx = ci.Cifti2VertexIndices(medial)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vert_idx),
                model_type=model_type,
                brain_structure=structure,
                vertex_indices=vert_idx,
                n_surface_vertices=surf_verts,
            )
            idx_offset += len(vert_idx)
            bm_ts = np.column_stack((bm_ts, ts))
        else:
            model_type = "CIFTI_MODEL_TYPE_VOXELS"
            vox = []
            ts = None
            for label in labels:
                ijk = np.nonzero(label_data == label)
                if ijk[0].size == 0:  # skip label if nothing matches
                    continue
                ts = (
                    mask_data[ijk].reshape((-1, 1))
                    if ts is None
                    else np.concatenate((ts, mask_data[ijk].reshape((-1, 1))))
                )
                vox += [
                    [ijk[0][idx], ijk[1][idx], ijk[2][idx]] for idx in range(len(ts))
                ]

            vox = ci.Cifti2VoxelIndicesIJK(vox)
            bm = ci.Cifti2BrainModel(
                index_offset=idx_offset,
                index_count=len(vox),
                model_type=model_type,
                brain_structure=structure,
                voxel_indices_ijk=vox,
            )
            idx_offset += len(vox)
            bm_ts = np.column_stack((bm_ts, ts.T))
        # add each brain structure to list
        brainmodels.append(bm)

    # add volume information
    brainmodels.append(
        ci.Cifti2Volume(
            mask_img.shape,
            ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, mask_img.affine),
        )
    )

    # generate Matrix information
    series_map = ci.Cifti2MatrixIndicesMap(
        (0,),
        "CIFTI_INDEX_TYPE_SERIES",
        number_of_series_points=timepoints,
        series_exponent=0,
        series_start=0.0,
        series_step=0.0,
        series_unit="SECOND",
    )
    geometry_map = ci.Cifti2MatrixIndicesMap(
        (1,), "CIFTI_INDEX_TYPE_BRAIN_MODELS", maps=brainmodels
    )
    # provide some metadata to CIFTI matrix
    if not metadata:
        metadata = {
            "surface": "fsLR",
            "volume": "MNI152NLin6Asym",
        }
    # generate and save CIFTI image
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    matrix.metadata = ci.Cifti2MetaData(metadata)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=bm_ts, header=hdr)
    if dtype is None:
        img.set_data_dtype(mask_img.get_data_dtype())
    else:
        img.set_data_dtype(dtype)
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")

    ci.save(img, out_file)
    return Path.cwd() / out_file


def tans_inputs_from_fmriprep(fmriprep_dir, out_dir, subject, overwrite=False):
    """
    Create a directory with all of the inputs needed for TANS from the outputs of fmriprep.

    Parameters
    ----------
    fmriprep_dir : str or path
        Path to the top level fmriprep output directory
        (subdirectories from here should be sub-?? and sourcedata, among others).
    out_dir : str or path
        Where you'd like the TANS input directory created
    subject : str
        Subject identifier, with or without sub-

    Returns
    -------

    """
    fmriprep_dir = Path(fmriprep_dir)
    if not fmriprep_dir.exists():
        raise ValueError(f"{fmriprep_dir} does not exist")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if subject[:4] == "sub-":
        subject = subject[4:]

    subjects_dir = fmriprep_dir / "sourcedata/freesurfer"
    if not subjects_dir.exists():
        raise ValueError(f"Couldn't find freesurfer directory at {subjects_dir}")

    subj_out_dir = out_dir / f"sub-{subject}/anat"
    subj_out_dir.mkdir(exist_ok=True, parents=True)
    out_t2 = subj_out_dir / f"sub-{subject}_desc-preproc_T2w.nii.gz"
    if not out_t2.exists():
        try:
            make_fmriprep_t2(fmriprep_dir, subject, subj_out_dir)
        except FileNotFoundError:
            pass
    for h in ["l", "r"]:
        freesurfer_resample_prep(
            subject, subjects_dir, h, subj_out_dir, overwrite=overwrite
        )
        for metric, new_metric_name in [
            ("area.mid", "midthickness_va"),
            ("sulc", None),
            ("curv", None)
        ]:
            metric_out = (
                subj_out_dir
                / f"sub-{subject}.{h.upper()}.{new_metric_name}.32k_fs_LR.shape.gii"
            )
            make_metric_gifti(metric, subject, subjects_dir, h, overwrite=overwrite)
            metric_resample(
                metric,
                subject,
                subjects_dir,
                h,
                subj_out_dir,
                new_metric_name=new_metric_name,
            )
    for metric in ["midthickness_va", "sulc", "curv"]:
        create_dense_scalar(
            metric, subject, subjects_dir, subj_out_dir, overwrite=overwrite
        )

    for surf in ["pial", "white", "inflated"]:
        for h in ["l", "r"]:
            make_surf_gifti(surf, subject, subjects_dir, h, overwrite=overwrite)
            surface_resample(
                surf, subject, subjects_dir, subj_out_dir, h, overwrite=overwrite
            )


def clean_bold(
    bold_path, out_dir, n_dummy, t_r, cfds_to_use=None, confounds=None, aroma=False, overwrite=False
):
    """
    Regresses the specified confounds from an fmriprep processed bold time series.

    Parameters
    ----------
    bold_path : str or path
        Path to fmriprep preproced bold image. Paths to several needed files will be
        assumed to correspond to typical fmriprep outputs based on this path name.
    out_dir : str or path
        Directory where results will be written. File names are based on the bold_path
    n_dummy : int
        Number of dummy volumes to trim from the beginning of the time series
    t_r : float
        TR in seconds
    cfds_to_use : DataFrame, optional
        Dataframe of confounds. If this is passed, confounds will be ignored.
    confounds : list of str
        List of confounds to regress out of the timeseries. Should correspond to column labels in confounds.tsv
    aroma : bool
        If true, include the aroma regressors from the confounds file

    Returns
    -------
    out_path : path
        Path to which the cleaned dataset was written

    """
    bold_path = Path(bold_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    bold_path = bold_path
    mask_path = bold_path.as_posix().replace("_desc-preproc_bold", "_desc-brain_mask")
    out_path = out_dir / bold_path.parts[-1].replace("_desc-preproc_", "_desc-cleaned_")
    json_out_path = Path(out_path.as_posix().replace(".nii.gz", ".json"))
    # if the output already exists, just return the path
    if not overwrite and out_path.exists():
        return out_path
    # pull confounds
    if cfds_to_use is not None:
        confound_names = list(cfds_to_use.columns)
        cfds_to_use = cfds_to_use.loc[n_dummy:, confound_names]
    elif confounds is not None:
        confound_names = confounds.copy()
        bold_parts = parse_bidsname(bold_path.parts[-1])
        for k in ["space", "res", "hemi"]:
            _ = bold_parts.pop(k, None)
        bold_parts["desc"] = "confounds"
        bold_parts["suffix"] = "timeseries"
        bold_parts["extension"] = "tsv"
        confounds_path = bold_path.parent / build_bidsname(bold_parts)
        cfds = pd.read_csv(confounds_path, sep="\t")
        if aroma:
            confound_names += [nn for nn in cfds.columns if "aroma" in nn]
        cfds_to_use = cfds.loc[n_dummy:, confound_names].copy()
    else:
        cfds_to_use = None
        confound_names = ""

    cleaned = nl.image.clean_img(
        nl.image.load_img(bold_path).slicer[:, :, :, n_dummy:],
        confounds=cfds_to_use,
        high_pass=0.01,
        low_pass=0.1,
        mask_img=nl.image.load_img(mask_path),
        t_r=t_r,
    )

    cleaned.to_filename(out_path)

    # save metadata too
    md = json.loads(Path(bold_path.as_posix().replace(".nii.gz", ".json")).read_text())
    md["Confounds"] = confound_names
    md["NSteadyStateRemoved"] = n_dummy
    json_out_path.write_text(json.dumps(md))

    return out_path


def mcf_from_fmriprep_confounds(confounds_path, mcf_path, n_dummy=0):
    """
    Extracts the 6 motion parameters and writes them in the same format as an FSL MCF file.
    parameters
    ----------
    confounds_path : str or path
        Path to fmriprep style tab delimited confounds.tsv file
    mcf_path : str or path
        Path where mcf file will be written
    n_dummy : int
        Number of dummy volumes, corresponds to rows from the start to drop
    """
    cfds = pd.read_csv(confounds_path, sep="\t")
    mcf = cfds.loc[
        n_dummy:, ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    ].copy()
    mcf.to_csv(mcf_path, sep=" ", index=None, header=None)


def get_correlation_array(timeseries, roi):
    """
    Calculate the correlation array of a timeseries with a region of interest (ROI) mask.

    Parameters
    ----------
    timeseries : str or list of str
        Path(s) to the cifti timeseries file(s). If multiple files are passed, they will be condatednated.
    roi : str
        Path to the cifti ROI mask file.

    Returns
    -------
    corr_array : np.ndarray
        Correlation array of the timeseries with the ROI mask.
    """

    # deal with timeseries in case it's more than one run
    if isinstance(timeseries, str) or isinstance(timeseries, Path):
        timeseries = [timeseries]
    ts_datas = []
    for ts in timeseries:
        timeseries_img = ci.load(ts)
        ts_datas.append(timeseries_img.get_fdata(dtype=np.float32))
    timeseries_data = np.vstack(ts_datas)

    # Load the roi data using nibabel
    roi_img = ci.load(roi)

    # Get the data array for the roi
    roi_data = roi_img.get_fdata()

    # Flatten the roi data to get the indices of the non-zero elements
    roi_indices = np.where(roi_data.flatten() != 0)[0]
    roi_tses = timeseries_data[:, roi_indices]

    corr_array = np.nan_to_num(
        np.dot(roi_tses.T, timeseries_data)
        / np.outer(
            np.linalg.norm(roi_tses, axis=0), np.linalg.norm(timeseries_data, axis=0)
        )
    )

    return corr_array


def get_correlation_map(
    timeseries: Union[Union[str, os.PathLike], list],
    stimroi_path: Union[str, os.PathLike],
    refroi_path: Union[str, os.PathLike],
    out_path: Optional[Union[str, os.PathLike]] = None,
    ref_weighting: str = "timeseries",
    invert_reference: bool = False,
):
    """
    Calculate functional connectivity maps for each element within the stimroi with reference to the mean time series
    from the refernce roi. If the reference roi is not binary, the timeseries will be weighted by values in the
    reference roi before averaging.

    Parameters
    ----------
    timeseries : str or list of str
        Path(s) to the cifti timeseries file(s). If multiple files are passed, they will be condatednated.
    stimroi_path : str
        Path to the cifti ROI mask file defining the area to be potentially stimulated.
    refroi_path : str
        Path to the cifti ROI mask file defining the area from which to calculate a mean timeseries.
    out_path : Optional[str], default None
        Path to the output file. If provided, save the result to this file path.
    ref_weighting: str, default "timeseries"
        How to handle values of the refroi if they are not binary. Currently only timeseries weighting before averaging
        is implemented.
    invert_reference: bool, default False
        If true, multiply referenc timeseries by -1 before averaging

    Returns
    -------
    corr_img : Cifti2Image
        The correlation map within the stimroi
    """

    if isinstance(timeseries, str) or isinstance(timeseries, Path):
        timeseries = [timeseries]
    ts_datas = []
    for ts in timeseries:
        timeseries_img = ci.load(ts)
        ts_datas.append(timeseries_img.get_fdata(dtype=np.float32))
    timeseries_data = np.vstack(ts_datas)

    stimroi_img = nb.load(stimroi_path)
    refroi_img = nb.load(refroi_path)

    # Get the data array for the refroi
    refroi_data = refroi_img.get_fdata()

    # Flatten the refroi data to get the indices of the non-zero elements
    refroi_indices = np.where(refroi_data.flatten() != 0)[0]
    refroi_weights = refroi_data[:, refroi_indices].T
    refroi_ts = timeseries_data[:, refroi_indices].T

    if invert_reference:
        refroi_ts = refroi_ts * -1
    if ref_weighting == "timeseries":
        weightedrefroi_ts = refroi_weights * refroi_ts
        ref_ts = weightedrefroi_ts.mean(0)

    else:
        raise NotImplementedError(
            f"{ref_weighting} is not implemented for ref_weighting"
        )

    # Get the data array for the stimroi
    stimroi_data = stimroi_img.get_fdata()

    # Flatten the refroi data to get the indices of the non-zero elements
    stimroi_indices = np.where(stimroi_data.flatten() != 0)[0]
    stimroi_tses = timeseries_data[:, stimroi_indices].T

    corr = np.nan_to_num(
        np.dot(ref_ts, stimroi_tses.T)
        / np.outer(np.linalg.norm(ref_ts), np.linalg.norm(stimroi_tses, axis=1))
    )
    corr_data = stimroi_data.copy()
    corr_data[stimroi_data != 0] = corr.squeeze()
    corr_img = ci.Cifti2Image(corr_data, stimroi_img.header)
    if not out_path is None:
        corr_img.to_filename(out_path.as_posix())

    return corr_img


def get_spatial_correlation(
    timeseries: Union[Union[str, os.PathLike], list],
    roi_path: Union[str, os.PathLike],
    out_path: Optional[Union[str, os.PathLike]] = None,
    comparison_path: Optional[Union[str, os.PathLike]] = None,
) -> ci.Cifti2Image:
    """
    Calculate functional connectivity maps for each element within a cifti ROI, then calculate the correlation
    between the connectivity map at each of those elements and a comparison connectivity map.

    Parameters
    ----------
    timeseries : str or list of str
        Path(s) to the cifti timeseries file(s). If multiple files are passed, they will be condatednated.
    roi_path : str
        Path to the cifti ROI mask file.
    out_path : Optional[str], default None
        Path to the output file. If provided, save the result to this file path.
    comparison_path : Optional[str], default None
        Path to the cifti comparison connecivity map file. If not provided, use the default DepressionCircuit mask.

    Returns
    -------
    spatial_corr_img : Cifti2Image
        The map of spatial correlations with the reference connectivity map.
    """
    if not out_path is None:
        out_path = Path(out_path)
        if not out_path.parent.exists():
            out_path.parent.mkdir(exist_ok=True, parents=True)
    if comparison_path is None:
        comparison_path = Path(
            resource_filename(
                "contarg", "data/rois/DepressionCircuit_space-fsLR_den-91k.dtseries.nii"
            )
        )

    comp_cifti = nb.load(comparison_path)
    # invert the depression circuit map since we're targeting TMS
    comp_cifti_data = comp_cifti.get_fdata(dtype=np.float32) * -1
    comp_cifti_hdr = comp_cifti.header
    axes = [comp_cifti_hdr.get_axis(i) for i in range(comp_cifti.ndim)]

    corr_array = get_correlation_array(timeseries, roi_path)
    spatial_corr = np.nan_to_num(
        np.dot(corr_array, comp_cifti_data.T)
        / np.outer(
            np.linalg.norm(corr_array, axis=1), np.linalg.norm(comp_cifti_data, axis=1)
        )
    )

    roi_cifti = nb.load(roi_path)
    roi_cifti_data = roi_cifti.get_fdata(dtype=np.float32)
    roi_cifti_hdr = roi_cifti.header

    spatial_corr_data = roi_cifti_data.copy()
    spatial_corr_data[roi_cifti_data == 1] = spatial_corr.squeeze()
    spatial_corr_img = ci.Cifti2Image(spatial_corr_data, roi_cifti_hdr)
    if not out_path is None:
        spatial_corr_img.to_filename(out_path.as_posix())

    return spatial_corr_img
