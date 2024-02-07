from collections import namedtuple
from pathlib import Path
import subprocess
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
import nilearn as nl
from nilearn import image
from statsmodels.stats import weightstats

from contarg.utils import (
    add_censor_columns,
    select_confounds,
    graph_from_triangles,
    surf_data_from_cifti,
    get_stimroi_path,
    get_refroi_path,
    SurfROI,
    load_timeseries, cross_spearman
)
from contarg.stimgrid import angle_between
from contarg.clustering import cluster_and_plot
from contarg.hierarchical import get_surface_cluster_stats
import templateflow
from contarg.interfaces.smriprep import SurfaceResample
import numpy as np
import nibabel as nb
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
# for plotting
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import seaborn as sns
from sklearn.decomposition import PCA
from mixedvines.mixedvine import MixedVine
from mixedvines.copula import GaussianCopula
from scipy.stats import norm
from scipy import stats


GII_PATTERN = ['sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_hemi-{hemi}][_space-{space}][_den-{density}][_desc-{desc}][_part-{part}]_{suffix<T1w|T2w|T2star|T2starw|FLAIR|FLASH|PD|PDw|PDT2|inplaneT[12]|angio|curv|inflated|midthickness|pial|sulc|thickness|white|mask>}{extension<.nii|.nii.gz|.surf.gii|.shape.gii>}']
TFM_SEG_PATTERN = ['sub-{subject}[/ses-{session}]/{datatype<anat|func>|anat}/sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_space-{space}][_desc-{desc}][_label-{label<GM|WM|CSF>}]_{suffix<probseg>}{extension<.nii.gz>}']
CFDS_PATTERN = ['sub-{subject}[/ses-{session}]/{datatype<anat|func>|anat}/sub-{subject}[_ses-{session}][_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}][_run-{run}][_space-{space}][_res-{res}][_desc-{desc}]_{suffix<timeseries|bold>}{extension<.tsv|.nii.gz>}']

def transform_gmseg(bold, layout, func_dir, overwrite=False):
    """
    Transform gmseg from T1w space to match the space of the passed bold file.

    Parameters
    ----------
    bold : bids.layout.models.BIDSImageFile
        The bold file.
    layout : bids.layout.layout.BIDSLayout
        The layout object.
    func_dir : str or Path
        The directory path for the root of the preprocessed functional files. Should not include subject and session subirectories.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.

    Returns
    -------
    Path
        The path to the transformed gmseg file.
    """
    # transform gmseg
    # get gmseg
    gmseg_t1w = layout.get(
        subject=bold.entities['subject'],
        datatype='anat',
        label='GM',
        suffix='probseg',
        extension='.nii.gz'
    )[0].path

    t1w_to_scanner_ents = bold.entities.copy()
    t1w_to_scanner_ents['desc'] = None
    t1w_to_scanner_ents['from'] = 'T1w'
    t1w_to_scanner_ents['to'] = 'scanner'
    t1w_to_scanner_ents['suffix'] = 'xfm'
    t1w_to_scanner_ents['extension'] = '.txt'
    t1w_to_scanner = layout.get(**t1w_to_scanner_ents)[0].path

    bref_ents = bold.entities.copy()
    bref_ents['suffix'] = 'boldref'
    bref_ents['desc'] = None
    bref = layout.get(**bref_ents)[0].path

    gmseg_bold_ents = bref_ents.copy()
    gmseg_bold_ents['label'] = 'GM'
    gmseg_bold_ents['suffix'] = 'probseg'
    gmseg_bold_ents['space'] = 'boldref'
    gmseg_bold_path = func_dir / layout.build_path(gmseg_bold_ents, path_patterns=TFM_SEG_PATTERN, validate=False,
                                                   absolute_paths=False)

    gmseg_bold_path.parent.mkdir(exist_ok=True, parents=True)
    if not gmseg_bold_path.exists() or overwrite:
        at = ApplyTransforms()
        at.inputs.input_image = gmseg_t1w
        at.inputs.reference_image = bref
        at.inputs.transforms = [t1w_to_scanner]
        at.inputs.interpolation = "LanczosWindowedSinc"
        at.inputs.output_image = gmseg_bold_path.as_posix()
        at.inputs.float = True
        _ = at.run()
    return gmseg_bold_path

def get_mean_gm_signal(bold, gmseg_bold_path, layout):
    """
    Get the mean GM timeseries from a bold timeseries.

    Parameters
    ----------
    bold : bids.layout.models.BIDSImageFile
        The bold file.
    gmseg_bold_path : str or Path
        The path to the gmseg file.
    layout : bids.layout.layout.BIDSLayout
        The layout object.

    Returns
    -------
    numpy.ndarray
        The mean GM signal.
    """
    # get tedana, fmriprep, and goodvoxel masks
    td_mask_ents = bold.entities.copy()
    td_mask_ents['desc'] = 'adaptiveGoodSignal'
    td_mask_ents['suffix'] = 'mask'
    tedana_mask = layout.get(**td_mask_ents)[0].path

    fp_mask_ents = bold.entities.copy()
    fp_mask_ents['desc'] = 'brain'
    fp_mask_ents['suffix'] = 'mask'
    fmriprep_mask = layout.get(**fp_mask_ents)[0].path

    # extract mean gm signal
    gmseg_dat = nl.image.load_img(gmseg_bold_path).get_fdata()
    bold_dat = nl.image.load_img(bold.path).get_fdata()
    td_mask_dat = nl.image.load_img(tedana_mask).get_fdata()
    fp_mask_dat = nl.image.load_img(fmriprep_mask).get_fdata()

    gmseg_dat = gmseg_dat * fp_mask_dat * td_mask_dat
    mean_gm_ts = (bold_dat * gmseg_dat[:, :, :, None]).sum(0).sum(0).sum(0) / gmseg_dat.sum()

    return mean_gm_ts

def cleaned_bold_to_mni(bold, cleaned_bold_path, layout, func_dir, overwrite=False, n_threads=1):
    """
    Transform cleaned bold to MNI152NLin6Asym res-2.

    Parameters
    ----------
    bold : bids.layout.models.BIDSImageFile
        The bold file.
    cleaned_bold_path : str or Path
        The path to the cleaned bold file.
    layout : bids.layout.layout.BIDSLayout
        The layout object.
    func_dir : str or Path
        The directory path for the root of the preprocessed functional files. Should not include subject and session subirectories.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    n_threads : int, optional
        The number of threads to use, by default 1.

    Returns
    -------
    Path
        The path to the cleaned MNI file.
    """
    # transform cleaned bold to MNI152NLin6Asym res-2
    scanner_to_t1w_ents = bold.entities.copy()
    scanner_to_t1w_ents['desc'] = None
    scanner_to_t1w_ents['from'] = 'scanner'
    scanner_to_t1w_ents['to'] = 'T1w'
    scanner_to_t1w_ents['suffix'] = 'xfm'
    scanner_to_t1w_ents['extension'] = '.txt'
    scanner_to_t1w = layout.get(return_type='file', **scanner_to_t1w_ents)[0]

    t1w_to_mni_ents = {}
    t1w_to_mni_ents['subject'] = bold.entities['subject']
    t1w_to_mni_ents['datatype'] = 'anat'
    t1w_to_mni_ents['from'] = 'T1w'
    t1w_to_mni_ents['to'] = 'MNI152NLin6Asym'
    t1w_to_mni_ents['mode'] = 'image'
    t1w_to_mni_ents['suffix'] = 'xfm'
    t1w_to_mni_ents['extension'] = '.h5'
    t1w_to_mni = layout.get(return_type='file', **t1w_to_mni_ents)[0]

    mni_boldref_ents = bold.entities.copy()
    mni_boldref_ents['space'] = 'MNI152NLin6Asym'
    mni_boldref_ents['res'] = '2'
    mni_boldref_ents['suffix'] = 'boldref'
    mni_boldref_ents['desc'] = None
    mni_boldref = layout.get(return_type='file', **mni_boldref_ents)[0]

    cleaned_mni_ents = layout.parse_file_entities(cleaned_bold_path)
    cleaned_mni_ents['space'] = 'MNI152NLin6Asym'
    cleaned_mni_ents['res'] = '2'
    cleaned_mni_path = func_dir / layout.build_path(cleaned_mni_ents, path_patterns=CFDS_PATTERN, validate=False, absolute_paths=False)

    t1w_boldref_ents = bold.entities.copy()
    t1w_boldref_ents['space'] = 'T1w'
    t1w_boldref_ents['suffix'] = 'boldref'
    t1w_boldref_ents['desc'] = None
    t1w_boldref = layout.get(return_type='file', **t1w_boldref_ents)[0]

    cleaned_t1w_ents = layout.parse_file_entities(cleaned_bold_path)
    cleaned_t1w_ents['space'] = 'T1w'
    cleaned_t1w_path =  func_dir / layout.build_path(cleaned_t1w_ents, path_patterns=CFDS_PATTERN, validate=False, absolute_paths=False)

    if not cleaned_t1w_path.exists() or overwrite:
        to_t1w = ApplyTransforms()
        to_t1w.inputs.reference_image = t1w_boldref
        to_t1w.inputs.interpolation = "LanczosWindowedSinc"
        to_t1w.inputs.num_threads = n_threads
        to_t1w.inputs.transforms = [scanner_to_t1w]
        to_t1w.inputs.input_image = cleaned_bold_path
        to_t1w.inputs.output_image = cleaned_t1w_path.as_posix()
        to_t1w.inputs.input_image_type = 3
        to_t1w.run()


    if not cleaned_mni_path.exists() or overwrite:
        to_mni = ApplyTransforms()
        to_mni.inputs.reference_image = mni_boldref
        to_mni.inputs.interpolation = "LanczosWindowedSinc"
        to_mni.inputs.num_threads = n_threads
        to_mni.inputs.transforms = [t1w_to_mni, scanner_to_t1w]
        to_mni.inputs.input_image = cleaned_bold_path
        to_mni.inputs.output_image = cleaned_mni_path.as_posix()
        to_mni.inputs.input_image_type = 3
        to_mni.run()
    return cleaned_t1w_path, cleaned_mni_path


def process_confounds(bold, layout, func_dir, regress_globalsignal=False, regress_motion=True, aroma=False, regress_gm=False,
                      overwrite=False, add_physio=False, n_dummy=4, **kwargs):
    """
        Process confounds, adding mean grey matter signal and censoring with contarg.utils.add_censor_columns

        Parameters
        ----------
        bold : bids.layout.models.BIDSImageFile
            The bold file.
        layout : bids.layout.layout.BIDSLayout
            The layout object.
        func_dir : str or Path
            The directory path for the root of the preprocessed functional files. Should not include subject and session subirectories.
        regress_globalsignal : bool, optional
            Add globalsignal to the used confounds, by default False.
        regress_motion : bool, optional
            Add motion regressors to the used confounds, by default True.
        aroma : bool, optionsl
            Add aroma regressors to the used confounds, by default False.
        regress_gm : bool, optional
            Add mean grey matter signal regressor to the used confounds, by default False.
        overwrite : bool, optional
            Whether to overwrite existing files, by default False.
        add_physio : bool, optional
            Whether to add physio regressors if they exist, by default False.
        n_dummy : int, optional
            The number of dummy volumes, by default 4.
        **kwargs
            Additional keyword arguments, passed to add_censor_columns

        Returns
        -------
        pd.DataFrame
            The selected confounds.

        Raises
        ------
        ValueError
            If the number of volumes in the confounds does not match the number of volumes in the bold file.
        """

    # process confounds
    confounds_ents = bold.entities.copy()
    confounds_ents['desc'] = 'confounds'
    confounds_ents['suffix'] = 'timeseries'
    confounds_ents['extension'] = '.tsv'
    confounds_path = layout.get(**confounds_ents)[0].path

    updated_confounds_path = func_dir / layout.build_path(confounds_ents, path_patterns=CFDS_PATTERN,
                                                          absolute_paths=False, validate=False)

    used_confounds_ents = confounds_ents.copy()
    used_confounds_ents['desc'] = 'usedconfounds'
    used_confounds_path = func_dir / layout.build_path(used_confounds_ents, path_patterns=CFDS_PATTERN,
                                                       absolute_paths=False, validate=False)

    if (not used_confounds_path.exists()) or overwrite:
        gmseg_bold_path = transform_gmseg(bold, layout, func_dir, overwrite=overwrite)
        # get fmriprep bold mask
        fp_mask_ents = bold.entities.copy()
        fp_mask_ents['desc'] = 'brain'
        fp_mask_ents['suffix'] = 'mask'
        fmriprep_mask = layout.get(**fp_mask_ents)[0].path

        # get mean grey matter time series
        mean_gm_ts = get_mean_gm_signal(bold, gmseg_bold_path, layout)

        cfds = add_censor_columns(confounds_path, fmriprep_mask, bold.path, n_dummy=n_dummy, **kwargs, )
        try:
            cfds["gm"] = mean_gm_ts
        except ValueError:
            cfds = cfds[n_dummy:].copy()
            cfds["gm"] = mean_gm_ts
        cfds.to_csv(updated_confounds_path, index=None, sep='\t')
        if regress_motion:
            confound_selectors = ["-motion", "-cosine", "-censor", "-dummy"]
        else:
            confound_selectors = ["-cosine", "-censor", "-dummy"]
        if regress_globalsignal:
            confound_selectors.append('-gs')
        if aroma:
            confound_selectors.append("-aroma")
        if regress_gm:
            confound_selectors.append("-gm")
        cfds_to_use = select_confounds(cfds, confound_selectors)
        cfds_to_use.to_csv(used_confounds_path, index=None, sep='\t')
    else:
        cfds_to_use = pd.read_csv(used_confounds_path, sep='\t')
    return cfds_to_use


def init_bold_to_grayords_wf(bold,
                             layout,
                             cleaned_bold_path,
                             cleaned_mni_path,
                             grayord_density='91k', estimate_goodvoxels=True,
                             omp_nthreads=1, mem_gb=5,
                             ):
    """
    Parameters
    ----------
    bold : bids.layout.models.BIDSImageFile
        The bold file.
    layout : bids.layout.layout.BIDSLayout
        The layout object.
    cleaned_bold_path : str
        Path to the cleaned bold
    cleaned_mni_path : str
        Path to the cleaned bold in mni space
    grayord_density : :class:`str`
        Either ``"91k"`` or ``"170k"``, representing the total *grayordinates*.
    estimate_goodvoxels : :class:`bool`
        Calculate mask excluding voxels with a locally high coefficient of variation to
        exclude from surface resampling
    omp_nthreads : :class:`int`
        Maximum number of threads an individual process may use
    mem_gb : :class:`float`
        Size of BOLD file in GB

    """
    from fmriprep.workflows.bold.resampling import init_bold_fsLR_resampling_wf, init_bold_grayords_wf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from nipype.interfaces.io import DataSink
    from nipype.pipeline import engine as pe

    subject = bold.entities['subject']

    make_gifti = init_bold_fsLR_resampling_wf(grayord_density, estimate_goodvoxels=estimate_goodvoxels, omp_nthreads=omp_nthreads,
                                              mem_gb=mem_gb, name='make_gifti')
    make_cifti = init_bold_grayords_wf(grayord_density, mem_gb=mem_gb, repetition_time=2.5, name='make_cifti')

    try:
        workflow = Workflow(name=f"run-{bold.entities['run']}_funcPreproc")
    except KeyError:
        workflow = Workflow(name=f'funcPreproc')

    make_gifti.get_node('inputnode').inputs.anat_ribbon = layout.get(
        subject=subject,
        desc='ribbon',
        suffix='mask',
        return_type='file'
    )[0]
    make_gifti.get_node('inputnode').inputs.bold_file = cleaned_bold_path.as_posix()
    make_gifti.get_node('inputnode').inputs.morphometrics = layout.get(
        subject=subject,
        datatype='anat',
        extension='shape.gii',
        return_type='file'
    )
    make_gifti.get_node('inputnode').inputs.sphere_reg_fsLR = layout.get(
        subject=subject,
        datatype='anat',
        space='fsLR',
        extension='surf.gii',
        return_type='file'
    )
    make_gifti.get_node('inputnode').inputs.surfaces = layout.get(
        subject=subject,
        datatype='anat',
        space=None,
        extension='surf.gii',
        return_type='file')

    gifti_sinker = pe.Node(DataSink(), name='gifti_sinker')
    gifti_sinker.inputs.base_directory = cleaned_bold_path.parent.as_posix()
    gifti_sinker.inputs.strip_dir = 'bold_fsLR'
    gifti_sinker.inputs.regexp_substitutions = [('_hemi_[LR]', ''),
                                                ('_resampling_wf', '')]
    gifti_sinker.inputs.substitutions = [('tpl', cleaned_bold_path.parts[-1].split('_desc')[0] + '_space'),
                                         ('_sphere.surf_masked',
                                          f"_desc{cleaned_bold_path.parts[-1].split('_desc')[-1].split('_')[0]}_bold")]

    cifti_sinker = pe.Node(DataSink(), name='cifti_sinker')
    cifti_sinker.inputs.base_directory = cleaned_bold_path.parent.as_posix()
    cifti_sinker.inputs.strip_dir = 'cifti_bold'
    cifti_sinker.inputs.substitutions = [('_space-MNI152NLin6Asym_res-2', '_space-fsLR_den-91k')]

    ciftimeta_sinker = pe.Node(DataSink(), name='ciftimeta_sinker')
    ciftimeta_sinker.inputs.base_directory = cleaned_bold_path.parent.as_posix()
    ciftimeta_sinker.inputs.strip_dir = 'cifti_metadata'
    ciftimeta_sinker.inputs.substitutions = [('bold.dtseries',
                                              cleaned_mni_path.parts[-1].replace('_space-MNI152NLin6Asym_res-2',
                                                                                 '_space-fsLR_den-91k').split('.')[0])]

    make_cifti.get_node('inputnode').inputs.bold_std = [cleaned_mni_path.as_posix()]
    make_cifti.get_node('inputnode').inputs.spatial_reference = ['MNI152NLin6Asym_res-2']

    workflow.connect([
        (make_gifti, gifti_sinker, [
            ('outputnode.bold_fsLR', '@bold_fsLR')
        ]),
        (make_gifti, make_cifti, [
            ('outputnode.bold_fsLR', 'inputnode.bold_fsLR')
        ]),
        (make_cifti, cifti_sinker, [
            ('outputnode.cifti_bold', '@cifti_bold')
        ]),
        (make_cifti, ciftimeta_sinker, [
            ('outputnode.cifti_metadata', '@cifti_metadata')
        ]),
    ])

    workflow.base_dir = cleaned_bold_path.parent
    return workflow


AllSurfaces = namedtuple('AllSurfaces', ['l', 'r'])
HemiSurfaces = namedtuple('SurfaceCollection', ['midthickness', 'pial', 'white', 'inflated'])
Surface = namedtuple('Surface', ['path', 'points', 'tris', 'G', 'idxs'])

def load_surfaces(subject, layout, anat_dir, overwrite=False):
    anat_dir = Path(anat_dir)
    anat_out_dir = anat_dir / f'sub-{subject}/anat'
    anat_out_dir.mkdir(exist_ok=True, parents=True)
    # transform surface to fsLR
    tmpsurfaces = {}
    for H in ['L', 'R']:
        ns = {}
        for surface in ['midthickness', 'pial', 'white', 'inflated']:
            try:
                orig_surface = layout.get(
                    subject=subject,
                    datatype='anat',
                    hemi=H,
                    suffix=surface,
                    extension='.surf.gii'
                )[0].path
                current_sphere = layout.get(
                    subject=subject,
                    datatype='anat',
                    space='fsLR',
                    desc='reg',
                    hemi=H,
                    suffix='sphere',
                    extension='.surf.gii'
                )[0].path

                new_sphere = templateflow.api.get(
                    "fsLR",
                    hemi=H,
                    density="32k",
                    suffix="sphere",
                    extension="surf.gii",
                    space=None
                ).as_posix()
                new_surf_ents = layout.parse_file_entities(orig_surface)
                new_surf_ents['space'] = 'fsLR'
                new_surface = Path(layout.build_path(new_surf_ents, path_patterns=GII_PATTERN, scope='derivatives',
                                                     strict=False, validate=False, absolute_paths=False)).parts[-1]
                new_surface = anat_out_dir / new_surface
                if not new_surface.exists() or overwrite:
                    surfresample = SurfaceResample(
                        surface_in = orig_surface,
                        current_sphere=current_sphere,
                        new_sphere=new_sphere,
                        method='BARYCENTRIC',
                        surface_out=new_surface
                    )
                    res = surfresample.run(cwd = anat_out_dir)
                points, triangles = nb.load(new_surface).agg_data()
                G = graph_from_triangles(triangles)
                surf = Surface(new_surface, points, triangles, G, np.arange(len(points)).astype(int))
                ns[surface]=surf
            except IndexError:
                ns[surface]=[]
        tmpsurfaces[H] = ns

    # load surface data structure
    l_hemi = HemiSurfaces(
        tmpsurfaces['L']['midthickness'],
        tmpsurfaces['L']['pial'],
        tmpsurfaces['L']['white'],
        tmpsurfaces['L']['inflated']
    )
    r_hemi = HemiSurfaces(
        tmpsurfaces['R']['midthickness'],
        tmpsurfaces['R']['pial'],
        tmpsurfaces['R']['white'],
        tmpsurfaces['R']['inflated']
    )
    surfaces = AllSurfaces(l_hemi, r_hemi)
    return surfaces


def load_liston_surfs(subject, src_surf_dir):
    tmpsurfaces = {}
    for H in ['L', 'R']:
        ns = {}
        for surface in ['midthickness', 'pial', 'white', 'inflated']:
            surface_path = src_surf_dir / f'sub-{subject}.{H}.{surface}.32k_fs_LR.surf.gii'
            points, triangles = nb.load(surface_path).agg_data()
            G = graph_from_triangles(triangles)
            surf = Surface(surface_path, points, triangles, G, np.arange(len(points)).astype(int))
            ns[surface]=surf
        tmpsurfaces[H] = ns

    # populate surface data structure
    l_hemi = HemiSurfaces(
        tmpsurfaces['L']['midthickness'],
        tmpsurfaces['L']['pial'],
        tmpsurfaces['L']['white'],
        tmpsurfaces['L']['inflated']
    )
    r_hemi = HemiSurfaces(
        tmpsurfaces['R']['midthickness'],
        tmpsurfaces['R']['pial'],
        tmpsurfaces['R']['white'],
        tmpsurfaces['R']['inflated']
    )
    surfaces = AllSurfaces(l_hemi, r_hemi)
    return surfaces


def get_fs_norms(triangles, coords, idxs=None, direction="out"):
    """
    Get vertex normal vectors based on freesurfer conventions as described here:
    https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferWiki/SurfaceNormal

    Parameters
    ----------
    triangles : numpy.ndarray
        The triangle connectivity array, where each row contains three indices representing the vertices of a triangle.
    coords : numpy.ndarray
        The vertex coordinates array, where each row contains the (x, y, z) coordinates of a vertex.
    idxs : numpy.ndarray, optional
        An optional array of vertex indices for which to calculate the normal vectors. If specified, only the normal
        vectors for these vertices will be returned. If not specified, normal vectors for all vertices will be returned.
    direction : {'out', 'in'}, optional
        The direction of the normal vectors to be calculated. 'out' calculates outward-facing normals (default),
        'in' calculates inward-facing normals.

    Returns
    -------
    numpy.ndarray
        An array of vertex normal vectors, where each row contains the (x, y, z) components of a normal vector.

    """
    if direction not in ['out', 'in']:
        return ValueError("direction must be one of 'out' or 'in'")

    if idxs is not None:
        tris = triangles[np.isin(triangles, idxs).sum(1) > 0]
    else:
        tris = triangles
        idxs = range(len(coords))

    # used for finding all the faces that each vertex is a member of
    trilut = {cix: [] for cix in np.arange(0, len(coords))}
    for vv in [0, 1, 2]:
        for tix, cix in enumerate(tris[:, vv]):
            trilut[cix].append(tix)

    tri_normals = np.cross(coords[tris[:, 1]] - coords[tris[:, 0]],
                           coords[tris[:, 2]] - coords[tris[:, 0]])
    vert_norms = np.array([tri_normals[trilut[idx]].mean(0) for idx in idxs])
    vert_norms /= np.linalg.norm(vert_norms, axis=1).reshape(-1, 1)
    if direction == 'out':
        return vert_norms
    else:
        return vert_norms * -1

def calculate_norm_cylinder(coords, normal_vectors, surface_points, max_diameter=1.5):
    r_a = coords
    r_b = normal_vectors

    e = r_b - r_a
    en = e / np.linalg.norm(e, axis=-1)[:, None]
    m = np.cross(r_a, r_b)
    mn = m / np.linalg.norm(e, axis=-1)[:, None]

    d = np.linalg.norm(mn[:, None, :] + np.cross(en[:, None, :], surface_points), axis=-1)

    r_q = surface_points + np.cross(en[:, None, :], (mn[:, None, :] + np.cross(en[:, None, :], surface_points)))

    w_a = np.linalg.norm(np.cross(r_q, r_b[:, None, :]), axis=-1) / np.linalg.norm(m, axis=-1)[:, None]
    w_b = np.linalg.norm(np.cross(r_q, r_a[:, None, :]), axis=-1) / np.linalg.norm(m, axis=-1)[:, None]
    return w_a, w_b, d

def scalpbreaker(points, idxs, scalp_points):
    rep_idx = cdist(points, scalp_points).min(1).argmin()
    return points[rep_idx].squeeze(), idxs[rep_idx].squeeze()

def reduce_surfaces_verts(points, idxs, minimum_distance, twobreaker=None, twobreaker_args=None):
    cluster = AgglomerativeClustering(n_clusters=None,
                                      distance_threshold=minimum_distance,
                                      metric='euclidean',
                                      linkage='complete')
    labels = cluster.fit_predict(points)
    c_points = []
    c_idxs = []
    for ll in np.unique(labels):
        ll_points = points[labels == ll]
        ll_idxs = idxs[labels == ll]
        if len(ll_idxs) == 1:
            c_points.append(ll_points.squeeze())
            c_idxs.append(ll_idxs.squeeze())
        elif len(ll_idxs) == 2:
            if twobreaker is None:
                # if there are two, just take the first one
                c_points.append(ll_points[0].squeeze())
                c_idxs.append(ll_idxs[0].squeeze())
            else:
                # use whatever function is passed to select a point
                tmp_point, tmp_idx = twobreaker(ll_points, ll_idxs, *twobreaker_args)
                c_points.append(tmp_point.squeeze())
                c_idxs.append(tmp_idx.squeeze())
        else:
            rep_idx = cdist([ll_points.mean(0)], ll_points).argmin()
            c_points.append(ll_points[rep_idx].squeeze())
            c_idxs.append(ll_idxs[rep_idx].squeeze())
    c_points = np.array(c_points)
    c_idxs = np.array(c_idxs)
    return c_points, c_idxs


def calc_stimgrid(subject, src_surf_dir, surf_info_dir,
                   headmodel_dir,
                   grid_out_dir, make_plots=True, stimroi="expandedcoleBA46",
                   refroi= "bilateralfullSGCsphere", overwrite=False, fmriprep=False,
                  layout=None, anat_dir=None
                  ):

    if subject[:4] == 'sub-':
        subject = subject[4:]

    src_surf_dir = Path(src_surf_dir)
    surf_info_dir = Path(surf_info_dir)
    headmodel_dir = Path(headmodel_dir)
    grid_out_dir = Path(grid_out_dir)
    grid_out = grid_out_dir / 'SearchGrid.npy'
    if grid_out.exists() and not overwrite:
        return np.load(grid_out)

    grid_out_figs = grid_out_dir / 'figures'

    if fmriprep:
        if layout is None:
            raise ValueError("Must pass a layout if fmriprep is True")
        if anat_dir is None:
            raise ValueError("Must pass an anat_dir if ")
        surfaces = load_surfaces(subject=subject, layout=layout, anat_dir=anat_dir)
    else:
        surfaces = load_liston_surfs(subject, src_surf_dir)

    scalp_path = headmodel_dir / f'm2m_{subject}/Skin.surf.gii'
    scalp_points, scalp_triangles = nb.load(scalp_path).agg_data()
    scalp_G = graph_from_triangles(scalp_triangles)
    scalp = Surface(scalp_path, scalp_points, scalp_triangles, scalp_G, np.arange(len(scalp_points)).astype(int))

    stimroi_mask = get_stimroi_path(stimroi, cifti=True)
    refroi_mask = get_refroi_path(refroi, cifti=True)

    # load ROIS
    stim_roi = SurfROI(surfaces.l.midthickness.path, 'left', roi=stimroi_mask)
    # create a dilated stim roi
    d1stim_roi = SurfROI(surfaces.l.midthickness.path, 'left', roi=stimroi_mask, dilate=1)
    lref_roi = SurfROI(surfaces.l.midthickness.path, 'left', take_largest_cc=True, roi=refroi_mask)
    rref_roi = SurfROI(surfaces.r.midthickness.path, 'right', take_largest_cc=True, roi=refroi_mask)


    # load sulcus data
    if fmriprep:
        sulc_nii = layout.get(
                        subject=subject,
                        datatype='anat',
                        space='fsLR',
                        density='91k',
                        suffix='sulc',
                        extension='.dscalar.nii'
                    )[0].path
    else:
        sulc_nii = surf_info_dir / f'sub-{subject}.sulc.32k_fs_LR.dscalar.nii'
    sulc = nb.load(sulc_nii)
    sulc_dat = sulc.get_fdata()
    l_sulc = surf_data_from_cifti(sulc_dat, sulc.header.get_axis(1), 'CIFTI_STRUCTURE_CORTEX_LEFT').squeeze()

    # calculate norms based on pial surface
    # use dilated stim roi just to avoid edge effects
    stimp_idxs = d1stim_roi.idxs
    stimp_coords = surfaces.l.pial.points[stimp_idxs]
    stimp_outnorms = get_fs_norms(surfaces.l.pial.tris, surfaces.l.pial.points, idxs=stimp_idxs)
    stimp_cplusn = stimp_coords + 10 * stimp_outnorms
    stimp_outm = np.cross(stimp_coords, stimp_outnorms)

    # find gyral verts
    w_a, w_b, d = calculate_norm_cylinder(stimp_coords, stimp_cplusn, surfaces.l.pial.points)
    max_diameter = 1.5
    stim_cyl_idxs, points_cyl_idxs = np.nonzero((w_a >= 0) & (w_a <= 1) & (w_b >= 0) & (w_b <= 1) & (d < max_diameter) & (d > 0))
    dif_points = d1stim_roi.idxs[stim_cyl_idxs] != points_cyl_idxs
    stim_cyl_idxs = stim_cyl_idxs[dif_points]
    points_cyl_idxs = points_cyl_idxs[dif_points]
    maybe_gyrus = np.nonzero(~np.isin(np.arange(len(stimp_coords)), stim_cyl_idxs))[0]
    maybe_gyrus_idxs = d1stim_roi.idxs[maybe_gyrus]
    # drop gyrus points that are below sulcal 0
    maybe_gyrus_idxs = maybe_gyrus_idxs[l_sulc[maybe_gyrus_idxs].squeeze() > 0]
    # drop gyrus points that don't have neighbors
    mg_has_neighbor = [np.isin(list(surfaces.l.pial.G.neighbors(mg)), maybe_gyrus_idxs).any() for mg in maybe_gyrus_idxs]
    maybe_gyrus_idxs = maybe_gyrus_idxs[mg_has_neighbor]
    maybe_gyrus_points = surfaces.l.pial.points[maybe_gyrus_idxs]

    # sulcal wall is the first two rows under the gyrus
    maybe_sw_idxs = np.unique([xx for mg in maybe_gyrus_idxs for xx in surfaces.l.pial.G.neighbors(mg) if xx not in maybe_gyrus_idxs and xx in stim_roi.idxs])
    maybe_sw_idxs = np.unique(
        np.hstack(
            ([xx for mg in maybe_sw_idxs for xx in surfaces.l.pial.G.neighbors(mg) if xx not in maybe_gyrus_idxs and xx in stim_roi.idxs],
             maybe_sw_idxs)
        )
    )
    # using midthichness and not pial on purpose for these coords so that stimulation is optmized for field mag and direction in the cortex
    maybe_sw_points = surfaces.l.midthickness.points[maybe_sw_idxs]
    maybe_sw = np.isin(stimp_idxs, maybe_sw_idxs)


    # use clustering to thin out clumped up vertices
    sw_points, sw_idxs = reduce_surfaces_verts(maybe_sw_points, maybe_sw_idxs, 2, scalpbreaker, twobreaker_args=[scalp_points])

    sw = np.isin(stimp_idxs, sw_idxs)

    # get values relative to scalp for QC
    # get angle between stimp_outnorms norm and norm of closest scalp point
    stimp_scalp_dists = cdist(stimp_coords, scalp.points)
    _, closescalp_idxs = np.nonzero(stimp_scalp_dists == stimp_scalp_dists.min(axis=1)[:, None])
    closescalp_points = scalp_points[closescalp_idxs]
    closescalp_outnorms = get_fs_norms(scalp.tris, scalp.points, idxs=closescalp_idxs)
    closescalp_pon = closescalp_points + closescalp_outnorms

    # calculate outnorm for scalp vertices
    # have to check to see how they're oriented
    dists = np.array([(cdist([sc], [csp]), cdist([sc], [cspon])) for sc, csp, cspon in zip(stimp_coords, closescalp_points, closescalp_pon)])
    dist_comp = (dists[:, 0] < dists[:, 1]).flatten()
    if dist_comp.mean() == 1:
        closescalp_innorms = -closescalp_outnorms
    elif dist_comp.mean() == 0:
        closescalp_innorms = closescalp_outnorms
        closescalp_outnorms = -closescalp_innorms
        closescalp_pon = closescalp_points + closescalp_outnorms
        dists = np.array([(cdist([sc], [csp]), cdist([sc], [cspon])) for sc, csp, cspon in zip(stimp_coords, closescalp_points, closescalp_pon)])
        dist_comp = (dists[:, 0] < dists[:, 1]).flatten()
    else:
        raise ValueError("Scalp triangles are not arranged systematically")

    norm_angles = np.array([angle_between(so, cso) for so, cso in zip(stimp_outnorms, closescalp_outnorms)])

    norm_div = norm_angles
    mg_norm_div = norm_div[maybe_gyrus]
    msw_norm_div = norm_div[maybe_sw]
    sw_norm_div = norm_div[sw]
    #osw_norm_div = norm_div[(l_sulc[stimp_idxs] > -2) & (l_sulc[stimp_idxs] < 2)]

    mg_scalpdist = cdist(surfaces.l.pial.points[maybe_gyrus_idxs], scalp_points).min(1)
    msw_scalpdist = cdist(maybe_sw_points, scalp_points).min(1)
    sw_scalpdist = cdist(sw_points, scalp_points).min(1)
    #osw_scalpdist = cdist(stimp_coords[(l_sulc[stimp_idxs] > -2) & (l_sulc[stimp_idxs] < 2)], scalp_points).min(1)

    sw_innorms = get_fs_norms(surfaces.l.midthickness.tris, surfaces.l.midthickness.points, idxs=sw_idxs, direction='in')

    stimgrid = np.array([sw_points, sw_innorms])
    np.save(grid_out, stimgrid, allow_pickle=False)

    if make_plots:
        fig, ax = plt.subplots(1)
        sns.kdeplot(mg_norm_div, cut=0, ax=ax, label='Maybe Gyrus')
        sns.kdeplot(msw_norm_div, cut=0, ax=ax, label='Maybe Sulcal Wall')
        sns.kdeplot(sw_norm_div, cut=0, ax=ax, label='Sulcal Wall')
        #sns.kdeplot(osw_norm_div, cut=0, ax=ax)
        ax.set_xlabel('Normal angle with nearest scalp normal (degrees)')
        ax.legend()
        fig.savefig(grid_out_figs / 'normal_angles.png')

        fig, ax = plt.subplots(1)
        sns.kdeplot(mg_scalpdist, cut=0, ax=ax, label='Maybe Gyrus')
        sns.kdeplot(msw_scalpdist, cut=0, ax=ax, label='Maybe Sulcal Wall')
        sns.kdeplot(sw_scalpdist, cut=0, ax=ax, label='Sulcal Wall')
        ax.set_xlabel('Distance to scalp (mm)')
        ax.legend()
        fig.savefig(grid_out_figs / 'scalp_dist.png')
        #sns.kdeplot(osw_scalpdist, cut=0, ax=ax)


    if make_plots:
        from mpl_toolkits.mplot3d import axes3d
        lp_tri_stim_mask = (np.isin(surfaces.l.pial.tris, np.nonzero(np.isin(np.arange(len(surfaces.l.pial.points)), stimp_idxs))[0]).sum(1)==3)
        lp_triangulation = tri.Triangulation(surfaces.l.pial.points[:,0], surfaces.l.pial.points[:,1], triangles=surfaces.l.pial.tris, mask=~lp_tri_stim_mask)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(neighbor_coords[:,0], neighbor_coords[:,1], neighbor_coords[:,2], 'o', color='blue')
        ax.plot_trisurf(
            surfaces.l.pial.points[:, 0],
            surfaces.l.pial.points[:, 1],
            surfaces.l.pial.points[:, 2],
            triangles=lp_triangulation.get_masked_triangles(),
            alpha=0.1,
            edgecolor='darkblue'
        )
        # ax.scatter(
        #        stimp_coords[:,0],
        #        stimp_coords[:,1],
        #        stimp_coords[:, 2],
        #        c=norm_angles,
        #        vmin=60,
        #        vmax=120,
        #        cmap='viridis_r',
        #        alpha=1,
        #        s=20,
        #        label='New SW'
        #       )

    #     ax.plot_trisurf(
    #         lw_points[:, 0],
    #         lw_points[:, 1],
    #         lw_points[:, 2],
    #         triangles=l_triangulation.get_masked_triangles(),
    #         alpha=0.2
    #     )

        ax.scatter(maybe_gyrus_points[:,0],
                   maybe_gyrus_points[:,1],
                   maybe_gyrus_points[:,2],
                   color='purple'
                  )
        ax.scatter(sw_points[:,0],
                   sw_points[:,1],
                   sw_points[:,2],
                   color='orange'
                  )

    #     ax.scatter(closescalp_points[:, 0],
    #                closescalp_points[:, 1],
    #                closescalp_points[:, 2],
    #                color='green'
    #               )
        # ax.scatter(points_to_plot[:,0],
        #            points_to_plot[:,1],
        #            points_to_plot[:,2],
        #            color='green'
        #           )
        # ax.scatter(*r_a[0], color='red')
        # ax.plot([r_a[0,0], r_b[0,0]],
        #          [r_a[0,1], r_b[0,1]],
        #          [r_a[0,2], r_b[0,2]],
        #           color='orange')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plotted_points = np.vstack((closescalp_points, stim_roi.coords))
        ax.set_xlim((plotted_points[:,0].min(), plotted_points[:,0].max()))
        ax.set_ylim((plotted_points[:,1].min(), plotted_points[:,1].max()))
        ax.set_zlim((plotted_points[:,2].min(), plotted_points[:,2].max()))
        ax.set_aspect("equal")
        ax.set_box_aspect(ax.get_box_aspect(), zoom=1.3)


        ax.grid(False)

        ax.view_init(elev=0, azim=180)

        plt.show()
        fig.savefig(grid_out_figs / 'cortex_folded.png')

    # make flattend cortical fig
    if make_plots:
        pass
        # TODO figure out confmap dependency mess
        # cm = BFF(surfaces.l.pial.points, surfaces.l.pial.tris)
        # cm.layout(normalize=False)
        # lp_triangulation = tri.Triangulation(cm.image.vertices[:,0], cm.image.vertices[:,1], triangles=cm.image.faces, mask=~lp_tri_stim_mask)
        #
        # cm_maybe_gyrus_points = cm.image.vertices[maybe_gyrus_idxs]
        # cm_maybe_sw_points = cm.image.vertices[maybe_sw_idxs]
        # cm_sulc_near_0 = cm.image.vertices[stim_roi.idxs][(l_sulc[stim_roi.idxs] > -2) & (l_sulc[stim_roi.idxs] < 2)]
        #
        # pca = PCA(2, random_state=0)
        # _ = pca.fit(cm.image.vertices[np.unique(lp_triangulation.get_masked_triangles())][:, :2])
        # flat_verts = pca.transform(cm.image.vertices[:, :2])
        #
        # cm_roi_points = cm.image.vertices[np.unique(lp_triangulation.get_masked_triangles())]
        # oref_points = np.array([
        #     [cm_roi_points[:,0].min(), (cm_roi_points[:,1].max() + cm_roi_points[:,1].min())/2],
        #     [cm_roi_points[:,0].max(), (cm_roi_points[:,1].max() + cm_roi_points[:,1].min())/2],
        #     [ (cm_roi_points[:,0].max() + cm_roi_points[:,0].min())/2, cm_roi_points[:,1].min()],
        #     [ (cm_roi_points[:,0].max() + cm_roi_points[:,0].min())/2, cm_roi_points[:,1].max()],
        # ])
        # tref_points = pca.transform(oref_points)
        # flat_maybe_gyrus_points = pca.transform(cm.image.vertices[maybe_gyrus_idxs, :2])
        # flat_sulc_near_0 = pca.transform(cm.image.vertices[stim_roi.idxs][(l_sulc[stim_roi.idxs] > -2) & (l_sulc[stim_roi.idxs] < 2), :2])
        # flat_maybe_sw_points = pca.transform(cm.image.vertices[sw_idxs, :2])
        # # print(tref_points)
        # # print(angle_between(tref_points[5], oref_points[5,1:]))
        # # print(180 - angle_between(tref_points[5], oref_points[5,1:]))
        # # ztheta = np.pi - angle_between(tref_points[5], oref_points[5,1:]) * (np.pi / 180)
        # # rot_mat = np.array([
        # #     [np.cos(ztheta),  -np.sin(ztheta)],
        # #     [np.sin(ztheta),  np.cos(ztheta)],
        # # ])
        # # rot_mat
        # # tref_points = np.matmul(rot_mat, tref_points.T).T
        # # flat_verts = np.matmul(rot_mat, flat_verts.T).T
        # # # print(tref_points)
        # # flip_mat =  np.array([[1,0], [0,1]])
        # # if tref_points[5,1] < 0:
        # #     flip_mat[1,1] = -1
        # # tref_points = np.matmul(flip_mat, tref_points.T).T
        # # flat_verts =  np.matmul(flip_mat, flat_verts.T).T
        # flip_mat =  np.array([[1,0], [0,1]])
        # if tref_points[1,0] < tref_points[0,0]:
        #     flip_mat[0,0] = -1
        # tref_points = np.matmul(flip_mat, tref_points.T).T
        # flat_verts =  np.matmul(flip_mat, flat_verts.T).T
        # flat_maybe_gyrus_points =  np.matmul(flip_mat, flat_maybe_gyrus_points.T).T
        # flat_sulc_near_0 =  np.matmul(flip_mat, flat_sulc_near_0.T).T
        # flat_maybe_sw_points =  np.matmul(flip_mat, flat_maybe_sw_points.T).T
        #
        # flip_mat =  np.array([[1,0], [0,1]])
        # if tref_points[3,1] < tref_points[2,1]:
        #     flip_mat[1,1] = -1
        # tref_points = np.matmul(flip_mat, tref_points.T).T
        # flat_verts =  np.matmul(flip_mat, flat_verts.T).T
        # flat_maybe_gyrus_points =  np.matmul(flip_mat, flat_maybe_gyrus_points.T).T
        # flat_sulc_near_0 =  np.matmul(flip_mat, flat_sulc_near_0.T).T
        # flat_maybe_sw_points =  np.matmul(flip_mat, flat_maybe_sw_points.T).T
        #
        # norm = Normalize(vmin=-10, vmax=10)
        # colors = mpl.colormaps.get_cmap('RdBu_r')(norm(-l_sulc[lp_triangulation.get_masked_triangles()].mean(1)))
        # mappable = ScalarMappable(cmap='RdBu_r', norm=norm)
        #
        # fig, ax = plt.subplots(1)
        # triplot = ax.tripcolor(
        #     flat_verts[:,0],
        #     flat_verts[:,1],
        #     -l_sulc[lp_triangulation.get_masked_triangles()].mean(1),
        #     triangles=lp_triangulation.get_masked_triangles(),
        #     cmap='RdBu_r',
        #     linewidth=1,
        #     vmin=-10,
        #     vmax=10,
        #     alpha=0.8,
        # )
        # flat_roi_points = flat_verts[np.unique(lp_triangulation.get_masked_triangles())]
        #
        # print(tref_points)
        #
        # ax.axis('off')
        # ax.set_xlim((flat_roi_points[:, 0].min(), flat_roi_points[:, 0].max()))
        # ax.set_ylim((flat_roi_points[:, 1].min(), flat_roi_points[:, 1].max()))
        #
        #
        # ax.scatter(flat_maybe_gyrus_points[:,0],
        #            flat_maybe_gyrus_points[:,1],
        #            color='red',
        #            alpha=1,
        #            s=20,
        #            label='Gyrus'
        #           )
        # ax.scatter(flat_sulc_near_0[:,0],
        #            flat_sulc_near_0[:,1],
        #            color='black',
        #            alpha=1,
        #            s=30,
        #            label='Old SW'
        #           )
        # ax.scatter(flat_maybe_sw_points[:,0],
        #            flat_maybe_sw_points[:,1],
        #            color='mediumpurple',
        #            alpha=1,
        #            s=20,
        #            label='New SW'
        #           )
        #
        # ax.plot(tref_points[:2,0], tref_points[:2, 1], color='black', label='Orig x dir', linestyle="dotted", alpha=0.7)
        # ax.scatter(*tref_points[0], color='black', alpha=0.5)
        # ax.plot(tref_points[2:,0], tref_points[2:, 1], color='black', label='Orig y dir', linestyle="dashed", alpha=0.7)
        # ax.scatter(*tref_points[2], color='black', alpha=0.5)
        #
        # ax.set_xlim((flat_roi_points[:, 0].min(), flat_roi_points[:, 0].max()))
        # ax.set_ylim((flat_roi_points[:, 1].min(), flat_roi_points[:, 1].max()))
        #
        #
        # ax.legend()
        # fig.savefig(grid_out_figs / 'flat_cortex.png')
    return stimgrid


def make_uncert_sims(dim, nsims, dist_std, angle_std):
    # if we want to randomly distribute points for simulation, use this code
    # model distribution as 2 dimension for deviation on surface of scalp
    # and 3 dimensions for deviation of angle
    if nsims % 2 == 0:
        nsims = nsims // 2
    else:
        nsims = (nsims + 1) // 2
    dim = 5
    simvine = MixedVine(dim)
    simvine.set_marginal(0, norm(0, dist_std))
    simvine.set_marginal(1, norm(0, dist_std))
    simvine.set_marginal(2, norm(0, angle_std))
    simvine.set_marginal(3, norm(0, angle_std))
    simvine.set_marginal(4, norm(0, angle_std))
    simvine.set_copula(1, 0, GaussianCopula(0))
    simvine.set_copula(1, 1, GaussianCopula(0))
    simvine.set_copula(1, 2, GaussianCopula(0))
    simvine.set_copula(1, 3, GaussianCopula(0))

    simvine.set_copula(2, 0, GaussianCopula(0))
    simvine.set_copula(2, 1, GaussianCopula(0))
    simvine.set_copula(2, 2, GaussianCopula(0))

    simvine.set_copula(3, 0, GaussianCopula(0))
    simvine.set_copula(3, 1, GaussianCopula(0))

    simvine.set_copula(4, 0, GaussianCopula(0))
    uncert_sims = simvine.rvs(nsims)
    uncert_sims = np.vstack([[0,0,0,0,0], uncert_sims, -1 * uncert_sims])
    return uncert_sims

def get_prob_vine(dim, dist_std, angle_std):
    # create vine for calculating probability
    # model distribution as 1 dimension for deviation on surface of scalp
    # and 3 dimensions for deviation of angle
    dim = 4
    vine = MixedVine(dim)
    vine.set_marginal(0, norm(0, dist_std))
    vine.set_marginal(1, norm(0, angle_std))
    vine.set_marginal(2, norm(0, angle_std))
    vine.set_marginal(3, norm(0, angle_std))
    vine.set_copula(1, 0, GaussianCopula(0))
    vine.set_copula(2, 0, GaussianCopula(0))
    vine.set_copula(3, 0, GaussianCopula(0))
    vine.set_copula(1, 1, GaussianCopula(0))
    vine.set_copula(2, 1, GaussianCopula(0))
    vine.set_copula(1, 2, GaussianCopula(0))
    return vine


def make_rot_mat(yaw, pitch, roll):
    a = yaw * np.pi / 180
    b = pitch * np.pi / 180
    c = roll * np.pi / 180

    rot_mat = np.zeros((3, 3))

    rot_mat[0, 0] = np.cos(b) * np.cos(c)
    rot_mat[0, 1] = (np.sin(a) * np.sin(b) * np.cos(c)) - (np.cos(a) * np.sin(c))
    rot_mat[0, 2] = (np.cos(a) * np.sin(b) * np.cos(c)) + (np.sin(a) * np.sin(c))

    rot_mat[1, 0] = np.cos(b) * np.sin(c)
    rot_mat[1, 1] = (np.sin(a) * np.sin(b) * np.sin(c)) + (np.cos(a) * np.cos(c))
    rot_mat[1, 2] = (np.cos(a) * np.sin(b) * np.sin(c)) - (np.sin(a) * np.cos(c))

    rot_mat[2, 0] = - np.sin(b)
    rot_mat[2, 1] = np.sin(a) * np.cos(b)
    rot_mat[2, 2] = np.cos(a) * np.cos(b)

    return rot_mat


def test_make_rot_mat():
    exp_90yaw = np.array(
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]]
    )
    exp_90pitch = np.array(
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]]
    )
    exp_90roll = np.array(
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]]
    )
    assert np.allclose(make_rot_mat(90, 0, 0), exp_90yaw)
    assert np.allclose(make_rot_mat(0, 90, 0), exp_90pitch)
    assert np.allclose(make_rot_mat(0, 0, 90), exp_90roll)

    assert np.allclose(make_rot_mat(90, 90, 0), np.matmul(exp_90pitch, exp_90yaw))
    assert np.allclose(make_rot_mat(90, 0, 90), np.matmul(exp_90roll, exp_90yaw))
    assert np.allclose(make_rot_mat(0, 90, 90), np.matmul(exp_90roll, exp_90pitch))
    assert np.allclose(make_rot_mat(90, 90, 90), np.matmul(exp_90roll, np.matmul(exp_90pitch, exp_90yaw)))


def setup_uncert_sims(headmodel_dir, sim_dir, dist_std, angle_std, distancetoscalp=2, outname=None, uncert_deviations_path=None, nsims=1000, overwrite=False):
    Sim_dir = Path(sim_dir)

    if outname is None:
        outname = 'uncertainty'
    uncert_dir = Sim_dir / outname
    uncert_dir.mkdir(exist_ok=True)
    settings_path = uncert_dir / 'settings.pkl.gz'

    if settings_path.exists() and not overwrite:
        return settings_path

    HeadModel_dir = Path(headmodel_dir)
    try:
        m2m_dir = sorted(HeadModel_dir.glob('m2m*'))[0]
    except IndexError:
        raise FileNotFoundError(f"No m2m directory found in {HeadModel_dir}")

    subject = '_'.join(m2m_dir.parts[-1].split('_')[1:])
    scalp_path = m2m_dir / 'Skin.surf.gii'
    scalp_points, scalp_triangles = nb.load(scalp_path).agg_data()
    scalp_G = graph_from_triangles(scalp_triangles)
    scalp = Surface(scalp_path, scalp_points, scalp_triangles, scalp_G, np.arange(len(scalp_points)).astype(int))

    outputs = pd.read_pickle(sim_dir / f'sub-{subject}_simulations.pkl.gz')

    if uncert_deviations_path is None:
        uncert_sims = make_uncert_sims(5, nsims, dist_std, angle_std)
    else:
        uncert_sims = np.load(uncert_deviations_path)
    vine = get_prob_vine(4, dist_std, angle_std)

    settings = []
    print(f"Calculating settings for {len(outputs)} simulations.")
    for oix, orow in outputs.iterrows():

        if oix % 25 == 0:
            print(oix, end=', ')
        bmat = np.vstack([orow.bxv, orow.byv, orow.bzv]).T
        # s = sim_struct.SESSION()
        # tms_list = s.add_tmslist()
        # tms_list.fnamecoil = coil_path.as_posix()
        # tms_list.solver_options='paradiso'

        shifted_points = []
        sncenters = []
        for usim in uncert_sims:
            if (usim[:2] == 0).all():
                shifted_point = np.array([np.nan, np.nan, np.nan])
                sncenter = np.array([orow.bx, orow.by, orow.bz])
            else:
                shifted_point = np.array([orow.bx, orow.by + usim[0], orow.bz + usim[1]])
                sncenter = np.array([0, 0, 0])
            shifted_points.append(shifted_point)
            sncenters.append(sncenter)

        # compromise between a single huge set of vectorized operations and doing it seperately for each point
        shifted_points = np.array(shifted_points)
        sncenters = np.array(sncenters)

        snidxs = np.nan * np.ones(len(sncenters))
        shift_dists = cdist(shifted_points, scalp.points)
        tmpsnidxs = (shift_dists == shift_dists.min(1).reshape(-1, 1)).nonzero()[1]
        snoutnorms = get_fs_norms(scalp.tris, scalp.points, idxs=tmpsnidxs)
        sncenters[pd.notnull(shifted_points[:, 0])] = scalp.points[tmpsnidxs] + (distancetoscalp * snoutnorms)
        snidxs[pd.notnull(shifted_points[:, 0])] = tmpsnidxs
        sndists = cdist(np.array([[orow.bx, orow.by, orow.bz]]), sncenters)[0]
        probs = vine.pdf(np.hstack([np.array([sndists]).T, uncert_sims[:, 2:]]))
        for sncenter, usim, sndist, snidx, prob in zip(sncenters, uncert_sims, sndists, snidxs, probs):
            new_mat = np.matmul(make_rot_mat(usim[2], usim[3], usim[4]), bmat)
            # pos = tms_list.add_position()
            matsimnibs = np.zeros((4, 4))
            matsimnibs[:3, :3] = new_mat
            matsimnibs[:3, 3] = sncenter
            matsimnibs[3, 3] = 1
            # pos.matsimnibs = matsimnibs
            settings.append(dict(
                oix=oix,
                center=sncenter,
                cix=snidx,
                mat=new_mat,
                yaw=usim[2],
                pitch=usim[3],
                roll=usim[4],
                matsimnibs=matsimnibs,
                origx=orow.bx,
                origy=orow.by,
                origz=orow.bz,
                sdist=sndist,
                prob=prob,
            ))
        # sessions.append(s)
    settings = pd.DataFrame(settings)
    settings['normed_prob'] = settings.groupby('oix').prob.transform(lambda x: x / x.sum())
    settings.to_pickle(settings_path)
    return settings_path


def run_clusters(subject, concat_nii, clust_outdir, src_surf_dir,
                 surf_source='liston',
                 maxrepdist=0.2,
                 medial_wall=None,
                 refroi='bilateralfullSGCsphere',
                 stimroi='expandedcoleBA46',
                 out_prefix='',
                 overwrite=False):
    """
    subject : str
        subject id
    concat_nii : path or list of paths
        A path (or list of paths) to a cifti of the functional scans that should be clustering.
    clust_outdir : Pathlib path
        Path to the output directory
    src_surf_dir : Pathlib path
        Path to directory where surfaces can be found, example: '/data/EDB/TMSpilot/liston/sub-24563/anat/T1w/fsaverage_LR32k'
    surf_source : str
        placeholder
    maxrepdist : float
        Maximum representative distance between cluster members
    medial_wall : dict or None
        dict of paths to medial wall mask cifties with l and r keys. If None, templateflow's 32k fslr masks are used.
    refroi : str
        name of the reference roi
    stimroi : str
        name of the roi where stimulation will be delivered
    out_prefix : str
        string preprended to output files
    overwrite : bool
        Should files be overwritten
    """

    if subject[:4] == 'sub-':
        subject = subject[4:]

    clust_outdir = Path(clust_outdir)
    clust_outdir.mkdir(exist_ok=True, parents=True)

    ref_outfile = clust_outdir / f'{out_prefix}refclusters.pkl.gz'
    ref_ts_file = clust_outdir / f'{out_prefix}refts.npz'
    ref_repts_file = clust_outdir / f'{out_prefix}refrepts.npz'
    dstim_outfile = clust_outdir / f'{out_prefix}dstimclusters.pkl.gz'
    dstim_verts_file = clust_outdir / f'{out_prefix}dstimverts.pkl.gz'

    outputs_present = (
            ref_outfile.exists()
            & ref_ts_file.exists()
            & ref_repts_file.exists()
            & dstim_outfile.exists()
            & dstim_verts_file.exists()
    )
    if not outputs_present or overwrite:

        # load surfaces
        if surf_source == 'liston':
            surfaces = load_liston_surfs(subject, src_surf_dir)
        else:
            raise NotImplementedError

        if medial_wall is None:
            medial_wall = {}
            medial_wall['l'] = templateflow.api.get(template='fsLR', density='32k', desc='nomedialwall', hemi='L')
            medial_wall['r'] = templateflow.api.get(template='fsLR', density='32k', desc='nomedialwall', hemi='R')

        stimroi_mask = get_stimroi_path(stimroi, cifti=True)
        refroi_mask = get_refroi_path(refroi, cifti=True)
        try:
            if len(concat_nii) == 1:
                timeseries = [concat_nii]
            else:
                timeseries = concat_nii
        except TypeError:
            timeseries = [concat_nii]

        lts_data, rts_data, ts_data = load_timeseries(timeseries)

        # in addition to dropping medial wall vertices, drop, vertices that are all 0
        l_medial_wall_mask = (~nb.load(medial_wall['l']).agg_data().astype(bool) | (np.all((lts_data == 0), axis=1)))
        r_medial_wall_mask = (~nb.load(medial_wall['r']).agg_data().astype(bool) | (np.all((rts_data == 0), axis=1)))

        stim_roi = SurfROI(surfaces.l.midthickness.path, 'left', timeseries, roi=stimroi_mask,
                           exclude_mask=l_medial_wall_mask)
        rref_roi = SurfROI(surfaces.r.midthickness.path, 'right', timeseries, take_largest_cc=True, roi=refroi_mask,
                           exclude_mask=r_medial_wall_mask)
        lref_roi = SurfROI(surfaces.l.midthickness.path, 'left', timeseries, take_largest_cc=True, roi=refroi_mask,
                           exclude_mask=l_medial_wall_mask)

        dstim_roi = SurfROI(surfaces.l.midthickness.path, 'left', timeseries, dilate=60, roi=stimroi_mask,
                            exclude_mask=l_medial_wall_mask | lref_roi._roi_dat.astype(bool))

        lref_labels, _, _ = cluster_and_plot(lref_roi.ts,
                                             maxrepdist,
                                             lref_roi.idxs,
                                             lref_roi.coords,
                                             connectivity=lref_roi.connectivity,
                                             min_verts=0,
                                             spearman=True,
                                             plot=True,
                                             plot_path=clust_outdir / f'figures/{out_prefix}lref_clusters.png'
                                             )
        lref_clusters = get_surface_cluster_stats(lref_labels,
                                                  lref_roi.ts,
                                                  lref_roi.idxs,
                                                  lref_roi.coords
                                                  )

        rref_labels, _, _ = cluster_and_plot(rref_roi.ts,
                                             maxrepdist,
                                             rref_roi.idxs,
                                             rref_roi.coords,
                                             connectivity=rref_roi.connectivity,
                                             min_verts=0,
                                             spearman=True,
                                             plot=True,
                                             plot_path=clust_outdir / f'figures/{out_prefix}rref_clusters.png'
                                             )
        rref_clusters = get_surface_cluster_stats(rref_labels,
                                                  rref_roi.ts,
                                                  rref_roi.idxs,
                                                  rref_roi.coords
                                                  )

        dstim_labels, _, _ = cluster_and_plot(dstim_roi.ts,
                                              maxrepdist,
                                              dstim_roi.idxs,
                                              dstim_roi.coords,
                                              connectivity=dstim_roi.connectivity,
                                              min_verts=0,
                                              spearman=True,
                                              plot=True,
                                              plot_path=clust_outdir / f'figures/{out_prefix}dstim_clusters.png'
                                              )
        dstim_clusters = get_surface_cluster_stats(dstim_labels,
                                                   dstim_roi.ts,
                                                   dstim_roi.idxs,
                                                   dstim_roi.coords
                                                   )

        rref_clusters['hemi'] = 'right'
        lref_clusters['hemi'] = 'left'
        ref_clusters = pd.concat([rref_clusters, lref_clusters])
        ref_repts = np.array(list(ref_clusters.repts.apply(lambda x: list(x))))
        ref_ts = np.vstack([rref_roi.ts, lref_roi.ts])
        ref_clusters.to_pickle(ref_outfile)
        np.save(ref_ts_file, ref_ts)
        np.save(ref_repts_file, ref_repts)

        dstim_refrep_corr, _, _ = cross_spearman(np.array(list(dstim_clusters.repts.values)),
                                                 np.array(list(ref_clusters.repts)))
        dstim_refrep_ds = weightstats.DescrStatsW(np.arctanh(dstim_refrep_corr).T, weights=ref_clusters.nvert.values)
        dstim_clusters['clust_corr'] = np.tanh(dstim_refrep_ds.mean)
        dstim_clusters['clust_corr_avet'], dstim_clusters['clust_corr_avep'] = dstim_refrep_ds.ztest_mean()

        n = ref_repts.shape[1]
        rs = dstim_clusters.clust_corr
        ts = rs * np.sqrt((n - 2) / ((rs + 1.0) * (1.0 - rs)))
        dstim_clusters['clust_corr_rp'] = stats.t.sf(np.abs(ts), n - 2) * 2
        dstim_clusters.to_pickle(dstim_outfile)

        dstim_verts = pd.DataFrame(index=dstim_roi.idxs)
        dstim_verts['idx'] = dstim_roi.idxs
        dstim_verts['id'] = dstim_labels
        dstim_verts = dstim_verts.merge(
            dstim_clusters.loc[:, ['id', 'nvert', 'clust_corr', 'clust_corr_avet', 'clust_corr_avep', 'clust_corr_rp']])
        dstim_verts.to_pickle(dstim_verts_file)
    else:
        ref_clusters = pd.read_pickle(ref_outfile)
        ref_ts = np.load(ref_ts_file)
        ref_repts = np.load(ref_repts_file)
        dstim_clusters = pd.read_pickle(dstim_outfile)
        dstim_verts = pd.read_pickle(dstim_verts_file)

    return ref_clusters, ref_ts, ref_repts, dstim_clusters, dstim_verts


def make_uncert_surfaces(subject, src_surf_dir, uncert_dir, overwrite=False, fmriprep=False,
                  layout=None, anat_dir=None):
    if fmriprep:
        if layout is None:
            raise ValueError("Must pass a layout if fmriprep is True")
        if anat_dir is None:
            raise ValueError("Must pass an anat_dir if ")
        surfaces = load_surfaces(subject=subject, layout=layout, anat_dir=anat_dir, overwrite=overwrite)
    else:
        surfaces = load_liston_surfs(subject, src_surf_dir)

    medial_wall = {}
    medial_wall['l'] = templateflow.api.get(template='fsLR', density='32k', desc='nomedialwall', hemi='L')
    medial_wall['r'] = templateflow.api.get(template='fsLR', density='32k', desc='nomedialwall', hemi='R')

    catted_means = uncert_dir / f"sub-{subject}_desc-magnEmean_stat.nii.gz"
    if not catted_means.exists() or overwrite:
        cmd = [
            "3dTcat",
            "-overwrite",
            "-prefix",
            catted_means,
            f"{uncert_dir.as_posix()}/*mean_magnE.nii.gz"
        ]
        subprocess.run(cmd, check=True)

    catted_stds = uncert_dir / f"sub-{subject}_desc-magnEstd_stat.nii.gz"
    if not catted_stds.exists() or overwrite:
        cmd = [
            "3dTcat",
            "-overwrite",
            "-prefix",
            catted_stds,
            f"{uncert_dir.as_posix()}/*std_magnE.nii.gz"
        ]
        subprocess.run(cmd, check=True)

    catted_counts = uncert_dir / f"sub-{subject}_desc-abovethreshactprobs_stat.nii.gz"
    if not catted_counts.exists() or overwrite:
        cmd = [
            "3dTcat",
            "-overwrite",
            "-prefix",
            catted_counts,
            f"{uncert_dir.as_posix()}/*abovethreshactprobs_magnE.nii.gz"
        ]
        subprocess.run(cmd, check=True)

    cifti_outs = []
    for metric_path in [catted_means, catted_stds, catted_counts]:
        cifti_out = metric_path.as_posix().replace(".nii.gz", ".dtseries.nii").replace("_desc-",
                                                                                       "_space-fsLR_den-32k_desc-")
        if not Path(cifti_out).exists() or overwrite:
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
    return cifti_outs