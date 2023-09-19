from collections import namedtuple
from pathlib import Path
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
import nilearn as nl
from nilearn import image
from contarg.utils import add_censor_columns, select_confounds, graph_from_triangles
import templateflow
from smriprep.interfaces.workbench import SurfaceResample
import numpy as np
import nibabel as nb
import pandas as pd

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
    return cleaned_mni_path


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
            new_surface = Path(layout.build_path(new_surf_ents, path_patterns=GII_PATTERN, scope='derivatives', strict=False, validate=False, absolute_paths=False)).parts[-1]
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