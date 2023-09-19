from pathlib import Path
import templateflow
from matplotlib import pyplot as plt  # Matlab-ish plotting commands
from nilearn import plotting as nlp   # Nice neuroimage plotting
from nilearn import image
import nibabel as nb
from nibabel import cifti2 as ci
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from pkg_resources import resource_filename
from contarg.tans import get_tans_inputs
from contarg.utils import get_stimroi_path, STIMROIS, REFROIS, parse_bidsname, get_refroi_path, surf_data_from_cifti
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def plot_surf(cifitimg, sulcal_depth=None, cifitimgb=None, cmap=None, vmin=None, vmax=None, colorbar=False,
              hemispheres=None, views=None, surf='midthickness', labela=None, labelb=None, rsurf=None, lsurf=None, **kwargs):
    if sulcal_depth is None:
        print("No sulcal depth provided, using data from pilot 1")
        sulcal_depth = ci.load(
            '/data/EDB/TMSpilot/derivatives/contarg/tans_test1/sub-24573/ses-1/anat/sub-24573.sulc.32k_fs_LR.dscalar.nii')
    if cmap is None:
        cmap = "Greens"
    if hemispheres is None:
        hemispheres = ['left']
    if views is None:
        views = ['lateral']
    if isinstance(hemispheres, str):
        hemispheres = [hemispheres]
    if isinstance(views, str):
        views = [views]
    if (lsurf is None) or (rsurf is None):
        if surf not in ['midthickness', 'inflated', 'veryinflated']:
            raise ValueError("surf must be one of ['midthickness', 'inflated', 'veryinflated']")
        lsurf = templateflow.api.get(
            "fsLR", hemi='L', density="32k", suffix=surf, extension="surf.gii"
        )
        rsurf = templateflow.api.get(
            "fsLR", hemi='R', density="32k", suffix=surf, extension="surf.gii"
        )

    fig, axes = plt.subplots(len(views), len(hemispheres), subplot_kw={'projection': '3d'}, dpi=250,
                             figsize=(len(hemispheres) * 4, len(views) * 4))
    n_plots = len(views) * len(hemispheres)
    kp = 0
    try:
        if len(axes.shape) == 1:
            if len(hemispheres) == 2:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.reshape(-1, 1)
    except AttributeError:
        axes = np.array([[axes]])

    # get data to plot
    if cifitimgb is None:
        ldata = surf_data_from_cifti(cifitimg.get_fdata(dtype=np.float32), cifitimg.header.get_axis(1),
                                     'CIFTI_STRUCTURE_CORTEX_LEFT').mean(axis=1)
        rdata = surf_data_from_cifti(cifitimg.get_fdata(dtype=np.float32), cifitimg.header.get_axis(1),
                                     'CIFTI_STRUCTURE_CORTEX_RIGHT').mean(axis=1)

        legend_elements = None
    else:
        adat = cifitimg.get_fdata(dtype=np.float32)
        bdat = cifitimgb.get_fdata(dtype=np.float32)
        res = np.zeros_like(adat)
        res[adat != 0] = 1
        res[bdat != 0] = 2
        res[(adat != 0) & (bdat != 0)] = 3
        ldata = surf_data_from_cifti(res, cifitimg.header.get_axis(1), 'CIFTI_STRUCTURE_CORTEX_LEFT').mean(axis=1)
        rdata = surf_data_from_cifti(res, cifitimgb.header.get_axis(1), 'CIFTI_STRUCTURE_CORTEX_RIGHT').mean(axis=1)

        # make nice colormap and legend
        c1 = "#de2d26"
        c2 = "#2171b5"
        c3 = "#3f007d"

        if labela is None:
            labela = "Map 1"
        if labelb is None:
            labelb = "Map 2"
        legend_elements = [Patch(facecolor=c1, label=labela),
                           Patch(facecolor=c2, label=labelb),
                           Patch(facecolor=c3, label="Overlap")
                           ]
        cmap = ListedColormap(["white", c1, c2, c3])
        vmin = 0
        vmax = 3
    for axrow, view in zip(axes, views):
        for ax, hemisphere in zip(axrow, hemispheres):
            kp += 1
            if kp == n_plots:
                cb = colorbar
            else:
                cb = False
            if hemisphere == 'left':

                _ = nlp.plot_surf(lsurf,
                                  ldata,
                                  bg_map=surf_data_from_cifti(sulcal_depth.get_fdata(dtype=np.float32),
                                                              sulcal_depth.header.get_axis(1),
                                                              'CIFTI_STRUCTURE_CORTEX_LEFT').mean(axis=1),
                                  cmap=cmap, vmin=vmin, vmax=vmax, colorbar=cb, hemi='left', axes=ax, figure=fig,
                                  bg_on_data=True, darkness=0.5, view=view, **kwargs)
                if legend_elements is not None and (kp == n_plots):
                    ax.legend(handles=legend_elements)
            elif hemisphere == 'right':
                _ = nlp.plot_surf(rsurf,
                                  rdata,
                                  bg_map=surf_data_from_cifti(sulcal_depth.get_fdata(dtype=np.float32),
                                                              sulcal_depth.header.get_axis(1),
                                                              'CIFTI_STRUCTURE_CORTEX_RIGHT').mean(axis=1),
                                  cmap=cmap, vmin=vmin, vmax=vmax, colorbar=cb, hemi='right', axes=ax, figure=fig,
                                  bg_on_data=True, darkness=0.5, view=view, **kwargs)
                if legend_elements is not None and (kp == n_plots):
                    ax.legend(handles=legend_elements)

    return fig