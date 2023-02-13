from pkg_resources import resource_filename
from pathlib import Path
import nilearn as nl
from nilearn import image, masking
import numpy as np
from contarg.utils import cluster

def test_cluster():
    data_path = Path(resource_filename("contarg", "test/data"))
    stat_img_path = data_path / 'derivatives/contarg/seedmap/test_ref/sub-02/func/sub-02_ses_desc-RefCon_stat.nii.gz'
    percentile = 10
    sign = "negative"
    connectivity = "NN3"
    stim_roi_path = data_path / 'derivatives/contarg/hierarchical/testing1sub_ref/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-DLPFCspheresClean_mask.nii.gz'
    ref_path = data_path / 'derivatives/contarg/cluster/test_ref/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-biggestclust_mask.nii.gz'
    out_path = data_path / 'derivatives/contarg/cluster/test/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-biggestclust_mask.nii.gz'


    out_path.parent.mkdir(parents=True, exist_ok=True)
    cluster(stat_img_path, out_path, stim_roi_path, percentile, sign, connectivity)
    out_dat = nl.image.load_img(out_path).get_fdata()
    ref_dat = nl.image.load_img(ref_path).get_fdata()
    assert np.allclose(out_dat, ref_dat)

if __name__ == "__main__":
    test_cluster()