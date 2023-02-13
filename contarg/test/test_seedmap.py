from pkg_resources import resource_filename
from pathlib import Path
import nilearn as nl
from nilearn import image, masking
import numpy as np
from contarg.seedmap import get_ref_vox_con


def test_get_ref_vox_con():
    data_path = Path(resource_filename("contarg", "test/data"))
    bold_paths = [
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz",
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz",
    ]
    mask_path = (
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-brain_mask.nii.gz"
    )
    refroi_path = (
        data_path
        / "derivatives/contarg/hierarchical/testing1sub/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-SGCsphere_mask.nii.gz"
    )
    out_dir = data_path / "derivatives/contarg/seedmap/test/sub-02/func"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sub-02_ses_desc-RefCon_stat.nii.gz"
    ref_path = (
        data_path
        / "derivatives/contarg/seedmap/test_ref/sub-02/func/sub-02_ses_desc-RefCon_stat.nii.gz"
    )
    ref_img = nl.image.load_img(ref_path)

    ref_vox_con = get_ref_vox_con(bold_paths, mask_path, refroi_path, 0.72, out_path)

    assert np.allclose(ref_vox_con.get_fdata(), ref_img.get_fdata())


def test_get_seedmap_vox_con():
    data_path = Path(resource_filename("contarg", "test/data"))
    bold_paths = [
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz",
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz",
    ]
    mask_path = (
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-brain_mask.nii.gz"
    )
    refroi_path = (
        data_path
        / "derivatives/contarg/hierarchical/testing1sub/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-SGCsphere_mask.nii.gz"
    )
    out_dir = data_path / "derivatives/contarg/seedmap/test/sub-02/func"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sub-02_ses_desc-RefCon_stat.nii.gz"
    ref_path = (
        data_path
        / "derivatives/contarg/seedmap/test_ref/sub-02/func/sub-02_ses_desc-RefCon_stat.nii.gz"
    )
    ref_img = nl.image.load_img(ref_path)

    ref_vox_con = get_ref_vox_con(bold_paths, mask_path, refroi_path, 0.72, out_path)

    assert np.allclose(ref_vox_con.get_fdata(), ref_img.get_fdata())

if __name__ == "__main__":
    test_get_ref_vox_con()
