from pkg_resources import resource_filename
from pathlib import Path
import nilearn as nl
from nilearn import image, masking
import numpy as np
import pandas as pd
from contarg.seedmap import get_ref_vox_con, get_seedmap_vox_con
from click.testing import CliRunner
from contarg.cli.cli import contarg
from contarg.cli.run_seedmap import run

def test_get_ref_vox_con():
    data_path = Path(resource_filename("contarg", "test/data"))
    bold_paths = [
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz",
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-2_space-T1w_desc-preproc_bold.nii.gz",
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
    out_path = out_dir / "sub-02_desc-RefCon_stat.nii.gz"
    ref_path = (
        data_path
        / "derivatives/contarg/seedmap/test_ref/sub-02/func/sub-02_desc-RefCon_stat.nii.gz"
    )
    ref_img = nl.image.load_img(ref_path)

    ref_vox_con = get_ref_vox_con(bold_paths, mask_path, refroi_path, 1.9, out_path)

    assert np.allclose(ref_vox_con.get_fdata(), ref_img.get_fdata())


def test_get_seedmap_vox_con():
    data_path = Path(resource_filename("contarg", "test/data"))
    roi_dir = Path(resource_filename("contarg", "data/rois"))
    bold_paths = [
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-2_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
    ]
    mask_path = (
        data_path
        / "derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-1_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz"
    )
    seedmap_path = (
        data_path
        / "derivatives/contarg/seedmap/hcp_working/SGCsphere_ses-REST1_space-MNI152NLin6Asym_res-2_desc-HCPgroupmask_mask.nii.gz"
    )
    stimroi_path = roi_dir/"DLPFCspheresmasked_space-MNI152NLin6Asym_res-02.nii.gz"
    out_dir = data_path / "derivatives/contarg/seedmap/test/sub-02/func"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sub-02_desc-SeedmapCon_stat.nii.gz"
    sm_con = get_seedmap_vox_con(bold_paths, mask_path, seedmap_path, stimroi_path, n_dummy=5, tr=1.9, out_path=out_path)
    ref_path = (
        data_path
        / "derivatives/contarg/seedmap/test_ref/sub-02/func/sub-02_desc-SeedmapCon_stat.nii.gz"
    )
    ref_img = nl.image.load_img(ref_path)

    assert np.allclose(sm_con.get_fdata(), ref_img.get_fdata())

# uncomment if you've commented out all the click decorators on run_seedmap.run
# def test_run():
#     # get paths
#     data_path = Path(resource_filename("contarg", "test/data"))
#     bids_dir = data_path / "ds002330"
#     derivatives_dir = data_path / "derivatives"
#     database_file = data_path / "pybids_0.15.2_db"
#     seedmap_path = (
#             data_path
#             / "derivatives/contarg/seedmap/hcp_working/SGCsphere_ses-REST1_space-MNI152NLin6Asym_res-2_desc-HCPgroupmask_mask.nii.gz"
#
#
#     )
#     run(bids_dir,
#         derivatives_dir,
#         database_file,
#         run_name="testing1sub",
#         stimroi_name="DLPFCspheres",
#         stimroi_path=None,
#         seedmap_path=seedmap_path,
#         space="T1w",
#         smoothing_fwhm=4,
#         n_dummy=5,
#         t_r=1.9,
#         target_method="cluster",
#         connectivity="NN3",
#         percentile=10,
#         subject="02",
#         session=None,
#         run=None,
#         echo=None,
#         njobs=1
#         )

def test_single_subject_MNI():
    # get paths
    data_path = Path(resource_filename("contarg", "test/data"))
    bids_dir = data_path / "ds002330"
    derivatives_dir = data_path / "derivatives"
    database_file = data_path / "pybids_0.15.2_db"
    seedmap_path = (
        data_path
        / "derivatives/contarg/seedmap/hcp_working/SGCsphere_ses-REST1_space-MNI152NLin6Asym_res-2_desc-HCPgroupmask_mask.nii.gz"
    )
    runner = CliRunner()
    result = runner.invoke(
        contarg,
        [
            "seedmap",
            "run",
            f"--bids-dir={bids_dir}",
            f"--derivatives-dir={derivatives_dir}",
            f"--database-file={database_file}",
            "--run-name=testing1subMNI",
            "--stimroi-name=DLPFCspheres",
            f"--seedmap-path={seedmap_path}",
            "--space=MNI152NLin6Asym",
            "--smoothing-fwhm=4",
            "--ndummy=5",
            "--tr=1.9",
            "--percentile=10",
            "--target-method=cluster",
            "--subject=02",
        ],
        catch_exceptions=False
    )
    assert result.exit_code == 0
    output_dir = (
        derivatives_dir / "contarg" / "seedmap" / "testing1subMNI" / "sub-02" / "func"
    )
    reference_dir = (
        derivatives_dir
        / "contarg"
        / "seedmap"
        / "testing1subMNI_ref"
        / "sub-02"
        / "func"
    )
    assert reference_dir.exists()
    assert output_dir.exists()
    niftis = sorted(reference_dir.glob("*.nii.gz"))
    tsvs = sorted(reference_dir.glob("*.tsv"))
    for nii in niftis:
        out_nii = output_dir / nii.name
        out_img = nl.image.load_img(out_nii)
        out_dat = out_img.get_fdata()
        ref_dat = nl.image.load_img(nii).get_fdata()
        assert np.allclose(out_dat, ref_dat)
    for tsv in tsvs:
        out_tsv = output_dir / tsv.name
        out_df = pd.read_csv(out_tsv, sep="\t")
        ref_df = pd.read_csv(tsv, sep="\t")
        assert out_df.equals(ref_df)

def test_single_subject():
    # get paths
    data_path = Path(resource_filename("contarg", "test/data"))
    bids_dir = data_path / "ds002330"
    derivatives_dir = data_path / "derivatives"
    database_file = data_path / "pybids_0.15.2_db"
    seedmap_path = (
        data_path
        / "derivatives/contarg/seedmap/hcp_working/SGCsphere_ses-REST1_space-MNI152NLin6Asym_res-2_desc-HCPgroupmask_mask.nii.gz"
    )
    runner = CliRunner()
    result = runner.invoke(
        contarg,
        [
            "seedmap",
            "run",
            f"--bids-dir={bids_dir}",
            f"--derivatives-dir={derivatives_dir}",
            f"--database-file={database_file}",
            "--run-name=testing1sub",
            "--stimroi-name=DLPFCspheres",
            f"--seedmap-path={seedmap_path}",
            "--space=T1w",
            "--smoothing-fwhm=4",
            "--ndummy=5",
            "--tr=1.9",
            "--percentile=10",
            "--target-method=cluster",
            "--subject=02",
        ],
        catch_exceptions=False
    )
    assert result.exit_code == 0
    output_dir = (
        derivatives_dir / "contarg" / "seedmap" / "testing1sub" / "sub-02" / "func"
    )
    reference_dir = (
        derivatives_dir
        / "contarg"
        / "seedmap"
        / "testing1sub_ref"
        / "sub-02"
        / "func"
    )
    assert reference_dir.exists()
    assert output_dir.exists()
    niftis = sorted(reference_dir.glob("*.nii.gz"))
    tsvs = sorted(reference_dir.glob("*.tsv"))
    for nii in niftis:
        out_nii = output_dir / nii.name
        out_img = nl.image.load_img(out_nii)
        out_dat = out_img.get_fdata()
        ref_dat = nl.image.load_img(nii).get_fdata()
        assert np.allclose(out_dat, ref_dat)
    for tsv in tsvs:
        out_tsv = output_dir / tsv.name
        out_df = pd.read_csv(out_tsv, sep="\t")
        ref_df = pd.read_csv(tsv, sep="\t")
        assert out_df.equals(ref_df)

if __name__ == "__main__":
    test_get_ref_vox_con()
    test_get_seedmap_vox_con()
