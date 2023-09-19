from click.testing import CliRunner
from contarg.cli.cli import contarg
from pkg_resources import resource_filename
from pathlib import Path
import nilearn as nl
from nilearn import image, masking
import numpy as np
import pandas as pd
import pytest


def test_single_subject():
    # get paths
    data_path = Path(resource_filename("contarg", "test/data"))
    bids_dir = data_path / "ds002330"
    derivatives_dir = data_path / "derivatives"
    database_file = data_path / "pybids_0.15.2_db"
    runner = CliRunner()
    result = runner.invoke(
        contarg,
        [
            "hierarchical",
            "run",
            f"--bids-dir={bids_dir}",
            f"--derivatives-dir={derivatives_dir}",
            f"--database-file={database_file}",
            "--run-name=testing1sub",
            "--space=T1w",
            "--smoothing-fwhm=3",
            "--ndummy=5",
            "--tr=1.9",
            "--subject=02",
            "--run=1",
            "--njobs=2",
        ],
    )
    if result.exit_code != 0:
        print(result.exception, flush=True)
    assert result.exit_code == 0
    output_dir = (
        derivatives_dir / "contarg" / "hierarchical" / "testing1sub" / "sub-02" / "func"
    )
    reference_dir = (
        derivatives_dir
        / "contarg"
        / "hierarchical"
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


@pytest.mark.skipif(
    "not config.getoption('--run-big')",
    reason="Only run when --run-big is given",
)
def test_multi_subject():
    # get paths
    data_path = Path(resource_filename("contarg", "test/data"))
    bids_dir = data_path / "ds002330"
    derivatives_dir = data_path / "derivatives"
    database_file = data_path / "pybids_0.15.2_db"
    runner = CliRunner()
    result = runner.invoke(
        contarg,
        [
            "hierarchical",
            "run",
            f"--bids-dir={bids_dir}",
            f"--derivatives-dir={derivatives_dir}",
            f"--database-file={database_file}",
            "--run-name=testing2subs",
            "--space=T1w",
            "--smoothing-fwhm=3",
            "--ndummy=5",
            "--tr=1.9",
            "--njobs=1",
        ],
    )
    assert result.exit_code == 0
    output_dir = (
        derivatives_dir
        / "contarg"
        / "hierarchical"
        / "testing2subs"
        / "sub-02"
        / "func"
    )
    reference_dir = (
        derivatives_dir
        / "contarg"
        / "hierarchical"
        / "testing2subs_ref"
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
    output_dir = (
        derivatives_dir
        / "contarg"
        / "hierarchical"
        / "testing2subs"
        / "sub-03"
        / "func"
    )
    reference_dir = (
        derivatives_dir
        / "contarg"
        / "hierarchical"
        / "testing2subs_ref"
        / "sub-03"
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


def test_get_normed_neg_minus_pos():

    pears_rs = np.array([[ 0.1, -0.3,  0.4],
                         [-0.2,  0.2 , -0.1],
                         [ 0.3, -0.1,  0.9 ]])

    mcsig = np.array([[False, True, True],
                      [False, False, False],
                      [True, False,  True]])

    cluster_sizes = np.array([5, 7, 8])

    # expected values
    num_neg_corr = np.array([7, 0, 0])
    num_pos_corr = np.array([8, 0, 13])
    norm_neg_minus_pos = np.array([-0.05,  0.  , -0.65])

    # test function
    out_num_neg_corr, out_num_pos_corr, out_norm_neg_minus_pos = get_normed_neg_minus_pos(pears_rs, mcsig, cluster_sizes)

    # test assertions
    np.testing.assert_allclose(out_num_neg_corr, num_neg_corr)
    np.testing.assert_allclose(out_num_pos_corr, num_pos_corr)
    np.testing.assert_allclose(out_norm_neg_minus_pos, norm_neg_minus_pos)


if __name__ == "__main__":
    test_single_subject()
    test_multi_subject()
    test_get_normed_neg_minus_pos()
