from click.testing import CliRunner
from contarg.cli.cli import contarg
from pkg_resources import resource_filename
from pathlib import Path
import nilearn as nl
from nilearn import image, masking
import numpy as np
import pandas as pd


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
            "--njobs=4",
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


if __name__ == "__main__":
    test_single_subject()
    test_multi_subject()
