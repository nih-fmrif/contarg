from pkg_resources import resource_filename
from pathlib import Path
import nilearn as nl
from nilearn import image, masking
import numpy as np
from contarg.utils import (
    cluster,
    build_bidsname,
    parse_bidsname,
    update_bidspath,
    get_rel_path,
    make_rel_symlink,
)
import pytest
import tempfile
import shutil


def test_cluster():
    data_path = Path(resource_filename("contarg", "test/data"))
    stat_img_path = (
        data_path
        / "derivatives/contarg/seedmap/test_ref/sub-02/func/sub-02_desc-RefCon_stat.nii.gz"
    )
    percentile = 10
    sign = "negative"
    connectivity = "NN3"
    stim_roi_path = (
        data_path
        / "derivatives/contarg/hierarchical/testing1sub_ref/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-DLPFCspheresClean_mask.nii.gz"
    )
    ref_path = (
        data_path
        / "derivatives/contarg/cluster/test_ref/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-biggestclust_mask.nii.gz"
    )
    out_path = (
        data_path
        / "derivatives/contarg/cluster/test/sub-02/func/sub-02_task-rest_run-1_atlas-Coords_space-T1w_desc-biggestclust_mask.nii.gz"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cluster(stat_img_path, out_path, stim_roi_path, percentile, sign, connectivity)
    out_dat = nl.image.load_img(out_path).get_fdata()
    ref_dat = nl.image.load_img(ref_path).get_fdata()
    assert np.allclose(out_dat, ref_dat)


def test_parse_and_build_bidsname():
    test_name = "sub-02_ses-1_task-rest_run-01_echo-2_acq-1.2.3.4_bad-wrong_break_me_space-T1w_desc-preproc_bold.nii.gz"
    with pytest.raises(ValueError):
        build_bidsname(parse_bidsname(test_name))
    test_name = "func/sub-02_ses-1_task-rest_run-01_echo-2_acq-1.2.3.4_space-T1w_desc-preproc_bold.nii.gz"
    expected_result = "func/sub-02_ses-1_task-rest_acq-1.2.3.4_run-01_echo-2_space-T1w_desc-preproc_bold.nii.gz"
    assert build_bidsname(parse_bidsname(test_name)) == expected_result
    test_name = "foo/func/sub-02_ses-1_task-rest_acq-1.2.3.4_run-01_echo-2_space-T1w_desc-preproc_bold.nii.gz"
    assert (
        build_bidsname(parse_bidsname(test_name))
        == "func/sub-02_ses-1_task-rest_acq-1.2.3.4_run-01_echo-2_space-T1w_desc-preproc_bold.nii.gz"
    )


def test_update_bidspath():
    # Test 1: Update sub and ses, exclude ses
    orig_path = "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
    new_bids_root = "/data/bids"
    updates = {"sub": "02", "ses": "02"}
    exclude = ["ses"]
    order = ["type", "sub", "acq", "task", "run", "modality", "suffix", "extension"]
    expected_result = "/data/bids/sub-02/anat/sub-02_T1w.nii.gz"

    result = update_bidspath(
        orig_path, new_bids_root, updates, exists=False, exclude=exclude, order=order
    )

    assert result.as_posix() == expected_result

    # Test 2: Update sub and ses, include ses
    orig_path = "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
    new_bids_root = "/data/bids"
    updates = {"sub": "02", "ses": "02"}
    exclude = []
    expected_result = "/data/bids/sub-02/ses-02/anat/sub-02_ses-02_T1w.nii.gz"

    result = update_bidspath(
        orig_path, new_bids_root, updates, exists=False, exclude=exclude
    )

    assert result.as_posix() == expected_result

    # Test 3: Update sub, ses and acq, include ses and acq
    orig_path = "sub-01/ses-01/anat/acq-mprage/sub-01_ses-01_acq-mprage_T1w.nii.gz"
    new_bids_root = "/data/bids"
    updates = {
        "sub": "02",
        "ses": "02",
        "acq": "mag",
        "type": "func",
        "suffix": "bold",
        "task": "rest",
    }
    exclude = []
    expected_result = (
        "/data/bids/sub-02/ses-02/func/sub-02_ses-02_task-rest_acq-mag_bold.nii.gz"
    )

    result = update_bidspath(
        orig_path, new_bids_root, updates, exists=False, exclude=exclude
    )

    assert result.as_posix() == expected_result

    # Test 4: Update sub and ses, with `exists=True`, raises FileNotFoundError if the new path doesn't exist
    orig_path = "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
    new_bids_root = "/data/bids"
    updates = {"sub": "02", "ses": "02"}
    exclude = []
    expected_result = "/data/bids/sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz"

    with pytest.raises(FileNotFoundError):
        result = update_bidspath(
            orig_path, new_bids_root, updates, exists=True, exclude=exclude
        )


def test_get_rel_path():
    # test if the function returns the correct relative path
    assert get_rel_path("a/b/c/d", "a/e/f") == Path("../../e/f")
    assert get_rel_path("a/b/c/d", "a/b/c/f") == Path("f")
    assert get_rel_path("a/b/c/d", "a/b/c/d/e/f") == Path("e/f")
    assert get_rel_path("a/b/c/d", "a/b/c/d") == Path(".")
    assert get_rel_path("/a/b/c/d", "/e/f/g/h") == Path("../../../e/f/g/h")


def test_make_rel_symlink():
    # create temporary directories and files for testing
    with tempfile.TemporaryDirectory() as tempdir:
        source_dir = Path(tempdir) / "source"
        target_dir = Path(tempdir) / "target"
        source_dir.mkdir()
        target_dir.mkdir()
        (target_dir / "file.txt").touch()

        # test if the function creates a relative symlink
        symlink_path = source_dir / "symlink"
        make_rel_symlink(symlink_path, target_dir / "file.txt")
        assert symlink_path.is_symlink()
        assert symlink_path.resolve() == (target_dir / "file.txt").resolve()

        # test if the function raises an error for an existing symlink that doesn't point to the target
        with pytest.raises(ValueError):
            make_rel_symlink(symlink_path, target_dir)


if __name__ == "__main__":
    test_cluster()
