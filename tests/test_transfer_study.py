from pathlib import Path

import pytest

from defect_classifier import development_study, transfer_study


def test_loading_test_split_is_rejected(tmp_path: Path):
    # create a fake test_split.csv file
    p = tmp_path / "test_split.csv"
    p.write_text("id,creation_time,severity,summary,description\n1,2020-01-01,normal,foo,bar")
    with pytest.raises(ValueError):
        development_study.load_development_only(p)


def test_verify_development_sha_mismatch(tmp_path: Path):
    p = tmp_path / "development_split.csv"
    p.write_text("id,creation_time,severity,summary,description\n1,2020-01-01,normal,foo,bar")
    # use an incorrect sha
    with pytest.raises(ValueError):
        transfer_study.verify_development_file(p, "deadbeef")


def test_find_select_development_by_sha(tmp_path: Path):
    reports_root = tmp_path / "reports"
    tables_dir = reports_root / "experiments" / "exp1" / "tables"
    tables_dir.mkdir(parents=True)
    dev = tables_dir / "development_split.csv"
    dev.write_text("id,creation_time,severity,summary,description\n1,2020-01-01,normal,foo,bar")
    # compute sha and verify selection
    # call select_development_by_sha by computing actual sha
    from defect_classifier.utils import sha256_file

    actual = sha256_file(dev)
    candidates = transfer_study.find_development_candidates(reports_root)
    assert dev in candidates
    selected = transfer_study.select_development_by_sha(candidates, actual)
    assert selected == dev
