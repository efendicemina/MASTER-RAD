from pathlib import Path

import pandas as pd

from defect_classifier import transfer_study


def test_process_processed_root_extracts_projects(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    # create a parquet with multiple projects
    df = pd.DataFrame(
        {
            "project": ["projA"] * 3 + ["projB"] * 2 + ["MYLYN"] * 1,
            "creation_time": ["2020-01-01"] * 6,
            "severity": ["normal", "major", "minor", "normal", "major", "minor"],
        }
    )
    pq = processed / "sample.parquet"
    df.to_parquet(pq)

    summary = transfer_study._process_processed_root(processed)
    assert "projA" in list(summary["project"])
    assert "projB" in list(summary["project"])
    assert "MYLYN" in list(summary["project"])
    assert any(summary[summary["project"] == "projA"]["rows"] == 3)


def test_candidate_matrix_checksum(tmp_path: Path):
    audit = tmp_path / "audit"
    audit.mkdir()
    df = pd.DataFrame([{"candidate_id": "c1", "model": "LogisticRegression"}])
    df.to_csv(audit / "candidate_matrix.csv", index=False)
    chk = transfer_study._candidate_matrix_checksum(audit)
    assert isinstance(chk, str)
    assert len(chk) == 64
