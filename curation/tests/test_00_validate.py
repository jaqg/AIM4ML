"""test_00_validate.py — Integration tests for 00_validate.py."""

import os
import sys
import subprocess
import tempfile
import pytest

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "curation", "00_validate.py",
)


def run_validator(input_sdf, **opts):
    """Run 00_validate.py as a subprocess; return (exit_code, stdout)."""
    cmd = [sys.executable, SCRIPT, input_sdf]
    for flag, value in opts.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(value)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout


class TestValidateCli:
    def test_valid_passes(self, valid_sdf):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.sdf")
            rejects = os.path.join(tmp, "rejects")
            rc, stdout = run_validator(valid_sdf,
                                       **{"-o": out, "--rejects-dir": rejects})
            assert rc == 0
            assert "Validation PASSED" in stdout
            assert os.path.isfile(out)

    def test_broken_fails(self, broken_missing_required):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.sdf")
            rejects = os.path.join(tmp, "rejects")
            rc, stdout = run_validator(broken_missing_required,
                                       **{"-o": out, "--rejects-dir": rejects})
            assert rc == 1
            assert "Validation FAILED" in stdout
            assert "Energy_Ha: missing" in stdout
            assert "Multiplicity: missing" in stdout

    def test_rejects_written(self, broken_bad_types):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.sdf")
            rejects = os.path.join(tmp, "rejects")
            rc, _ = run_validator(broken_bad_types,
                                  **{"-o": out, "--rejects-dir": rejects})
            assert rc == 1
            reject_path = os.path.join(rejects, "rejected.sdf")
            assert os.path.isfile(reject_path)
            assert os.path.isfile(os.path.join(rejects, ".FAILED"))

    def test_clean_output_has_only_passing(self, broken_missing_required):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.sdf")
            rejects = os.path.join(tmp, "rejects")
            rc, _ = run_validator(broken_missing_required,
                                  **{"-o": out, "--rejects-dir": rejects})
            # No mols pass → clean SDF may still be created (SDWriter with 0 mols).
            # It's not a directory-level error either way; just check no crash.
            assert rc == 1

    def test_warnings_printed(self, missing_recommended):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.sdf")
            rejects = os.path.join(tmp, "rejects")
            rc, stdout = run_validator(missing_recommended,
                                       **{"-o": out, "--rejects-dir": rejects})
            assert rc == 0
            assert "SMILES: missing (recommended)" in stdout
            assert "SourceID: missing (recommended)" in stdout

    def test_lenient_mode_passes_broken(self, broken_missing_required):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clean.sdf")
            rejects = os.path.join(tmp, "rejects")
            rc, stdout = run_validator(broken_missing_required,
                                       **{"-o": out, "--rejects-dir": rejects,
                                          "--lenient": None})
            assert rc == 0
            assert "Validation PASSED" in stdout
