"""test_stages.py — integration tests for 01/02/03 scripts."""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from rdkit import Chem

from util import curated_df

_SEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # selection/
S1 = os.path.join(_SEL, "01_descriptors.py")
S2 = os.path.join(_SEL, "02_env_analysis.py")
S3 = os.path.join(_SEL, "03_select.py")

SMILES = [
    "CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1",
    "CC(=O)O", "CC(=O)N", "C1CCCCC1",
]


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    assert r.returncode == 0, r.stderr
    return r


def _write_curated(tmp_path, smiles=None):
    d = tmp_path / "curated"
    d.mkdir(parents=True, exist_ok=True)
    curated_df(smiles or SMILES).to_parquet(d / "batch_0000.parquet", index=False)
    return d


def _run_descriptors(tmp_path, smiles=None, extra=None):
    curated = _write_curated(tmp_path, smiles)
    out = tmp_path / "out"
    cmd = [sys.executable, S1, "-i", str(curated), "-o", str(out)]
    if extra:
        cmd.extend(extra)
    _run(cmd)
    return out


class TestStage1:
    def test_output_columns(self, tmp_path):
        out = _run_descriptors(tmp_path)
        df = pd.read_parquet(out / "descriptors.parquet")
        assert "morgan_fp" in df.columns
        assert "atom_envs" in df.columns
        assert "mol_block" in df.columns
        assert "smiles" in df.columns
        assert len(df) == len(SMILES)

    def test_dedup_duplicate_compoundid(self, tmp_path):
        curated = _write_curated(tmp_path)
        df = curated_df(SMILES)
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)  # duplicate row 0
        df.to_parquet(curated / "batch_0000.parquet", index=False)
        out = tmp_path / "out"
        _run([sys.executable, S1, "-i", str(curated), "-o", str(out)])
        desc = pd.read_parquet(out / "descriptors.parquet")
        assert len(desc) == len(SMILES)  # duplicate removed

    def test_unparseable_logged(self, tmp_path):
        curated = _write_curated(tmp_path)
        df = curated_df(SMILES)
        df.loc[0, "mol_block"] = "garbage not a mol block"
        df.to_parquet(curated / "batch_0000.parquet", index=False)
        out = tmp_path / "out"
        r = _run([sys.executable, S1, "-i", str(curated), "-o", str(out)])
        desc = pd.read_parquet(out / "descriptors.parquet")
        assert len(desc) == len(SMILES) - 1
        assert "skipped 1" in r.stdout

    def test_skipped_rare_element_flagged(self, tmp_path):
        # neutral N valence-4 (sanitize fails) carrying a rare S atom
        m = Chem.RWMol()
        a_s = m.AddAtom(Chem.Atom("S"))
        a_n = m.AddAtom(Chem.Atom("N"))
        cs = [m.AddAtom(Chem.Atom("C")) for _ in range(4)]
        for c in cs:
            m.AddBond(a_n, c, Chem.BondType.SINGLE)
        m.AddBond(a_s, cs[0], Chem.BondType.SINGLE)
        bad_mb = Chem.MolToMolBlock(m)

        curated = _write_curated(tmp_path)
        df = curated_df(["CCO"])
        df.loc[0, "mol_block"] = bad_mb
        df.to_parquet(curated / "batch_0000.parquet", index=False)
        out = tmp_path / "out"
        _run([sys.executable, S1, "-i", str(curated), "-o", str(out)])
        skips = pd.read_csv(out / "skipped.csv")
        assert len(skips) == 1
        assert "S" in skips.iloc[0]["rare_elements"]

    def test_soap_stub(self, tmp_path):
        curated = _write_curated(tmp_path)
        r = subprocess.run(
            [sys.executable, S1, "-i", str(curated), "-o", str(tmp_path),
             "--metric", "soap"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "DScribe" in r.stderr


SCOPE_SMILES = ["CCO", "CCCl", "CC(F)(F)F", "CCS", "c1ccccc1"]


class TestStage1Scope:
    """D64 scope filter: --allowed-elements + --min/--max-atoms."""

    def test_allowed_elements_filters(self, tmp_path):
        out = _run_descriptors(tmp_path, smiles=SCOPE_SMILES,
                               extra=["--allowed-elements", "C,N,O"])
        desc = pd.read_parquet(out / "descriptors.parquet")
        # CCCl / CCF3 / CCS excluded (Cl, F, S outside whitelist)
        assert len(desc) == 2
        all_elems = {e for es in desc["heavy_elements"] for e in es}
        assert all_elems <= {"C", "N", "O"}  # no Cl/F/S leaked
        assert "O" in all_elems  # CCO retained

    def test_allowed_elements_case_insensitive(self, tmp_path):
        out = _run_descriptors(tmp_path, smiles=SCOPE_SMILES,
                               extra=["--allowed-elements", "c,n,o"])
        desc = pd.read_parquet(out / "descriptors.parquet")
        assert len(desc) == 2

    def test_invalid_element_symbol_rejected(self, tmp_path):
        curated = _write_curated(tmp_path)
        r = subprocess.run(
            [sys.executable, S1, "-i", str(curated), "-o", str(tmp_path),
             "--allowed-elements", "C,N,Xx"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "unknown element" in r.stderr

    def test_min_atoms_inclusive(self, tmp_path):
        # heavy counts: CC=2, CCO=3, CCCCCC=6
        out = _run_descriptors(tmp_path, smiles=["CC", "CCO", "CCCCCC"],
                               extra=["--min-atoms", "3"])
        desc = pd.read_parquet(out / "descriptors.parquet")
        assert len(desc) == 2  # CCO (==3, boundary) and CCCCCC

    def test_max_atoms_inclusive(self, tmp_path):
        out = _run_descriptors(tmp_path, smiles=["CC", "CCO", "CCCCCC"],
                               extra=["--max-atoms", "3"])
        desc = pd.read_parquet(out / "descriptors.parquet")
        assert len(desc) == 2  # CC and CCO (==3, boundary)

    def test_min_gt_max_rejected(self, tmp_path):
        curated = _write_curated(tmp_path)
        r = subprocess.run(
            [sys.executable, S1, "-i", str(curated), "-o", str(tmp_path),
             "--min-atoms", "20", "--max-atoms", "10"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
        assert "min-atoms" in r.stderr

    def test_exclusion_report_split_by_reason(self, tmp_path):
        # CCCl -> element exclusion; CC -> size exclusion
        out = _run_descriptors(tmp_path, smiles=["CCO", "CCCl", "CC"],
                               extra=["--allowed-elements", "C,N,O",
                                      "--min-atoms", "3"])
        desc = pd.read_parquet(out / "descriptors.parquet")
        assert len(desc) == 1  # only CCO

    def test_element_histogram_printed(self, tmp_path):
        r = subprocess.run(
            [sys.executable, S1, "-i", str(_write_curated(tmp_path)),
             "-o", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "elements:" in r.stdout
        assert "C=" in r.stdout


class TestParseAllowedElements:
    """Unit tests for the --allowed-elements spec parser."""

    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("s1_mod", S1)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_normalizes_case_and_dedupes(self):
        mod = self._load()
        assert mod.parse_allowed_elements("c,N,o,C") == {"C", "N", "O"}

    def test_unknown_symbol_raises(self):
        mod = self._load()
        with pytest.raises(ValueError, match="unknown element"):
            mod.parse_allowed_elements("C,N,Xx")

    def test_empty_spec_raises(self):
        mod = self._load()
        with pytest.raises(ValueError):
            mod.parse_allowed_elements("  , ,")


class TestStage2:
    def _descriptors(self, tmp_path, smiles=None):
        return _run_descriptors(tmp_path, smiles)

    def test_four_csvs(self, tmp_path):
        out = self._descriptors(tmp_path)
        a = tmp_path / "analysis"
        _run([sys.executable, S2, "-i", str(out / "descriptors.parquet"),
              "-o", str(a)])
        for name in ["env_frequency.csv", "cost_curve.csv",
                     "forced_choice.csv", "bond_types.csv"]:
            assert (a / name).exists()

    def test_forced_choice_classification(self, tmp_path):
        out = self._descriptors(tmp_path, smiles=["CC", "CCC", "CCO"])
        a = tmp_path / "analysis"
        _run([sys.executable, S2, "-i", str(out / "descriptors.parquet"),
              "-o", str(a)])
        fc = pd.read_csv(a / "forced_choice.csv")
        assert (fc.loc[fc["n_mols"] == 1, "pick_type"] == "forced").all()
        assert (fc.loc[fc["n_mols"] > 1, "pick_type"] == "choice").all()


class TestStage3:
    def _run(self, tmp_path, extra=None, smiles=None):
        out = _run_descriptors(tmp_path, smiles)
        sel_out = tmp_path / "sel"
        cmd = [sys.executable, S3, "-i", str(out / "descriptors.parquet"),
               "-o", str(sel_out), "--n", "5", "--tier1-min", "1", "--ff", "0.5",
               "--floor-budget", "0.2", "--depth-budget", "0.2"]
        if extra:
            cmd.extend(extra)
        _run(cmd)
        return sel_out

    def test_outputs_written(self, tmp_path):
        out = self._run(tmp_path)
        assert (out / "selection.parquet").exists()
        for name in ["coverage.csv", "thin_coverage.csv",
                     "bond_types.csv", "property_js.csv"]:
            assert (out / "reports" / name).exists()
        assert (out / "selection_summary.json").exists()

    def test_selection_bounds_and_labels(self, tmp_path):
        out = self._run(tmp_path)
        sel = pd.read_parquet(out / "selection.parquet")
        assert len(sel) <= 5
        assert sel["CompoundID"].is_unique
        assert set(sel["pass"]) <= {"floor", "diversity", "depth"}
        assert "mol_block" in sel.columns
        assert "score" in sel.columns

    def test_deterministic(self, tmp_path):
        o1 = self._run(tmp_path / "r1")
        o2 = self._run(tmp_path / "r2")
        s1 = pd.read_parquet(o1 / "selection.parquet")
        s2 = pd.read_parquet(o2 / "selection.parquet")
        assert list(s1["CompoundID"]) == list(s2["CompoundID"])

    def test_summary_counts_consistent(self, tmp_path):
        out = self._run(tmp_path)
        summary = json.loads((out / "selection_summary.json").read_text())
        sel = pd.read_parquet(out / "selection.parquet")
        assert summary["n_selected"] == len(sel)
        assert summary["n_floor"] == (sel["pass"] == "floor").sum()
        assert summary["n_diversity"] == (sel["pass"] == "diversity").sum()
        assert summary["n_depth"] == (sel["pass"] == "depth").sum()

    def test_floor_only_pass(self, tmp_path):
        out = self._run(tmp_path, extra=["--passes", "floor"])
        sel = pd.read_parquet(out / "selection.parquet")
        assert set(sel["pass"]) == {"floor"}

    def test_floor_k_mode(self, tmp_path):
        out = self._run(tmp_path, extra=["--floor-k", "3"])
        summary = json.loads((out / "selection_summary.json").read_text())
        assert summary["floor_k"] == 3

    def test_n_geq_pool_selects_all(self, tmp_path):
        out = self._run(tmp_path, extra=["--n", "100"])
        sel = pd.read_parquet(out / "selection.parquet")
        assert len(sel) == len(SMILES)
