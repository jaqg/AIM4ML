"""test_schema.py — Unit tests for lib/schema.py."""

from lib.schema import COMPUTED_TAGS, OPTIONAL_TAGS, RECOMMENDED_TAGS, REQUIRED_TAGS, compound_id


class TestCompoundId:
    def test_deterministic(self):
        cid1 = compound_id("CCO")
        cid2 = compound_id("CCO")
        assert cid1 == cid2

    def test_different_smiles_different_id(self):
        assert compound_id("CCO") != compound_id("CCN")

    def test_known_hash(self):
        # MD5 of "CCO" — deterministic reference value
        import hashlib

        expected = hashlib.md5(b"CCO").hexdigest()
        assert compound_id("CCO") == expected


class TestTagDefinitions:
    def test_required_keys(self):
        assert set(REQUIRED_TAGS.keys()) == {"Energy_Ha", "FormalCharge", "Multiplicity"}

    def test_required_types(self):
        assert REQUIRED_TAGS["Energy_Ha"] is float
        assert REQUIRED_TAGS["FormalCharge"] is int
        assert REQUIRED_TAGS["Multiplicity"] is int

    def test_recommended_keys(self):
        assert set(RECOMMENDED_TAGS.keys()) == {"SMILES", "SourceID"}

    def test_optional_keys(self):
        assert set(OPTIONAL_TAGS.keys()) == {"HOMO_Ha", "LUMO_Ha", "HL_Gap_Ha", "PartialCharges"}

    def test_no_overlap_required_recommended(self):
        assert not (set(REQUIRED_TAGS) & set(RECOMMENDED_TAGS))

    def test_computed_tags_are_not_input(self):
        all_input = set(REQUIRED_TAGS) | set(RECOMMENDED_TAGS) | set(OPTIONAL_TAGS)
        assert not (all_input & set(COMPUTED_TAGS))


class TestParquetSchema:
    """Schema drift safety net: all stage columns registered, warning on extras."""

    def test_columns_cover_all_stages(self):
        from lib.schema import PARQUET_COLUMNS

        for col in [
            "energy_status",
            "filter_status",
            "filter_reason",
            "n_fragments",
            "CanonicalSMILES",
            "ICONF",
            "Formula",
            "dedup_status",
            "dedup_reason",
            "AtropisomerKey",
            "stereo_status",
            "reorder_status",
            "conformer_status",
            "cluster_id",
        ]:
            assert col in PARQUET_COLUMNS, col

    def test_optional_is_complement_of_required(self):
        from lib.schema import PARQUET_COLUMNS, PARQUET_OPTIONAL, PARQUET_REQUIRED

        assert set(PARQUET_OPTIONAL) == set(PARQUET_COLUMNS) - set(PARQUET_REQUIRED)

    def test_unregistered_column_warns(self):
        import os
        import tempfile
        import warnings

        from lib.parquet_io import write_batch

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.parquet")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                write_batch(
                    p,
                    [
                        {
                            "mol_block": "x",
                            "num_atoms": 1,
                            "num_bonds": 0,
                            "Energy_Ha": -1.0,
                            "FormalCharge": 0,
                            "Multiplicity": 1,
                            "UNREGISTERED": "v",
                        }
                    ],
                )
            assert any("UNREGISTERED" in str(x.message) for x in w)

    def test_read_batch_backfills_late_stage_columns(self):
        import math
        import os
        import tempfile

        from lib.parquet_io import read_batch, write_batch

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.parquet")
            write_batch(
                p,
                [
                    {
                        "mol_block": "x",
                        "num_atoms": 1,
                        "num_bonds": 0,
                        "Energy_Ha": -1.0,
                        "FormalCharge": 0,
                        "Multiplicity": 1,
                    }
                ],
            )
            rows = read_batch(p)
            assert len(rows) == 1
            # late-stage columns accessible without KeyError; null is None
            # (string cols) or NaN (int32 cols through pandas)
            for col in ["cluster_id", "filter_status", "conformer_status", "energy_status"]:
                assert col in rows[0], col
                v = rows[0][col]
                assert v is None or (isinstance(v, float) and math.isnan(v))
