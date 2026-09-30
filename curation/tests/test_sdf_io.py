"""test_sdf_io.py — Unit tests for lib/sdf_io.py."""

from lib.schema import OPTIONAL_TAGS, RECOMMENDED_TAGS, REQUIRED_TAGS
from lib.sdf_io import read_sdf, validate_tags

# -- read_sdf -------------------------------------------------------------


class TestReadSdf:
    def test_reads_all_molecules(self, valid_sdf):
        entries = read_sdf(valid_sdf)
        assert len(entries) == 3

    def test_parses_tags_correctly(self, valid_sdf):
        entries = read_sdf(valid_sdf)
        mol, tags = entries[0]
        assert mol is not None
        assert tags["Energy_Ha"] == "-500.0"
        assert tags["Multiplicity"] == "1"


# -- validate_tags --------------------------------------------------------


class TestValidateTags:
    SPECS = (REQUIRED_TAGS, RECOMMENDED_TAGS, OPTIONAL_TAGS)

    def test_all_valid_pass(self, valid_sdf):
        entries = read_sdf(valid_sdf)
        passed, rejected, warnings = validate_tags(entries, *self.SPECS)
        assert len(passed) == 3
        assert len(rejected) == 0
        assert len(warnings) == 0

    def test_missing_required_rejected(self, broken_missing_required):
        entries = read_sdf(broken_missing_required)
        passed, rejected, _ = validate_tags(entries, *self.SPECS)
        assert len(passed) == 0
        assert len(rejected) == 2

    def test_bad_types_rejected(self, broken_bad_types):
        entries = read_sdf(broken_bad_types)
        passed, rejected, _ = validate_tags(entries, *self.SPECS)
        assert len(passed) == 0
        assert len(rejected) == 2

    def test_multiplicity_zero_rejected(self, broken_mult_zero):
        entries = read_sdf(broken_mult_zero)
        passed, rejected, _ = validate_tags(entries, *self.SPECS)
        assert len(passed) == 0
        assert len(rejected) == 1

    def test_missing_recommended_warns_not_reject(self, missing_recommended):
        entries = read_sdf(missing_recommended)
        passed, rejected, warnings = validate_tags(entries, *self.SPECS)
        assert len(passed) == 1
        assert len(rejected) == 0
        assert len(warnings) == 1
        idx = list(warnings.keys())[0]
        assert any("SMILES" in w for w in warnings[idx])
        assert any("SourceID" in w for w in warnings[idx])

    def test_optional_missing_warns_not_rejects(self, missing_optional):
        entries = read_sdf(missing_optional)
        passed, rejected, warnings = validate_tags(entries, *self.SPECS)
        assert len(passed) == 1
        assert len(rejected) == 0
        assert len(warnings) == 1
        idx = list(warnings.keys())[0]
        assert any("HOMO_Ha" in w for w in warnings[idx])

    def test_unknown_tags_pass_through(self, valid_sdf):
        entries = read_sdf(valid_sdf)
        mol, tags = entries[0]
        mol.SetProp("CustomField", "hello")
        tags["CustomField"] = "hello"
        passed, rejected, _ = validate_tags([(mol, tags)], *self.SPECS)
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_strict_false_warns_not_rejects(self, broken_missing_required):
        entries = read_sdf(broken_missing_required)
        passed, rejected, warnings = validate_tags(entries, *self.SPECS, strict=False)
        assert len(passed) == 2  # both pass in lenient mode
        assert len(rejected) == 0
        assert len(warnings) == 2

    def test_reject_errors_contain_detail(self, broken_bad_types):
        entries = read_sdf(broken_bad_types)
        _, rejected, _ = validate_tags(entries, *self.SPECS)
        idx0 = list(rejected.keys())[0]
        errs = rejected[idx0][2]
        assert any("not parseable as float" in e for e in errs)
        idx1 = list(rejected.keys())[1]
        errs1 = rejected[idx1][2]
        assert any("not parseable as integer" in e for e in errs1)
