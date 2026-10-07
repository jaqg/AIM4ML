"""
realspace_fixtures.py — Realspace xyz fixture texts + helpers (T1).

Moved out of conftest.py: realspace test files used `from conftest import`,
which collides with selection/tests/conftest.py (same top-level module name)
when curation + selection tests run in one pytest session.  Unique module
name -> no shadowing.

Coordinates are %.6f-representable so xyz_io round-trips are value-exact.
"""

XYZ_HEH_EXTXYZ = """2
Properties=species:S:1:pos:R:3 charge=1 multiplicity=1
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""

XYZ_HEH_PERATOM_CHARGE = """2
Properties=species:S:1:pos:R:3:charge:R:1 charge=1 multiplicity=1
He 0.000000 0.000000 0.000000 0.300000
H 0.772000 0.000000 0.000000 -0.300000
"""

XYZ_HEH_PLAIN = """2
HeH+ cation
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""

XYZ_TWO_FRAMES = XYZ_HEH_EXTXYZ + "\n" + XYZ_HEH_PLAIN

XYZ_MALFORMED_EOF = """3
Properties=species:S:1:pos:R:3
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""

XYZ_LI4_EXTXYZ = """4
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
Li 0.000000 0.000000 0.000000
Li 3.000000 0.000000 0.000000
Li 1.000000 2.500000 0.300000
Li 0.800000 0.600000 2.700000
"""

XYZ_B2H6_EXTXYZ = """8
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
B 0.000000 0.000000 1.190000
B 0.000000 0.000000 -1.190000
H 0.000000 1.200000 0.000000
H 0.000000 -1.200000 0.000000
H 1.030000 0.000000 1.830000
H -1.030000 0.000000 1.830000
H 1.030000 0.000000 -1.830000
H -1.030000 0.000000 -1.830000
"""

# Rigid translation of XYZ_HEH_EXTXYZ by (5, 5, 5): same composition group
# (HHe, charge=1, mult=1), DIFFERENT geometry_key (registry identity is
# absolute-position), RMSD 0 A -> must be caught by near-dup clustering.
XYZ_HEH_EXTXYZ_TRANSLATED = """2
Properties=species:S:1:pos:R:3 charge=1 multiplicity=1
He 5.000000 5.000000 5.000000
H 5.772000 5.000000 5.000000
"""

# Li4 conformer 2 (P4 fixture: atom 3 moved ~1.95 A -> RMSD 0.479 A, kept).
XYZ_LI4_CONFORMER2_EXTXYZ = """4
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
Li 0.000000 0.000000 0.000000
Li 3.000000 0.000000 0.000000
Li 1.000000 2.500000 0.300000
Li 2.300000 1.600000 0.900000
"""

# B2H2 isomer pair (P4 fixtures: RMSD > 0.25 A -> both kept, same group).
XYZ_ISO_A_EXTXYZ = """4
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
B 0.000000 0.000000 0.000000
H -1.000000 0.000000 0.000000
B 1.700000 0.000000 0.000000
H 2.700000 0.000000 0.000000
"""

XYZ_ISO_B_EXTXYZ = """4
Properties=species:S:1:pos:R:3 charge=0 multiplicity=1
B 0.000000 0.000000 0.000000
H 1.200000 0.900000 0.000000
B 3.000000 0.000000 0.000000
H 1.800000 -0.900000 0.000000
"""

XYZ_BAD_CHARGE_HEADER = """2
Properties=species:S:1:pos:R:3 charge=notanumber multiplicity=1
He 0.000000 0.000000 0.000000
H 0.772000 0.000000 0.000000
"""


def write_xyz_text(tmp_path, text, name="input.xyz"):
    """Write xyz fixture text to tmp_path/name; returns the Path."""
    path = tmp_path / name
    path.write_text(text)
    return path
