"""Linetype / lineweight classification tests.

These cover the failure this project was built to fix: a model that "only
computes coordinates" cannot tell a solid line from a dashed one, so the
classification is done from measured ink runs instead.
"""

from __future__ import annotations

from autocad_mcp.trace import linetype


def pattern(runs: list[tuple[bool, int]]) -> list[bool]:
    """Build an occupancy list from (is_ink, length) runs."""
    out: list[bool] = []
    for is_ink, length in runs:
        out.extend([is_ink] * length)
    return out


def test_solid_line_is_continuous():
    verdict = linetype.classify_linetype(pattern([(True, 200)]))
    assert verdict.linetype == "CONTINUOUS"
    assert verdict.confidence > 0.8


def test_sub_pixel_gaps_do_not_make_a_line_dashed():
    """Anti-aliasing noise must not turn a solid line into a dashed one."""
    occupancy = pattern([(True, 60), (False, 1), (True, 60), (False, 1), (True, 60)])
    verdict = linetype.classify_linetype(occupancy, min_run_px=2)
    assert verdict.linetype == "CONTINUOUS"


def test_uniform_dashes_at_large_period_are_dashed():
    occupancy = pattern([(True, 40), (False, 20)] * 6)
    verdict = linetype.classify_linetype(occupancy, min_run_px=2, scale=1.0)
    assert verdict.linetype == "DASHED"
    assert verdict.dashes >= 5
    assert 0.5 < verdict.duty < 0.75


def test_same_shape_at_small_period_is_hidden():
    """DASHED and HIDDEN share a 2:1 ratio; the period decides which one it is."""
    occupancy = pattern([(True, 8), (False, 4)] * 8)
    verdict = linetype.classify_linetype(occupancy, min_run_px=2, scale=1.0)
    assert verdict.linetype == "HIDDEN"


def test_dashed_stays_dashed_when_the_scale_is_small():
    """The same pixel pattern at a smaller drawing scale is a longer period."""
    occupancy = pattern([(True, 40), (False, 20)] * 6)
    assert linetype.classify_linetype(occupancy, scale=1.0).linetype == "DASHED"
    assert linetype.classify_linetype(occupancy, scale=0.1).linetype == "HIDDEN"


def test_alternating_long_short_is_center():
    occupancy = pattern([(True, 60), (False, 10), (True, 12), (False, 10)] * 4)
    verdict = linetype.classify_linetype(occupancy, min_run_px=2)
    assert verdict.linetype == "CENTER"
    assert verdict.pattern in ("Ls", "sL")


def test_long_short_short_is_phantom():
    occupancy = pattern([(True, 60), (False, 10), (True, 12), (False, 10), (True, 12), (False, 10)] * 3)
    verdict = linetype.classify_linetype(occupancy, min_run_px=2)
    assert verdict.linetype == "PHANTOM"


def test_run_lengths_trims_outer_gaps():
    occupancy = pattern([(False, 5), (True, 10), (False, 4), (True, 10), (False, 7)])
    dashes, gaps = linetype.run_lengths(occupancy)
    assert dashes == [10.0, 10.0]
    assert gaps == [4.0]


def test_empty_pattern_is_not_a_crash():
    verdict = linetype.classify_linetype([])
    assert verdict.linetype == "CONTINUOUS"
    assert verdict.confidence == 0.0


def test_classify_lineweight_snaps_to_the_iso_ladder():
    assert linetype.snap_lineweight(0.13) == 13
    assert linetype.snap_lineweight(0.26) == 25
    assert linetype.snap_lineweight(5.0) == 200

    lineweight, width, is_thick = linetype.classify_lineweight(4.0, scale=0.25)
    assert width == 1.0
    assert lineweight == 100
    assert is_thick


def test_layer_convention_mapping():
    assert linetype.layer_for("CONTINUOUS", True)[:2] == ("Thick", "outline")
    assert linetype.layer_for("CONTINUOUS", False)[:2] == ("Thin", "thin")
    assert linetype.layer_for("HIDDEN", True)[:2] == ("Hidden", "hidden")
    assert linetype.layer_for("CENTER", False)[:2] == ("Center", "center")
    assert linetype.layer_for("PHANTOM", False)[2] == 4  # cyan, as drafters draw centre lines


def test_linetype_patterns_cover_the_classifier_output():
    for name in ("CONTINUOUS", "DASHED", "HIDDEN", "CENTER", "PHANTOM"):
        assert name in linetype.LINETYPE_PATTERNS
