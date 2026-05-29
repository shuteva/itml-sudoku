"""
test_solvers.py – Smoke test (no dataset required)
Run:  python test_solvers.py
"""

from sudoku_solver import CSPSolver, ANNSolver, HybridSolver, is_correct

# 10 hand-verified puzzle / solution pairs sampled from the CSV format
SAMPLES = [
    ("070000043040009610800634900094052000358460020000800530080070091902100005007040802",
     "679518243543729618821634957794352186358461729216897534485276391962183475137945862"),
    ("301086504046521070500000001400800002080347900009050038004090200008734090007208103",
     "371986524846521379592473861463819752285347916719652438634195287128734695957268143"),
    ("048301560360008090910670003020000935509010200670020010004002107090100008150834029",
     "748391562365248791912675483421786935589413276673529814834962157296157348157834629"),
]


def test_csp():
    solver = CSPSolver()
    passed = 0
    for puzzle, solution in SAMPLES:
        result, rt = solver.solve(puzzle)
        ok = is_correct(result, solution)
        status = "PASS" if ok else "FAIL"
        print(f"  CSP [{status}]  {rt*1000:.1f} ms")
        if ok:
            passed += 1
    print(f"  CSP: {passed}/{len(SAMPLES)} correct\n")
    return passed == len(SAMPLES)


def test_ann_fallback():
    """Test ANN rule-based fallback (no TF needed)."""
    solver = ANNSolver()      # TF absent → rule-based
    passed = 0
    for puzzle, solution in SAMPLES:
        result, rt = solver.solve(puzzle)
        # Rule-based only fills naked singles; accuracy < 100% is expected
        print(f"  ANN-fallback  {rt*1000:.2f} ms  "
              f"{'correct' if is_correct(result, solution) else 'partial'}")
    print()


def test_hybrid():
    csp    = CSPSolver()
    ann    = ANNSolver()
    solver = HybridSolver(ann, csp)
    passed = 0
    for puzzle, solution in SAMPLES:
        result, rt = solver.solve(puzzle)
        ok = is_correct(result, solution)
        status = "PASS" if ok else "FAIL"
        print(f"  Hybrid [{status}]  {rt*1000:.1f} ms")
        if ok:
            passed += 1
    print(f"  Hybrid: {passed}/{len(SAMPLES)} correct\n")
    return passed == len(SAMPLES)


if __name__ == "__main__":
    print("=" * 50)
    print("Smoke Tests – Mia Shuteva 89231346")
    print("=" * 50)
    csp_ok    = test_csp()
    test_ann_fallback()
    hybrid_ok = test_hybrid()

    if csp_ok and hybrid_ok:
        print("All deterministic tests passed ✓")
    else:
        print("Some tests failed – check output above.")
