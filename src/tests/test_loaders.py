"""Tests for loaders.py -- WHICH weights, and WHERE they run.

No weights and no download: everything here is about the two facts a run has
to be able to state afterwards. Both used to be observations rather than
requests -- the checkpoint was read back off whatever the Hub happened to
serve, and the device was whatever the machine offered -- and both are the kind
of thing that lands in a difference of differences with no other signature in
the data.

Run: python -m tests.test_loaders
"""
import glob
import json
import os
import sys

import loaders


# ============================================================ the checkpoint

def test_the_pinned_revision_is_a_full_commit_hash():
    """A tag or a branch name would defeat the point: those move, and then the
    pin records an intention rather than a snapshot."""
    rev = loaders.MODEL_REVISION
    assert isinstance(rev, str) and len(rev) == 40, rev
    assert all(c in "0123456789abcdef" for c in rev), rev


def test_the_pin_matches_every_run_already_on_disk():
    """The tripwire that makes the constant mean something. runs/*_pin.json
    records the checkpoint each committed file was generated against; if
    MODEL_REVISION moves without those being regenerated, the repo is claiming
    weights that produced none of its data.

    A deliberate bump is meant to fail here until the data is regenerated or
    moved to runs/archive/, which is exempt precisely because it is historical.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pins = sorted(glob.glob(os.path.join(here, "..", "runs", "*_pin.json")))
    checked = 0
    for p in pins:
        with open(p) as f:
            rev = json.load(f).get("model_revision")
        if not rev:
            continue
        checked += 1
        assert rev == loaders.MODEL_REVISION, (
            f"{os.path.basename(p)} was generated at {rev} but "
            f"loaders.MODEL_REVISION is now {loaders.MODEL_REVISION}. Either "
            f"the bump is unintended, or that run is a different experiment "
            f"and belongs in runs/archive/.")
    assert checked, "no pinned runs found to check against"


# ================================================================= the device

def test_an_unknown_device_is_refused_by_name():
    """Before the model loads, not after: `.to(device)` is reached once 8GB of
    weights are already materialised, so a typo used to cost a full load.

    SystemExit and not ValueError, so every entry point prints the message
    instead of a traceback without seven copies of the same try/except."""
    for bad in ("gpu", "cuda1", "metal", "cuda:x"):
        try:
            loaders.pick_device(bad)
        except SystemExit as e:
            assert bad in str(e), e
            continue
        raise AssertionError(f"--device {bad!r} must be refused")


def test_an_unavailable_backend_is_refused_here_too():
    """Spelling is not the only failure. `--device cuda` on a Mac is a valid
    name for a backend this machine does not have, and it fails identically
    late if it is not caught identically early."""
    import torch
    absent = ("cuda" if not torch.cuda.is_available() else None)
    if absent is None:
        absent = ("mps" if not (getattr(torch.backends, "mps", None)
                                and torch.backends.mps.is_available())
                  else None)
    if absent is None:                       # a box with both: nothing to test
        return
    try:
        loaders.pick_device(absent)
    except SystemExit as e:
        assert "no" in str(e).lower(), e
        return
    raise AssertionError(f"--device {absent} must be refused on this machine")


def test_cpu_is_always_available_and_always_wins_when_asked():
    assert loaders.pick_device("cpu") == "cpu"
    assert loaders.pick_device("CPU ") == "cpu", "normalised, not rejected"
    # An explicit device beats the tiny default, so a weightless smoke test can
    # still be pointed at the accelerator it will really run on.
    assert loaders.pick_device("cpu", tiny=True) == "cpu"
    assert loaders.pick_device(None, tiny=True) == "cpu"


def test_backend_of_ignores_the_card_index():
    """What determines the numerics is the backend. Two cards of the same
    architecture are not the confound MPS-vs-CUDA is, and a guard that refused
    them would cost more than it buys."""
    assert loaders.backend_of("cuda:3") == "cuda"
    assert loaders.backend_of("cuda") == "cuda"
    assert loaders.backend_of("mps") == "mps"


def test_hardware_reports_where_the_run_went_not_where_it_could_have():
    """hardware() used to detect a device independently of pick_device, so
    `--device cpu` on a CUDA box recorded "cuda" in the manifest. One question
    with two answers is the failure this module exists to prevent."""
    hw = loaders.hardware("cpu")
    assert hw["device"] == "cpu" and hw["backend"] == "cpu", hw
    assert hw["gpu"] is None, hw
    assert hw["torch"], hw


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
