"""merge_runs.py: the cross-pod checks pin_guard cannot make.

Every test builds real files in tmp_path and calls main() through argv,
because the refusals are the product: a merge that silently pools two
worlds is the same failure class as a resume that silently spans two
backends, one step earlier in the pipeline.
"""
import json

import pytest

import merge_runs


HW_5090 = {"device": "cuda", "backend": "cuda", "torch": "2.13.0",
           "gpu": "NVIDIA GeForce RTX 5090"}
HW_5090_B = {"device": "cuda:1", "backend": "cuda", "torch": "2.13.0",
             "gpu": "NVIDIA GeForce RTX 5090"}
HW_4090 = {"device": "cuda", "backend": "cuda", "torch": "2.13.0",
           "gpu": "NVIDIA GeForce RTX 4090"}
HW_MPS = {"device": "mps", "backend": "mps", "torch": "2.13.0", "gpu": None}


def pin(hardware=HW_5090, revision="rev-a", ids=(1, 2, 3, 4), **over):
    p = {"model_revision": revision,
         "hardware": hardware,
         "hardware_history": [hardware],
         "dataset": {"rows": 500, "content_sha256": "sha-a",
                     "ids": list(ids)}}
    p.update(over)
    return p


def rec(id_, cond, **over):
    r = dict(id=id_, cond=cond, dataset="math500", seed=0, band="14-19",
             raw="\\boxed{7}", gold="7", n_tok=3, cap=128, hit_cap=False,
             device="cuda", secs=1.0, n_modified=0, calibration=False,
             difficulty=None, calib_role=None)
    r.update(over)
    return r


def write(tmp_path, name, records, pin_dict):
    path = tmp_path / name
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(str(path).replace(".jsonl", "_pin.json"), "w") as f:
        json.dump(pin_dict, f)
    return str(path)


def read_out(out):
    with open(out) as f:
        recs = [json.loads(line) for line in f]
    with open(out.replace(".jsonl", "_pin.json")) as f:
        return recs, json.load(f)


# ==================================================== the merge itself

def test_merges_disjoint_cells_and_unions_the_pin(tmp_path, capsys):
    a = write(tmp_path, "a.jsonl",
              [rec(1, "cot_intact"), rec(2, "cot_intact")], pin())
    b = write(tmp_path, "b.jsonl",
              [rec(1, "cot_random"), rec(2, "cot_random")],
              pin(hardware=HW_5090_B))
    out = str(tmp_path / "merged.jsonl")
    assert merge_runs.main([a, b, "--out", out]) == 0
    recs, p = read_out(out)
    assert {(r["id"], r["cond"]) for r in recs} == {
        (1, "cot_intact"), (2, "cot_intact"),
        (1, "cot_random"), (2, "cot_random")}
    # first input's hardware stays the origin; both appear in the history
    assert p["hardware"] == HW_5090
    assert p["hardware_history"] == [HW_5090, HW_5090_B]
    assert [m["records"] for m in p["merged_from"]] == [2, 2]
    assert "4" in capsys.readouterr().out


def test_dedupes_identical_records_and_reports_it(tmp_path, capsys):
    shared = rec(1, "cot_intact")
    a = write(tmp_path, "a.jsonl", [shared, rec(2, "cot_intact")], pin())
    b = write(tmp_path, "b.jsonl", [shared, rec(1, "cot_random")], pin())
    out = str(tmp_path / "merged.jsonl")
    merge_runs.main([a, b, "--out", out])
    recs, _ = read_out(out)
    assert len(recs) == 3
    assert "1 identical duplicate" in capsys.readouterr().out


def test_out_may_be_an_input_but_never_a_bystander(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    b = write(tmp_path, "b.jsonl", [rec(1, "cot_random")], pin())
    # in place: everything is read before anything is written
    merge_runs.main([a, b, "--out", a])
    recs, _ = read_out(a)
    assert len(recs) == 2
    other = write(tmp_path, "other.jsonl", [rec(2, "cot_intact")], pin())
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        merge_runs.main([a, b, "--out", other])


# ==================================================== cross-pin refusals

def test_refuses_a_missing_or_hardwareless_pin(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    bare = tmp_path / "b.jsonl"
    bare.write_text(json.dumps(rec(1, "cot_random")) + "\n")
    with pytest.raises(SystemExit, match="no pin file"):
        merge_runs.main([a, str(bare), "--out", str(tmp_path / "m.jsonl")])
    nohw = pin()
    del nohw["hardware"]
    b = write(tmp_path, "c.jsonl", [rec(1, "cot_random")], nohw)
    with pytest.raises(SystemExit, match="no hardware"):
        merge_runs.main([a, b, "--out", str(tmp_path / "m.jsonl")])


def test_refuses_revision_dataset_and_backend_mismatches(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    out = str(tmp_path / "m.jsonl")

    b = write(tmp_path, "rev.jsonl", [rec(1, "cot_random")],
              pin(revision="rev-B"))
    with pytest.raises(SystemExit, match="revision"):
        merge_runs.main([a, b, "--out", out])

    wrong = pin()
    wrong["dataset"]["content_sha256"] = "sha-B"
    b = write(tmp_path, "ds.jsonl", [rec(1, "cot_random")], wrong)
    with pytest.raises(SystemExit, match="content_sha256"):
        merge_runs.main([a, b, "--out", out])

    b = write(tmp_path, "mps.jsonl", [rec(1, "cot_random")],
              pin(hardware=HW_MPS))
    with pytest.raises(SystemExit, match="backends"):
        merge_runs.main([a, b, "--out", out])


def test_gpu_mix_is_refused_without_the_flag_and_recorded_with_it(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    b = write(tmp_path, "b.jsonl", [rec(1, "cot_random")],
              pin(hardware=HW_4090))
    out = str(tmp_path / "m.jsonl")
    with pytest.raises(SystemExit, match="allow-gpu-mix"):
        merge_runs.main([a, b, "--out", out])
    merge_runs.main([a, b, "--out", out, "--allow-gpu-mix"])
    _, p = read_out(out)
    assert p["hardware_history"] == [HW_5090, HW_4090]
    assert {m["gpu"] for m in p["merged_from"]} == {
        HW_5090["gpu"], HW_4090["gpu"]}


# ==================================================== record-level refusals

def test_refuses_conflicting_duplicates(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    b = write(tmp_path, "b.jsonl", [rec(1, "cot_intact", raw="\\boxed{8}")],
              pin())
    with pytest.raises(SystemExit, match="DIFFERENT record"):
        merge_runs.main([a, b, "--out", str(tmp_path / "m.jsonl")])


def test_refuses_calibration_records_and_foreign_ids(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    b = write(tmp_path, "cal.jsonl", [rec(1, "cot_random", calibration=True)],
              pin())
    with pytest.raises(SystemExit, match="calibration"):
        merge_runs.main([a, b, "--out", str(tmp_path / "m.jsonl")])
    b = write(tmp_path, "ids.jsonl", [rec(99, "cot_random")], pin())
    with pytest.raises(SystemExit, match="outside the pinned sample"):
        merge_runs.main([a, b, "--out", str(tmp_path / "m.jsonl")])


def test_refuses_mixed_seed_band_or_dataset(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec(1, "cot_intact")], pin())
    b = write(tmp_path, "band.jsonl", [rec(1, "cot_random", band="14-33")],
              pin())
    with pytest.raises(SystemExit, match="disagree on band"):
        merge_runs.main([a, b, "--out", str(tmp_path / "m.jsonl")])


# ==================================================== the file run.py resumes

def test_merged_file_scans_like_a_run_file(tmp_path):
    """The consumer contract: run.py's resume scan keys on (id, cond) and
    its gate reads matched pairs from one file. A merged file must present
    exactly the union, once each."""
    a = write(tmp_path, "a.jsonl",
              [rec(i, "cot_intact") for i in (1, 2, 3)], pin())
    b = write(tmp_path, "b.jsonl",
              [rec(i, "cot_random") for i in (1, 2, 3)], pin())
    out = str(tmp_path / "m.jsonl")
    merge_runs.main([a, b, "--out", out])
    done = set()
    with open(out) as f:
        for line in f:
            r = json.loads(line)
            key = (r["id"], r["cond"])
            assert key not in done
            assert not r.get("calibration")
            done.add(key)
    assert len(done) == 6
