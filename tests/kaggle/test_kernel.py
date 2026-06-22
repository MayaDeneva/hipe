from hipe.kaggle.kernel import render_kernel_script, PINNED_COMMIT


def test_kernel_script_has_required_steps():
    s = render_kernel_script("https://github.com/u/r.git", "configs/xlmr.yaml")
    # clones the project repo
    assert "https://github.com/u/r.git" in s
    assert "git" in s
    assert "clone" in s
    # clones the pinned HIPE data
    assert "HIPE-2026-data" in s
    assert PINNED_COMMIT in s
    # installs from the cloned CODE dir using -m pip
    assert "-m" in s
    # runs the harness via python -m hipe.cli
    assert "hipe.cli" in s
    # outputs go to the kernel working dir
    assert "HIPE_RUNS_DIR" in s
    assert "/kaggle/working/runs" in s
    assert "configs/xlmr.yaml" in s
    # it is valid Python
    compile(s, "<kernel>", "exec")


def test_kernel_pins_torch_to_avoid_gpu_mismatch():
    from hipe.kaggle.kernel import render_kernel_script
    s = render_kernel_script("https://github.com/u/r.git", "configs/xlmr.yaml")
    assert "constraints.txt" in s
    assert "torch==" in s            # pin Kaggle's pre-installed torch
    assert "-c" in s                 # pip install with the constraints file
    compile(s, "<kernel>", "exec")
