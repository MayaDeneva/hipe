from hipe.kaggle.kernel import render_kernel_script, PINNED_COMMIT


def test_kernel_script_has_required_steps():
    s = render_kernel_script("configs/xlmr.yaml")
    # extracts the bundle before installing
    assert "bundle.zip" in s
    assert "extractall" in s
    # installs from the extracted writable CODE dir using -m pip
    assert "-m" in s
    assert "[ml]" in s
    # clones the pinned HIPE data
    assert "HIPE-2026-data" in s
    assert PINNED_COMMIT in s
    # runs the harness via python -m hipe.cli
    assert "hipe.cli" in s
    # outputs go to the kernel working dir
    assert "HIPE_RUNS_DIR" in s
    assert "/kaggle/working/runs" in s
    assert "configs/xlmr.yaml" in s
    # it is valid Python
    compile(s, "<kernel>", "exec")
