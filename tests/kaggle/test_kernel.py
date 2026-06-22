from hipe.kaggle.kernel import render_kernel_script, PINNED_COMMIT


def test_kernel_script_has_required_steps():
    s = render_kernel_script("configs/xlmr.yaml")
    # installs the package from the input dataset with the ml extra
    assert "/kaggle/input/hipe-code" in s
    assert "pip" in s and "[ml]" in s
    # clones the pinned HIPE data
    assert "HIPE-2026-data" in s
    assert PINNED_COMMIT in s
    # runs the harness with outputs going to the kernel working dir
    assert "HIPE_RUNS_DIR" in s
    assert "/kaggle/working/runs" in s
    assert "configs/xlmr.yaml" in s
    # it is valid Python
    compile(s, "<kernel>", "exec")
