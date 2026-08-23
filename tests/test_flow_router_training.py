from pathlib import Path

from training.train_flow_router import canonicalize_gguf_export


def test_canonicalize_gguf_export_moves_q4_k_m_artifact(tmp_path: Path):
    generated = tmp_path / "gemma-3-1b-it.Q4_K_M.gguf"
    generated.write_bytes(b"gguf")
    canonical = tmp_path / "flow-router-gemma3-1b.Q4_K_M.gguf"

    result = canonicalize_gguf_export(
        {"gguf_files": [str(generated)]},
        canonical,
    )

    assert result == canonical
    assert canonical.read_bytes() == b"gguf"
    assert not generated.exists()
