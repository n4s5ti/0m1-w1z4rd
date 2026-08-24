import __future__

import builtins
import hashlib
import importlib
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def _trainer_module():
    return importlib.import_module("training.train_curriculum")


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_valid_curriculum_splits(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Explain cargo build.",
                "input": "",
                "output": "`cargo build` compiles the current package.\n",
                "category": "cargo",
            },
            {
                "instruction": "Show git status.",
                "input": "",
                "output": "`git status --short` summarizes tracked changes.\n",
                "category": "git",
            },
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "instruction": "Describe rg searching.",
                "input": "",
                "output": "`rg pattern` searches repository text.\n",
                "category": "ripgrep",
            }
        ],
    )
    return train_path, test_path


def test_resolve_training_paths_scopes_default_artifacts_to_selected_output_dir_and_honors_explicit_canonical_path(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    output_dir = tmp_path / "personal-multi-cli-curriculum"

    default_paths = trainer.resolve_training_paths(output_dir)

    assert isinstance(default_paths, trainer.TrainingPaths)
    assert default_paths.adapter_dir == output_dir / "adapter"
    assert default_paths.merged_dir == output_dir / "merged"
    assert default_paths.canonical_gguf == output_dir / "wiz4rd-personal.Q4_K_M.gguf"
    assert default_paths.canonical_gguf != Path("models/flow-router-gemma3-1b.Q4_K_M.gguf")

    explicit_canonical_gguf = tmp_path / "approved-models" / "personal.Q4_K_M.gguf"

    explicit_paths = trainer.resolve_training_paths(
        output_dir,
        canonical_gguf=explicit_canonical_gguf,
    )

    assert explicit_paths.canonical_gguf == explicit_canonical_gguf


def test_prepare_training_records_exact_input_counts_digests_and_isolated_personal_model_paths(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    train_path, test_path = _write_valid_curriculum_splits(tmp_path)
    output_dir = tmp_path / "personal-multi-cli-curriculum"

    plan = trainer.prepare_training(train_path, test_path, output_dir)

    assert isinstance(plan, trainer.CurriculumTrainingPlan)
    assert plan.train_rows == 2
    assert plan.test_rows == 1
    assert plan.train_sha256 == "08d89a3fbd40c66698497fc0a8ef15e46105beae452be2d46269e27879f66902"
    assert plan.test_sha256 == "d836693992e537055aa194ba0419fd643957a965285d48cc08d1a32b056c600d"
    assert plan.train_sha256 == hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert plan.test_sha256 == hashlib.sha256(test_path.read_bytes()).hexdigest()
    assert plan.paths == trainer.TrainingPaths(
        adapter_dir=output_dir / "adapter",
        merged_dir=output_dir / "merged",
        canonical_gguf=output_dir / "wiz4rd-personal.Q4_K_M.gguf",
    )
    assert plan.paths.canonical_gguf != Path("models/flow-router-gemma3-1b.Q4_K_M.gguf")
    with pytest.raises(FrozenInstanceError):
        plan.train_rows = 999  # type: ignore[misc]


def test_prepare_training_honors_an_explicit_personal_gguf_path(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    train_path, test_path = _write_valid_curriculum_splits(tmp_path)
    explicit_canonical_gguf = tmp_path / "approved-models" / "personal.Q4_K_M.gguf"

    plan = trainer.prepare_training(
        train_path,
        test_path,
        tmp_path / "output",
        canonical_gguf=explicit_canonical_gguf,
    )

    assert plan.paths.canonical_gguf == explicit_canonical_gguf


def test_prepare_training_rejects_an_empty_train_split(tmp_path: Path) -> None:
    trainer = _trainer_module()
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text("", encoding="utf-8")
    _write_jsonl(
        test_path,
        [
            {
                "instruction": "Describe rg searching.",
                "input": "",
                "output": "`rg pattern` searches repository text.\n",
                "category": "ripgrep",
            }
        ],
    )

    with pytest.raises(ValueError):
        trainer.prepare_training(train_path, test_path, tmp_path / "output")


def test_prepare_training_rejects_an_empty_test_split(tmp_path: Path) -> None:
    trainer = _trainer_module()
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Explain cargo build.",
                "input": "",
                "output": "`cargo build` compiles the current package.\n",
                "category": "cargo",
            }
        ],
    )
    test_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        trainer.prepare_training(train_path, test_path, tmp_path / "output")


def test_prepare_training_rejects_a_prompt_shared_by_train_and_test(tmp_path: Path) -> None:
    trainer = _trainer_module()
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Explain cargo build.",
                "input": "",
                "output": "`cargo build` compiles the current package.\n",
                "category": "cargo",
            }
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "instruction": "Explain cargo build.",
                "input": "",
                "output": "`cargo build --release` compiles optimized artifacts.\n",
                "category": "cargo",
            }
        ],
    )

    with pytest.raises(ValueError):
        trainer.prepare_training(train_path, test_path, tmp_path / "output")


def test_prepare_training_rejects_a_target_command_shared_by_train_and_test(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Compile the package.",
                "input": "",
                "output": "COMMAND: cargo build\n",
                "category": "cargo",
            }
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "instruction": "Use cargo to compile the package.",
                "input": "",
                "output": "COMMAND: cargo build\n",
                "category": "cargo",
            }
        ],
    )

    with pytest.raises(ValueError, match="target command"):
        trainer.prepare_training(train_path, test_path, tmp_path / "output")


def test_prepare_training_import_and_preflight_do_not_require_unsloth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_path, test_path = _write_valid_curriculum_splits(tmp_path)
    original_import = builtins.__import__

    def reject_unsloth(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "unsloth" or name.startswith("unsloth."):
            raise AssertionError("curriculum preflight must not import Unsloth")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_unsloth)
    sys.modules.pop("training.train_curriculum", None)

    trainer = importlib.import_module("training.train_curriculum")
    plan = trainer.prepare_training(train_path, test_path, tmp_path / "output")

    assert plan.train_rows == 2


def test_module_defers_pep_604_annotations_for_python_3_8_compatibility() -> None:
    trainer = _trainer_module()

    assert trainer.resolve_training_paths.__code__.co_flags & __future__.annotations.compiler_flag
    assert trainer.resolve_training_paths.__annotations__["output_dir"] == "str | Path"
    assert trainer.resolve_training_paths.__annotations__["canonical_gguf"] == "str | Path | None"


def test_exact_match_accuracy_ignores_harmless_whitespace_and_command_prefix() -> None:
    trainer = _trainer_module()

    accuracy = trainer.exact_match_accuracy(
        [" \tCOMMAND:  git   status --short\n", "COMMAND:\tcargo build"],
        ["git status --short", "  cargo build  "],
    )

    assert accuracy == 1.0


def test_exact_match_accuracy_does_not_change_command_tokens() -> None:
    trainer = _trainer_module()

    accuracy = trainer.exact_match_accuracy(
        ["COMMAND: git status --short"],
        ["git status"],
    )

    assert accuracy == 0.0


def test_canonicalize_gguf_export_moves_one_quantized_artifact_to_personal_path(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    candidate = merged_dir / "adapter.Q4_K_M.gguf"
    candidate.write_bytes(b"gguf")
    canonical_path = tmp_path / "personal-output" / "wiz4rd-personal.Q4_K_M.gguf"

    exported_path = trainer.canonicalize_gguf_export(merged_dir, canonical_path)

    assert exported_path == canonical_path
    assert canonical_path.read_bytes() == b"gguf"
    assert not candidate.exists()


def test_canonicalize_gguf_export_rejects_missing_quantized_artifact(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    canonical_path = tmp_path / "personal-output" / "wiz4rd-personal.Q4_K_M.gguf"

    with pytest.raises(ValueError):
        trainer.canonicalize_gguf_export(merged_dir, canonical_path)

    assert not canonical_path.exists()


def test_canonicalize_gguf_export_rejects_multiple_quantized_artifacts(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    first_candidate = merged_dir / "first.Q4_K_M.gguf"
    second_candidate = merged_dir / "second.Q4_K_M.gguf"
    first_candidate.write_bytes(b"first")
    second_candidate.write_bytes(b"second")
    canonical_path = tmp_path / "personal-output" / "wiz4rd-personal.Q4_K_M.gguf"

    with pytest.raises(ValueError):
        trainer.canonicalize_gguf_export(merged_dir, canonical_path)

    assert first_candidate.exists()
    assert second_candidate.exists()
    assert not canonical_path.exists()


def test_canonicalize_gguf_export_rejects_the_flow_router_canonical_artifact(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    candidate = merged_dir / "adapter.Q4_K_M.gguf"
    candidate.write_bytes(b"gguf")
    flow_router_canonical_path = tmp_path / "models" / "flow-router-gemma3-1b.Q4_K_M.gguf"

    with pytest.raises(ValueError):
        trainer.canonicalize_gguf_export(merged_dir, flow_router_canonical_path)

    assert candidate.exists()
    assert not flow_router_canonical_path.exists()


def test_prepare_model_for_inference_uses_unsloth_transition_before_eval() -> None:
    trainer = _trainer_module()
    calls: list[tuple[str, object]] = []

    class FakeModel:
        def eval(self) -> None:
            calls.append(("eval", self))

    class FakeFastModel:
        @staticmethod
        def for_inference(model: object) -> None:
            calls.append(("for_inference", model))

    model = FakeModel()

    trainer._prepare_model_for_inference(FakeFastModel, model)

    assert calls == [("for_inference", model), ("eval", model)]


def test_parse_args_accepts_a_saved_adapter_for_evaluation_and_export(tmp_path: Path) -> None:
    trainer = _trainer_module()
    adapter_path = tmp_path / "adapter"

    args = trainer.parse_args(["--adapter-path", str(adapter_path)])

    assert args.adapter_path == adapter_path


def test_main_evaluates_a_saved_adapter_without_export_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    adapter_path = tmp_path / "adapter"
    captured: dict[str, object] = {}

    def fake_export_adapter(
        selected_adapter: Path,
        test_path: Path,
        output_dir: Path,
        canonical_gguf: object,
        *,
        evaluate_only: bool,
    ) -> None:
        captured.update(
            {
                "adapter_path": selected_adapter,
                "test_path": test_path,
                "output_dir": output_dir,
                "canonical_gguf": canonical_gguf,
                "evaluate_only": evaluate_only,
            }
        )

    monkeypatch.setattr(trainer, "export_adapter", fake_export_adapter)

    trainer.main(
        [
            "--adapter-path",
            str(adapter_path),
            "--test",
            str(tmp_path / "test.jsonl"),
            "--output",
            str(tmp_path / "output"),
            "--evaluate-only",
        ]
    )

    assert captured == {
        "adapter_path": adapter_path,
        "test_path": tmp_path / "test.jsonl",
        "output_dir": tmp_path / "output",
        "canonical_gguf": None,
        "evaluate_only": True,
    }


def test_export_adapter_evaluate_only_does_not_write_a_gguf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(
        test_path,
        [
            {
                "instruction": "Translate to personal command: Search shell history",
                "input": "",
                "output": "COMMAND: atuin search\n",
                "category": "atuin",
            }
        ],
    )

    class FakeModel:
        def eval(self) -> None:
            pass

    class FakeFastModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return FakeModel(), object()

        @staticmethod
        def for_inference(model: object) -> None:
            pass

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    class FakeUnsloth:
        FastModel = FakeFastModel

    evaluated: list[object] = []
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "unsloth", FakeUnsloth())
    monkeypatch.setattr(
        trainer,
        "_evaluate",
        lambda model, tokenizer, examples: evaluated.extend(examples) or 1.0,
    )
    monkeypatch.setattr(
        trainer,
        "_export",
        lambda *args, **kwargs: pytest.fail("evaluate-only mode attempted GGUF export"),
    )

    result = trainer.export_adapter(
        adapter_path,
        test_path,
        tmp_path / "output",
        evaluate_only=True,
    )

    assert result is None
    assert len(evaluated) == 1


def test_configure_unsloth_runtime_disables_compiled_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer_module()
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)

    trainer._configure_unsloth_runtime()

    assert trainer.os.environ["UNSLOTH_COMPILE_DISABLE"] == "1"


def test_canonicalize_gguf_export_accepts_unsloth_sibling_export_directory(
    tmp_path: Path,
) -> None:
    trainer = _trainer_module()
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    export_dir = tmp_path / "merged_gguf"
    export_dir.mkdir()
    candidate = export_dir / "gemma-3-1b-it.Q4_K_M.gguf"
    candidate.write_bytes(b"gguf")
    canonical_path = tmp_path / "personal-output" / "wiz4rd-personal.Q4_K_M.gguf"

    exported_path = trainer.canonicalize_gguf_export(merged_dir, canonical_path)

    assert exported_path == canonical_path
    assert canonical_path.read_bytes() == b"gguf"
    assert not candidate.exists()
