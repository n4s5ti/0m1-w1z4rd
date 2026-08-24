from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eval.contamination import audit, load_jsonl

DEFAULT_MODEL = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
DEFAULT_TRAIN = Path("data/personal_curriculum_train.jsonl")
DEFAULT_TEST = Path("data/personal_curriculum_test.jsonl")
DEFAULT_OUTPUT = Path("models/wiz4rd-personal")
DEFAULT_EPOCHS = 20.0
SEED = 42


@dataclass(frozen=True)
class TrainingPaths:
    adapter_dir: Path
    merged_dir: Path
    canonical_gguf: Path


@dataclass(frozen=True)
class CurriculumTrainingPlan:
    train_rows: int
    test_rows: int
    train_sha256: str
    test_sha256: str
    paths: TrainingPaths


def resolve_training_paths(
    output_dir: str | Path,
    canonical_gguf: str | Path | None = None,
) -> TrainingPaths:
    output_dir = Path(output_dir)
    return TrainingPaths(
        adapter_dir=output_dir / "adapter",
        merged_dir=output_dir / "merged",
        canonical_gguf=(
            Path(canonical_gguf)
            if canonical_gguf is not None
            else output_dir / "wiz4rd-personal.Q4_K_M.gguf"
        ),
    )


def prepare_training(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    canonical_gguf: str | Path | None = None,
) -> CurriculumTrainingPlan:
    train_path = Path(train_path)
    test_path = Path(test_path)
    train = load_jsonl(train_path)
    test = load_jsonl(test_path)

    if not train or not test:
        raise ValueError("training and test splits must both contain examples")

    if not audit(train, test).is_clean:
        raise ValueError("training and test splits have prompt contamination")

    train_commands = {_normalized_command(example.command) for example in train}
    test_commands = {_normalized_command(example.command) for example in test}
    if train_commands & test_commands:
        raise ValueError("training and test splits have target command contamination")

    return CurriculumTrainingPlan(
        train_rows=len(train),
        test_rows=len(test),
        train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
        paths=resolve_training_paths(output_dir, canonical_gguf),
    )


def _normalized_command(command: str) -> str:
    command = command.strip()
    if command.startswith("COMMAND:"):
        command = command[len("COMMAND:") :]
    return " ".join(command.split())


def exact_match_accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if len(predictions) != len(expected):
        raise ValueError("predictions and expected values must have the same length")
    if not expected:
        return 0.0
    matches = sum(
        _normalized_command(prediction) == _normalized_command(answer)
        for prediction, answer in zip(predictions, expected)
    )
    return matches / len(expected)


def canonicalize_gguf_export(merged_dir: str | Path, canonical_path: str | Path) -> Path:
    merged_dir = Path(merged_dir)
    canonical_path = Path(canonical_path)
    if "flow-router" in canonical_path.name.lower():
        raise ValueError("refusing to overwrite a flow-router canonical artifact")
    export_roots = [
        merged_dir,
        merged_dir.with_name(f"{merged_dir.name}_gguf"),
    ]
    candidates = [
        candidate
        for export_root in export_roots
        for candidate in export_root.rglob("*.Q4_K_M.gguf")
        if candidate.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one Q4_K_M GGUF export for {merged_dir}, " f"found {len(candidates)}"
        )

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = candidates[0]
    if candidate.resolve() != canonical_path.resolve():
        candidate.replace(canonical_path)
    return canonical_path


def _response_mask_parts(rendered_row: str) -> tuple[str, str]:
    if "<start_of_turn>" in rendered_row:
        return "<start_of_turn>user\n", "<start_of_turn>model\n"
    if "<|im_start|>" in rendered_row:
        return "<|im_start|>user\n", "<|im_start|>assistant\n"
    raise RuntimeError(f"unrecognized chat template: {rendered_row!r}")


def _generate_command(model, tokenizer, instruction: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    prompt_tokens = inputs["input_ids"].shape[1]
    return tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True)


def _configure_unsloth_runtime() -> None:
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


def _prepare_model_for_inference(fast_model, model) -> None:
    fast_model.for_inference(model)
    model.eval()


def _evaluate(model, tokenizer, test_examples) -> float:
    predictions = [
        _generate_command(model, tokenizer, example.instruction) for example in test_examples
    ]
    accuracy = exact_match_accuracy(predictions, [example.command for example in test_examples])
    print(f"held-out exact accuracy: {accuracy:.4f}")
    return accuracy


def _export(model, tokenizer, paths: TrainingPaths) -> Path:
    paths.merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(paths.merged_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    exported_path = canonicalize_gguf_export(paths.merged_dir, paths.canonical_gguf)
    print(f"GGUF export: {exported_path}")
    return exported_path


def export_adapter(
    adapter_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    canonical_gguf: str | Path | None = None,
    *,
    evaluate_only: bool = False,
) -> Path | None:
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_path}")

    test_examples = load_jsonl(Path(test_path))
    if not test_examples:
        raise ValueError("test split must contain examples")
    paths = resolve_training_paths(output_dir, canonical_gguf)

    _configure_unsloth_runtime()
    import torch
    from unsloth import FastModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local personal model export")

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    _prepare_model_for_inference(FastModel, model)
    _evaluate(model, tokenizer, test_examples)
    if evaluate_only:
        return None
    return _export(model, tokenizer, paths)


def train(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    canonical_gguf: str | Path | None = None,
    base_model: str = DEFAULT_MODEL,
    epochs: float = DEFAULT_EPOCHS,
) -> Path:
    plan = prepare_training(train_path, test_path, output_dir, canonical_gguf)
    train_examples = load_jsonl(Path(train_path))
    test_examples = load_jsonl(Path(test_path))
    _configure_unsloth_runtime()

    import torch
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

    # Dynamic imports preserve Unsloth's required patch-before-import order.
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local personal training")

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    rows = [
        {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example.instruction},
                    {"role": "assistant", "content": f"COMMAND: {example.command}"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
        for example in train_examples
    ]
    trainer = trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets.Dataset.from_list(rows),
        args=trl.SFTConfig(
            output_dir=str(plan.paths.adapter_dir / "trainer"),
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_strategy="no",
            max_length=512,
            dataset_text_field="text",
            packing=False,
            seed=SEED,
            report_to="none",
        ),
    )
    instruction_part, response_part = _response_mask_parts(rows[0]["text"])
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )
    supervised_tokens = [token for token in trainer.train_dataset[0]["labels"] if token != -100]
    if "COMMAND:" not in tokenizer.decode(supervised_tokens):
        raise RuntimeError("response mask excluded the command")

    trainer.train()
    model.save_pretrained(plan.paths.adapter_dir)
    tokenizer.save_pretrained(plan.paths.adapter_dir)
    _prepare_model_for_inference(FastModel, model)

    _evaluate(model, tokenizer, test_examples)
    return _export(model, tokenizer, plan.paths)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a personal command curriculum")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-gguf", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        plan = prepare_training(args.train, args.test, args.output, args.canonical_gguf)
        print(f"validated {plan.train_rows} training and {plan.test_rows} held-out rows")
        return
    if args.evaluate_only and args.adapter_path is None:
        raise ValueError("--evaluate-only requires --adapter-path")
    if args.adapter_path is not None:
        export_adapter(
            args.adapter_path,
            args.test,
            args.output,
            args.canonical_gguf,
            evaluate_only=args.evaluate_only,
        )
        return
    train(
        args.train,
        args.test,
        args.output,
        args.canonical_gguf,
        args.base_model,
        args.epochs,
    )


if __name__ == "__main__":
    main()
