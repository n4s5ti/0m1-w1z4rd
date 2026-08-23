from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from eval.contamination import audit, load_jsonl, self_audit

DEFAULT_MODEL = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
DEFAULT_TRAIN = Path("data/flows_training.jsonl")
DEFAULT_TEST = Path("data/flows_test_handwritten.jsonl")
DEFAULT_OUTPUT = Path("models/flow-router-gemma3-1b-q4km")
CANONICAL_GGUF = Path("models/flow-router-gemma3-1b.Q4_K_M.gguf")


def canonicalize_gguf_export(export: Mapping[str, object], canonical_path: Path) -> Path:
    gguf_files = export.get("gguf_files")
    if not isinstance(gguf_files, list):
        raise RuntimeError("Unsloth GGUF export did not report generated files")

    candidates = [
        Path(path)
        for path in gguf_files
        if isinstance(path, (str, Path)) and Path(path).name.endswith(".Q4_K_M.gguf")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Q4_K_M GGUF export, found {candidates}")

    generated = candidates[0]
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != canonical_path.resolve():
        generated.replace(canonical_path)
    return canonical_path


def validate_inputs(train_path: Path, test_path: Path):
    train_examples = load_jsonl(train_path)
    test_examples = load_jsonl(test_path)
    report = audit(train_examples, test_examples)
    if not report.is_clean:
        raise RuntimeError(f"training/test contamination detected:\n{report.format()}")

    commands = sorted({example.command for example in train_examples})
    print(report.format())
    print(f"training rows: {len(train_examples)}")
    print(f"held-out rows: {len(test_examples)}")
    print(f"commands: {commands}")
    print(f"self-audit: {self_audit(train_examples)}")
    return train_examples


def train(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    base_model: str,
    epochs: float,
) -> None:
    train_examples = validate_inputs(train_path, test_path)

    from unsloth import FastModel  # noqa: I001 - Unsloth must load before TRL.
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local flow-router training")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("VRAM: %.1f GiB" % (torch.cuda.get_device_properties(0).total_memory / (1024**3)))

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
        random_state=42,
    )

    rows = []
    for example in train_examples:
        messages = [
            {"role": "user", "content": example.instruction},
            {"role": "assistant", "content": f"COMMAND: {example.command}"},
        ]
        rows.append(
            {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }
        )
    train_dataset = Dataset.from_list(rows)

    probe = rows[0]["text"]
    if "<start_of_turn>" in probe:
        instruction_part = "<start_of_turn>user\n"
        response_part = "<start_of_turn>model\n"
    elif "<|im_start|>" in probe:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    else:
        raise RuntimeError(f"unrecognized chat template: {probe!r}")

    bf16 = torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir="outputs/flow-router",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        fp16=not bf16,
        bf16=bf16,
        logging_steps=1,
        save_strategy="no",
        max_length=512,
        dataset_text_field="text",
        packing=False,
        seed=42,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    supervised_tokens = [token for token in trainer.train_dataset[0]["labels"] if token != -100]
    supervised_text = tokenizer.decode(supervised_tokens)
    if "COMMAND:" not in supervised_text:
        raise RuntimeError(f"response mask excluded the command: {supervised_text!r}")

    stats = trainer.train()
    print(f"training runtime: {stats.metrics['train_runtime'] / 60:.1f} minutes")
    print(f"training loss: {stats.metrics['train_loss']:.4f}")

    lora_dir = output_dir.with_name(f"{output_dir.name}-lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    export = model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    gguf_path = canonicalize_gguf_export(export, CANONICAL_GGUF)
    print(f"GGUF export: {gguf_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Omi flow router")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        validate_inputs(args.train, args.test)
        return
    train(args.train, args.test, args.output, args.base_model, args.epochs)


if __name__ == "__main__":
    main()
