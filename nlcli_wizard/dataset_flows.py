"""Deterministically generate flow-router training records from Omi's flow catalog.

The generated set is for training only. Evaluation belongs in the independently
written ``data/flows_test_handwritten.jsonl`` holdout so generated phrasing never
becomes an evaluation fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from eval.contamination import Example
from eval.splits import Split, split_by_command, write_jsonl

DEFAULT_CATALOG = Path(__file__).resolve().parent / "catalog" / "flows.json"
DEFAULT_OUTPUT = Path("data/flows_training.jsonl")
INSTRUCTION_PREFIX = "Translate to flow command: "


class FlowDatasetGenerator:
    """Generate deterministic Alpaca-format examples for the configured flows."""

    def __init__(self, catalog_path: Path = DEFAULT_CATALOG) -> None:
        self.catalog_path = catalog_path

    def load_catalog(self) -> List[Dict[str, object]]:
        """Return flow entries sorted by name from the shared Omi manifest."""
        with self.catalog_path.open(encoding="utf-8") as catalog_file:
            catalog = json.load(catalog_file)

        entries = catalog.get("flows", catalog) if isinstance(catalog, dict) else catalog
        if not isinstance(entries, list):
            raise ValueError("flows catalog must be a list or an object with a 'flows' list")

        flows: List[Dict[str, object]] = []
        names = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each flow catalog entry must be an object")
            name = entry.get("name")
            description = entry.get("description")
            if not isinstance(name, str) or not name:
                raise ValueError("each flow catalog entry needs a non-empty name")
            if not isinstance(description, str) or not description:
                raise ValueError("each flow catalog entry needs a non-empty description")
            if name in names:
                raise ValueError("flow catalog names must be unique: {}".format(name))
            names.add(name)
            flows.append(entry)

        if not flows:
            raise ValueError("flows catalog must contain at least one flow")
        return sorted(flows, key=lambda flow: str(flow["name"]))

    @staticmethod
    def _record(utterance: str, command: str, explanation: str) -> Dict[str, str]:
        return {
            "instruction": INSTRUCTION_PREFIX + utterance,
            "input": "",
            "output": "COMMAND: {}\n".format(command),
        }

    def _flow_records(self, flow: Dict[str, object]) -> List[Dict[str, str]]:
        name = str(flow["name"])
        description = str(flow["description"])
        magic_words = flow.get("magic_words", [])
        readable_name = name.replace("_", " ")
        phrases = [
            "start the {} flow".format(readable_name),
            "launch {}".format(readable_name),
            "open the {} experience".format(description.lower()),
            "run the workflow for {}".format(description.lower()),
            "I need {} now".format(description.lower()),
            "activate the {} routine".format(readable_name),
            "can you begin {}".format(description.lower()),
            "take me into the {} flow".format(readable_name),
            "please kick off {}".format(description.lower()),
            "bring up {}".format(readable_name),
            "let's use the {} workflow".format(description.lower()),
            "initiate {}".format(description.lower()),
        ]
        if isinstance(magic_words, list):
            phrases.extend(
                "use {}".format(word) for word in magic_words if isinstance(word, str) and word
            )

        command = "flow " + name
        explanation = "Launches the {} flow".format(description)
        return [self._record(phrase, command, explanation) for phrase in phrases]

    def _negative_records(self) -> List[Dict[str, str]]:
        utterances = [
            "what time is it",
            "tell me a joke",
            "what is the weather outside",
            "set a timer for five minutes",
            "remind me to call Sam tomorrow",
            "turn the volume down",
            "how do I make pasta",
            "read my latest email",
            "play some music",
            "what is on my calendar",
            "find a coffee shop nearby",
            "translate this sentence to Spanish",
            "what does this word mean",
            "calculate fifteen times seven",
            "give me the latest news",
            "take a note about groceries",
            "how much battery do I have",
            "send a message to Jordan",
            "search the web for local events",
            "turn on the living room lights",
            "what was the score last night",
            "book a ride home",
            "read the next item on my to-do list",
            "find the nearest pharmacy",
        ]
        return [
            self._record(utterance, "none", "No configured flow matches this request")
            for utterance in utterances
        ]

    def generate_records(self) -> List[Dict[str, str]]:
        """Build the complete training set without random sampling or shuffling."""
        records: List[Dict[str, str]] = []
        for flow in self.load_catalog():
            records.extend(self._flow_records(flow))
        records.extend(self._negative_records())
        return records

    @staticmethod
    def command_level_split(records: Sequence[Dict[str, str]], test_fraction: float = 0.2) -> Split:
        """Expose the project's command-level splitter for generated-only probes.

        The production holdout intentionally is not produced here: it measures new
        phrasings for the fixed catalog and therefore may share target commands.
        """
        examples = [
            Example(
                instruction=record["instruction"],
                command=record["output"].split("COMMAND: ", 1)[1].split("\n", 1)[0],
            )
            for record in records
        ]
        return split_by_command(examples, test_fraction=test_fraction, seed=42)

    def write_training(self, output_path: Path = DEFAULT_OUTPUT) -> int:
        """Write deterministic training JSONL and return its record count."""
        records = self.generate_records()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record) + "\n")
        return len(records)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Omi flow-router training data.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--command-holdout",
        type=Path,
        help="Optionally write a generated command-disjoint probe using eval/splits.py.",
    )
    args = parser.parse_args(argv)

    generator = FlowDatasetGenerator(args.catalog)
    records = generator.generate_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")

    if args.command_holdout:
        split = generator.command_level_split(records)
        write_jsonl(split.test, args.command_holdout)

    print("Generated {} training records at {}".format(len(records), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
