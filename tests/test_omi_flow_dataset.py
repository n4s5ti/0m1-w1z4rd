from pathlib import Path

from nlcli_wizard.dataset_flows import DEFAULT_CATALOG, FlowDatasetGenerator


def test_default_catalog_is_owned_by_this_repository():
    repository = Path(__file__).resolve().parent.parent

    assert DEFAULT_CATALOG == repository / "nlcli_wizard" / "catalog" / "flows.json"
    assert DEFAULT_CATALOG.is_file()


def test_flow_dataset_uses_only_catalog_targets_and_none():
    generator = FlowDatasetGenerator()
    records = generator.generate_records()
    allowed = {"none"}
    allowed.update("flow " + str(flow["name"]) for flow in generator.load_catalog())

    targets = {
        record["output"].removeprefix("COMMAND: ").strip()
        for record in records
    }

    assert targets == allowed
    assert all("CONFIDENCE:" not in record["output"] for record in records)
    assert all("EXPLANATION:" not in record["output"] for record in records)
