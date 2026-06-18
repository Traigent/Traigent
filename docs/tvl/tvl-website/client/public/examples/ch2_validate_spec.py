"""Quick helper for Chapter 2 to inspect a TVL spec and print the knobs."""
from pathlib import Path
import textwrap

import yaml


def load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_sections(spec: dict, required: list[str]) -> list[str]:
    missing = [section for section in required if section not in spec]
    return missing


def print_configuration_space(space: dict) -> None:
    print("Configuration Space")
    print("-------------------")
    for name, definition in space.items():
        type_ = definition.get("type", "unknown")
        unit = definition.get("unit")
        details = []
        if "values" in definition:
            details.append(f"values={definition['values']}")
        if "range" in definition:
            details.append(f"range={definition['range']}")
        info = ", ".join(details)
        if unit:
            info = f"{info} ({unit})" if info else f"unit={unit}"
        print(f"- {name} [{type_}]: {info}")


def main() -> None:
    spec_path = Path(__file__).with_name("ch2_hello_tvl.tvl.yml")
    spec = load_spec(spec_path)

    required_sections = ["spec", "metadata", "configuration_space", "objectives", "optimization"]
    missing = ensure_sections(spec, required_sections)
    if missing:
        print("Missing sections:", ", ".join(missing))
        return

    print(textwrap.dedent(
        """\
        ✅ Spec loaded successfully.
        id       : {id}
        version  : {version}
        owner    : {owner}
        """
    ).format(
        id=spec["spec"].get("id", "<unknown>"),
        version=spec["spec"].get("version", "<unknown>"),
        owner=spec["metadata"].get("owner", "<unknown>"),
    ))

    print_configuration_space(spec["configuration_space"])


if __name__ == "__main__":
    main()
