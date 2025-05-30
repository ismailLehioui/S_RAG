import os
import json
import yaml


def load_json_data(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml_data(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenarios_from_file(scenario_path: str):
    ext = os.path.splitext(scenario_path)[1].lower()
    if ext in [".yaml", ".yml"]:
        data = load_yaml_data(scenario_path)
        # YAML: either a list of scenarios or a dict with a 'scenarios' key
        if isinstance(data, dict) and "scenarios" in data:
            return data["scenarios"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Format YAML non supporté dans {scenario_path}")
    elif ext == ".json":
        with open(scenario_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "scenarios" in data:
            return data["scenarios"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Format JSON non supporté dans {scenario_path}")
    else:
        # Assume plain text: one scenario per line or block
        with open(scenario_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Split by double newline or line if needed
        scenarios = [s.strip() for s in content.split("\n\n") if s.strip()]
        if not scenarios:
            scenarios = [s.strip() for s in content.splitlines() if s.strip()]
        return scenarios


def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
