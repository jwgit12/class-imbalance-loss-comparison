from pathlib import Path

import yaml

# -----------------------------------------------------------
# Load config from YAML
# -----------------------------------------------------------
def load_config():
    """
    Loads the experiment configuration from `experiment_config.yaml`.

    Returns:
        dict: Parsed configuration values.
    """
    config_path = Path(__file__).parent / "experiment_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)

