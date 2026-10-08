"""Functional ComfyUI client for submitting workflows to Modal workers.

Follows strict functional programming principles:
- Pure functions for template parsing and parameter substitution.
- Explicit I/O functions for network and disk interactions.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_workflow_template(workflow_path: str) -> Dict[str, Any]:
    """Reads and parses a ComfyUI API workflow JSON template."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_workflow_params(
    template: Dict[str, Any], substitutions: Dict[str, Any]
) -> Dict[str, Any]:
    """Returns a new workflow dictionary with placeholders substituted.

    Replaces string occurrences like '{{PROMPT}}' or specific node values.
    """
    raw_str = json.dumps(template)
    for placeholder, val in substitutions.items():
        if isinstance(val, (int, float)):
            # Replace numeric placeholders directly if in string format
            raw_str = raw_str.replace(f'"{{{{{placeholder}}}}}"', str(val))
            raw_str = raw_str.replace(f"{{{{{placeholder}}}}}", str(val))
        else:
            raw_str = raw_str.replace(f"{{{{{placeholder}}}}}", str(val))

    return json.loads(raw_str)


def save_media_bytes(data: bytes, output_path: str) -> str:
    """Writes binary media bytes (PNG/MP4/WAV) to disk and returns the string path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path.resolve())
