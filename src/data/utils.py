import json
from pathlib import Path
from typing import Dict, Iterator


def iter_json(file_path: Path) -> Iterator[Dict]:
    """Iterate over JSON lines in a file.

    Parameters
    ----------
    file_path : Path
        Path to the JSON lines file

    Yields
    ------
    dict
        Parsed JSON object from each line

    Notes
    -----
    This is a core utility for loading the OAG dataset files which are stored
    in JSON lines format (one JSON object per line).
    """
    with open(file_path, encoding="utf8") as f:
        for line in f:
            yield json.loads(line)
