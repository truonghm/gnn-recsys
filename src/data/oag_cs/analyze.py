import argparse
import json
import os
import tempfile
import zipfile
from collections import Counter


def iter_json(filename):
    """Iterate over each line of a file containing JSON format."""
    with open(filename, encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def iter_lines(raw_path, data_type):
    """Iterate over each line of each txt file in the OAG dataset of a specific data type and parse the JSON into a dictionary.

    :param raw_path: str The directory where the raw zip files are located.
    :param data_type: str The type of data, one of author, paper, venue, affiliation.
    :return: Iterable[dict]
    """
    with tempfile.TemporaryDirectory() as tmp:
        for zip_file in os.listdir(raw_path):
            if zip_file.startswith(f"mag_{data_type}s"):
                with zipfile.ZipFile(os.path.join(raw_path, zip_file)) as z:
                    for txt in z.namelist():
                        print(f"{zip_file}\\{txt}")
                        txt_file = z.extract(txt, tmp)
                        yield from iter_json(txt_file)
                        os.remove(txt_file)


def analyze(args):
    total = 0
    max_fields = set()
    min_fields = None
    field_count = Counter()
    sample = None
    for d in iter_lines(args.raw_path, args.type):
        total += 1
        keys = [k for k in d if d[k]]
        max_fields.update(keys)
        if min_fields is None:
            min_fields = set(keys)
        else:
            min_fields.intersection_update(keys)
        field_count.update(keys)
        if len(keys) == len(max_fields):
            sample = d
    print("Data type:", args.type)
    print("Total:", total)
    print("Max field set:", max_fields)
    print("Min field set:", min_fields)
    print("Field occurrence %:", {k: "{0:.2f}".format((v / total) * 100) for k, v in field_count.items()})
    print("Example:", sample)


def main():
    parser = argparse.ArgumentParser(description="Analyse Microsoft OAG Dataset")
    parser.add_argument("type", choices=["author", "paper", "venue", "affiliation"], help="data type")
    parser.add_argument("raw_path", help="path to raw dataset")
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
