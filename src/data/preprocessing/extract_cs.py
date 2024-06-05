import argparse
import json

from src.config import settings
from src.data.preprocessing.analyze import iter_lines
from src.data.preprocessing.constants import COMPUTER_SCIENCE_L2_TAGS, COMPUTER_SCIENCE_TAG

START_YEAR = 2010
END_YEAR = 2021


def extract_papers(raw_path):
    valid_keys = ["title", "authors", "venue", "year", "indexed_abstract", "fos", "references"]
    cs_fields = set(COMPUTER_SCIENCE_L2_TAGS)
    for p in iter_lines(raw_path, "paper"):
        if not all(p.get(k) for k in valid_keys):
            continue
        fos = {f["name"] for f in p["fos"]}
        abstract = parse_abstract(p["indexed_abstract"])
        if (
            COMPUTER_SCIENCE_TAG in fos and not fos.isdisjoint(cs_fields) and START_YEAR <= p["year"] <= END_YEAR
            # and len(p["title"]) <= 200
            # and len(abstract) <= 4000
            # and 1 <= len(p["authors"]) <= 20
            # and 1 <= len(p["references"]) <= 100
        ):
            try:
                yield {
                    "id": p["id"],
                    "title": p["title"],
                    "authors": [a["id"] for a in p["authors"]],
                    "venue": p["venue"]["id"],
                    "year": p["year"],
                    "abstract": abstract,
                    "fos": list(fos),
                    "references": p["references"],
                    "n_citation": p.get("n_citation", 0),
                }
            except KeyError:
                pass


def parse_abstract(indexed_abstract):
    try:
        abstract = json.loads(indexed_abstract)
        words = [""] * abstract["IndexLength"]
        for w, idx in abstract["InvertedIndex"].items():
            for i in idx:
                words[i] = w
        return " ".join(words)
    except json.JSONDecodeError:
        return ""


def extract_authors(raw_path, author_ids):
    for a in iter_lines(raw_path, "author"):
        if a["id"] in author_ids:
            yield {
                "id": a["id"],
                "name": a["name"],
                "org": int(a["last_known_aff_id"]) if "last_known_aff_id" in a else None,
            }


def extract_venues(raw_path, venue_ids):
    for v in iter_lines(raw_path, "venue"):
        if v["id"] in venue_ids:
            yield {"id": v["id"], "name": v["DisplayName"]}


def extract_institutions(raw_path, institution_ids):
    for i in iter_lines(raw_path, "affiliation"):
        if i["id"] in institution_ids:
            yield {"id": i["id"], "name": i["DisplayName"]}


def extract(args):
    print("Extracting papers...")
    paper_ids, author_ids, venue_ids, fields = set(), set(), set(), set()
    output_path = settings.DATA_DIR / "cs"
    # check if mag_papers.txt already exists
    if (output_path / "mag_papers.txt").exists():
        print(f"File {output_path / 'mag_papers.txt'} already exists")
        # use the existing file to get the paper_ids, author_ids, venue_ids, and fields
        with open(output_path / "mag_papers.txt", "r", encoding="utf8") as f:
            for line in f:
                p = json.loads(line)
                paper_ids.add(p["id"])
                author_ids.update(p["authors"])
                venue_ids.add(p["venue"])
                fields.update(p["fos"])
    else:
        with open(output_path / "mag_papers.txt", "w", encoding="utf8") as f:
            for p in extract_papers(args.raw_path):
                paper_ids.add(p["id"])
                author_ids.update(p["authors"])
                venue_ids.add(p["venue"])
                fields.update(p["fos"])
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")
        print(f"Paper extraction completed, saved to {f.name}")
    print(
        f"Number of papers {len(paper_ids)},\n" \
            + f"number of scholars {len(author_ids)},\n" \
            + f"number of journals {len(venue_ids)},\n" \
            + f"number of fields {len(fields)}"
    )

    print("Extracting scholars...")
    institution_ids = set()
    if (output_path / "mag_authors.txt").exists():
        print(f"File {output_path / 'mag_authors.txt'} already exists")
        # use the existing file to get the institution_ids
        with open(output_path / "mag_authors.txt", "r", encoding="utf8") as f:
            for line in f:
                a = json.loads(line)
                if a["org"]:
                    institution_ids.add(a["org"])
    else:
        with open(output_path / "mag_authors.txt", "w", encoding="utf8") as f:
            for a in extract_authors(args.raw_path, author_ids):
                if a["org"]:
                    institution_ids.add(a["org"])
                json.dump(a, f, ensure_ascii=False)
                f.write("\n")
        print(f"Scholar extraction completed, saved to {f.name}")
    print(f"Number of institutions {len(institution_ids)}")

    print("Extracting journals...")
    if (output_path / "mag_venues.txt").exists():
        print(f"File {output_path / 'mag_venues.txt'} already exists, skipping")
    else:
        with open(output_path / "mag_venues.txt", "w", encoding="utf8") as f:
            for v in extract_venues(args.raw_path, venue_ids):
                json.dump(v, f, ensure_ascii=False)
                f.write("\n")
        print(f"Journal extraction completed, saved to {f.name}")

    print("Extracting institutions...")
    if (output_path / "mag_institutions.txt").exists():
        print(f"File {output_path / 'mag_institutions.txt'} already exists, skipping")
        with open(output_path / "mag_institutions.txt", "w", encoding="utf8") as f:
            for i in extract_institutions(args.raw_path, institution_ids):
                json.dump(i, f, ensure_ascii=False)
                f.write("\n")
        print(f"Institution extraction completed, saved to {f.name}")

    print("Extracting fields...")
    if (output_path / "mag_fields.txt").exists():
        print(f"File {output_path / 'mag_fields.txt'} already exists, skipping")
        return
    fields.remove(COMPUTER_SCIENCE_TAG)
    fields = sorted(fields)
    with open(output_path / "mag_fields.txt", "w", encoding="utf8") as f:
        for i, field in enumerate(fields):
            json.dump({"id": i, "name": field}, f, ensure_ascii=False)
            f.write("\n")
    print(f"Field extraction completed, saved to {f.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract a subset of the OAG dataset in the field of computer science")
    parser.add_argument("raw_path", help="Directory where the original zip file is located")
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
