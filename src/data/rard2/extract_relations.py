import argparse
import csv
from collections import defaultdict

from tqdm import tqdm


def extract_paper_info(file_path):
    paper_info = defaultdict(set)

    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader):
            source_id = row["source_doc_id"]
            rec_id = row["rec_doc_id"]
            if source_id != "null":
                paper_info[source_id].add(rec_id)
            if rec_id != "null":
                paper_info[rec_id]

    return paper_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--threshold", type=int, default=5)

    args = parser.parse_args()

    paper_info = extract_paper_info(args.file_path)

    with open(args.save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "recommended_papers"])
        for paper_id, recommended in tqdm(paper_info.items()):

            if not recommended:
                continue
            if len(recommended) < args.threshold:
                continue
            writer.writerow([paper_id, ",".join(recommended)])

if __name__ == "__main__":
    main()
