import argparse
import os

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix


def create_implicit_paper_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df["item_ids"] = df["item_ids"].str.split()

    all_items = []
    user_indices = []
    for idx, (user, items) in enumerate(zip(df["user_id"], df["item_ids"])):
        all_items.extend(items)
        user_indices.extend([idx] * len(items))

    unique_users = df["user_id"].unique()
    unique_items = sorted(set(all_items))
    user_to_index = {user: idx for idx, user in enumerate(unique_users)}
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}

    rows = [user_to_index[df["user_id"].iloc[i]] for i in user_indices]
    cols = [item_to_index[item] for item in all_items]
    data = np.ones(len(rows))

    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))
    item_item_matrix = user_item_matrix.T @ user_item_matrix

    item_item_df = pd.DataFrame(item_item_matrix.toarray(), index=unique_items, columns=unique_items)
    item_item_df = item_item_df.div(item_item_df.max().max())

    return item_item_df


def create_implicit_paper_relations(df: pd.DataFrame) -> pd.DataFrame:
    df["item_ids"] = df["item_ids"].str.split()

    paper_relations = {}
    paper_counts = {}

    for items in df["item_ids"]:
        for item in items:
            if item not in paper_relations:
                paper_relations[item] = {}
            for related_item in items:
                if related_item != item:
                    paper_relations[item][related_item] = paper_relations[item].get(related_item, 0) + 1
            paper_counts[item] = paper_counts.get(item, 0) + 1

    ranked_relations = {}
    for item, related_items in paper_relations.items():
        ranked_list = sorted(related_items.items(), key=lambda x: x[1], reverse=True)
        ranked_relations[item] = [rel_item for rel_item, _ in ranked_list]

    relations_df = pd.DataFrame(
        [(k, v) for k, v in ranked_relations.items() if v], columns=["paper_id", "related_papers"]
    )

    return relations_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-type", type=str, choices=["matrix", "relations"], default="relations")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise ValueError("Data dir is not a directory")

    files = os.listdir(args.data_dir)
    data = []
    for file in files:
        with open(os.path.join(args.data_dir, file), "r") as f:
            for line in f:
                numbers = line.strip().split()
                user_id = numbers[0]
                item_ids = " ".join(numbers[1:])
                data.append([user_id, item_ids])

    if args.output_type == "relations":
        df = pd.DataFrame(data, columns=["user_id", "item_ids"])
        item_item_relations = create_implicit_paper_relations(df)
        item_item_relations.to_json(args.output_path, orient="records", lines=True)
    else:
        df = pd.DataFrame(data, columns=["user_id", "item_ids"])
        print("Shape of input data:", df.shape)
        item_item_matrix = create_implicit_paper_matrix(df)
        item_item_matrix.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
