import argparse
from collections import Counter
from pathlib import Path


BEHAVIORS = ["click_deep", "click_only", "click_hate", "non_click"]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    code_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="Summarize behavior-type sizes for Group A/B/C experiments.")
    parser.add_argument("--doublehead-dir", type=Path, default=code_dir / "data_doublehead")
    parser.add_argument("--groupa-dir", type=Path, default=code_dir / "data-groupA")
    parser.add_argument("--groupb-dir", type=Path, default=code_dir / "data-groupB")
    parser.add_argument("--groupc-dir", type=Path, default=code_dir / "data-groupC")
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
        help="Optional double-head-format validation TSV. The current pipeline does not create validation by default.",
    )
    return parser.parse_args()


def count_din_labels(path):
    counts = Counter()
    if not path.exists():
        return counts

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            label = line.split("\t", 1)[0]
            counts[label] += 1
    return counts


def count_behaviors(path, required=False):
    counts = Counter()
    if path is None:
        return counts
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing behavior file: {path}")
        return counts

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: idx for idx, name in enumerate(header)}
        if "behavior" not in col:
            raise ValueError(f"{path} must contain a behavior column.")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            counts[fields[col["behavior"]]] += 1
    return counts


def count_groupc_decisions(path):
    counts = Counter()
    if not path.exists():
        return counts

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: idx for idx, name in enumerate(header)}
        for line in f:
            fields = line.rstrip("\n").split("\t")
            counts[fields[col["decision"]]] += 1
    return counts


def three_class_counts(counts):
    return {
        "click_deep": counts["click_deep"],
        "click_only": counts["click_only"] + counts["click_hate"],
        "non_click": counts["non_click"],
    }


def group_split_summary(group, split, behavior_counts, groupc_decisions=None, din_counts=None):
    three = three_class_counts(behavior_counts)
    click_deep = three["click_deep"]
    click_only = three["click_only"]
    non_click = three["non_click"]

    if group == "A":
        if split == "train":
            included = click_deep + click_only + non_click
            dropped = 0
            positives = click_deep + click_only
            negatives = non_click
        else:
            included = click_deep + non_click
            dropped = click_only
            positives = click_deep
            negatives = non_click
    elif group == "B":
        if split == "train":
            included = click_deep + click_only + non_click
            dropped = 0
            positives = click_deep
            negatives = click_only + non_click
        else:
            included = click_deep + non_click
            dropped = click_only
            positives = click_deep
            negatives = non_click
    elif group == "C":
        if split == "train":
            decisions = groupc_decisions or Counter()
            real_dislike = decisions["negative_real_dislike"]
            dropped_click_only = max(0, click_only - real_dislike)
            included = click_deep + non_click + real_dislike
            dropped = dropped_click_only
            positives = click_deep
            negatives = non_click + real_dislike
        else:
            included = click_deep + non_click
            dropped = click_only
            positives = click_deep
            negatives = non_click
    else:
        raise ValueError(f"Unknown group: {group}")

    return {
        "group": group,
        "split": split,
        "raw_click_deep": click_deep,
        "raw_click_only": click_only,
        "raw_non_click": non_click,
        "included_total": included,
        "dropped_click_only": dropped,
        "din_positive": positives,
        "din_negative": negatives,
        "actual_din_positive": din_counts["1"] if din_counts else "",
        "actual_din_negative": din_counts["0"] if din_counts else "",
    }


def print_table(rows):
    headers = [
        "group",
        "split",
        "raw_click_deep",
        "raw_click_only",
        "raw_non_click",
        "included_total",
        "dropped_click_only",
        "din_positive",
        "din_negative",
        "actual_din_positive",
        "actual_din_negative",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[h]) for h in headers))


def main():
    args = parse_args()
    train_counts = count_behaviors(args.doublehead_dir / "doublehead_train.tsv", required=True)
    test_counts = count_behaviors(args.doublehead_dir / "doublehead_test.tsv", required=True)
    val_counts = count_behaviors(args.validation_file)
    groupc_decisions = count_groupc_decisions(args.groupc_dir / "groupC_click_only_decisions.tsv")

    group_dirs = {
        "A": args.groupa_dir,
        "B": args.groupb_dir,
        "C": args.groupc_dir,
    }

    rows = []
    for group in ["A", "B", "C"]:
        group_dir = group_dirs[group]
        train_din_counts = count_din_labels(group_dir / "local_train_splitByUser")
        test_din_counts = count_din_labels(group_dir / "local_test_splitByUser")
        rows.append(group_split_summary(group, "train", train_counts, groupc_decisions, train_din_counts))
        rows.append(group_split_summary(group, "test", test_counts, groupc_decisions, test_din_counts))
        if val_counts:
            rows.append(group_split_summary(group, "validation", val_counts, groupc_decisions))

    print_table(rows)
    if not val_counts:
        print("\nNOTE: validation is not shown because the current pipeline does not create a validation split.")
        print("Pass --validation-file with a double-head-format TSV if you add one later.")

    if train_counts["click_hate"] or test_counts["click_hate"] or val_counts["click_hate"]:
        print("\nNOTE: raw_click_only includes click_hate, because both are clicked-but-not-click+deep.")
        print(
            "Raw click_hate counts: "
            f"train={train_counts['click_hate']}, "
            f"test={test_counts['click_hate']}, "
            f"validation={val_counts['click_hate']}"
        )


if __name__ == "__main__":
    main()
