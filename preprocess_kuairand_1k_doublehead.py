import argparse
import pickle
from collections import defaultdict
from pathlib import Path

from preprocess_kuairand_1k import (
    SEQ_SEP,
    build_video_cat_array,
    build_vocab,
    is_click_deep,
    iter_standard_logs,
    row_int,
    trim_history,
)


HEADER = [
    "sample_id",
    "pref_label",
    "pref_mask",
    "hes_label",
    "hes_mask",
    "behavior",
    "uid",
    "mid",
    "cat",
    "hist_mids",
    "hist_cats",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description="Build KuaiRand-1K double-head training data for click-only reclassification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "dataset" / "KuaiRand-1K" / "data",
        help="Directory containing KuaiRand-1K csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parent / "data_doublehead",
        help="Output directory for double-head data and vocab files.",
    )
    parser.add_argument(
        "--history-maxlen",
        type=int,
        default=100,
        help="Keep at most the most recent N clicked items in user history.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional debug limit for each log file.",
    )
    return parser.parse_args()


def behavior_type(click, click_deep, hate):
    if click_deep:
        return "click_deep"
    if click == 0:
        return "non_click"
    if hate == 1:
        return "click_hate"
    return "click_only"


def labels_for_behavior(behavior):
    if behavior == "click_deep":
        return 1, 1, 0, 1
    if behavior == "non_click":
        return 0, 1, 0, 0
    if behavior == "click_only":
        return 0, 0, 1, 1
    if behavior == "click_hate":
        return 0, 0, 0, 0
    raise ValueError(f"Unknown behavior type: {behavior}")


def format_doublehead_sample(sample_id, behavior, uid, mid, cat, mids_hist, cats_hist):
    pref_label, pref_mask, hes_label, hes_mask = labels_for_behavior(behavior)
    return "\t".join(
        [
            str(sample_id),
            str(pref_label),
            str(pref_mask),
            str(hes_label),
            str(hes_mask),
            behavior,
            str(uid),
            str(mid),
            str(cat),
            SEQ_SEP.join(str(x) for x in mids_hist),
            SEQ_SEP.join(str(x) for x in cats_hist),
        ]
    )


def main():
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_features_path = data_dir / "video_features_basic_1k.csv"
    train_path = output_dir / "doublehead_train.tsv"
    test_path = output_dir / "doublehead_test.tsv"
    item_info_path = output_dir / "item-info"
    reviews_info_path = output_dir / "reviews-info"
    uid_voc_path = output_dir / "uid_voc.pkl"
    mid_voc_path = output_dir / "mid_voc.pkl"
    cat_voc_path = output_dir / "cat_voc.pkl"

    video_to_cat = build_video_cat_array(video_features_path)

    history_mid = defaultdict(list)
    history_cat = defaultdict(list)

    observed_uids = set()
    observed_mids = set()
    observed_cats = set()

    counts = defaultdict(int)
    sample_id = 0

    print("Building double-head samples ...")
    print("  Head1 preference: click+deep vs non-click")
    print("  Head2 hesitation: click-only vs click+deep, only on clicked samples")
    with train_path.open("w", encoding="utf-8") as train_f, \
            test_path.open("w", encoding="utf-8") as test_f, \
            reviews_info_path.open("w", encoding="utf-8") as reviews_f:
        header = "\t".join(HEADER)
        train_f.write(header + "\n")
        test_f.write(header + "\n")

        for split_name, row in iter_standard_logs(data_dir, args.max_rows_per_file):
            counts["total_rows"] += 1

            uid = int(row["user_id"])
            mid = int(row["video_id"])
            click = row_int(row, "is_click")
            hate = row_int(row, "is_hate")
            click_deep = is_click_deep(row)
            behavior = behavior_type(click, click_deep, hate)
            cat = int(video_to_cat[mid]) if mid < len(video_to_cat) else 0

            observed_uids.add(uid)
            observed_mids.add(mid)
            observed_cats.add(cat)
            reviews_f.write(f"{uid}\t{mid}\n")

            mids_hist = history_mid[uid]
            cats_hist = history_cat[uid]

            if mids_hist:
                line = format_doublehead_sample(sample_id, behavior, uid, mid, cat, mids_hist, cats_hist)
                if split_name == "test":
                    test_f.write(line + "\n")
                    counts["test_samples"] += 1
                    counts[f"test_{behavior}"] += 1
                else:
                    train_f.write(line + "\n")
                    counts["train_samples"] += 1
                    counts[f"train_{behavior}"] += 1
                sample_id += 1
            else:
                counts["skipped_no_history"] += 1

            if click == 1:
                counts["clicked_rows"] += 1
                mids_hist.append(mid)
                cats_hist.append(cat)
                trim_history(mids_hist, args.history_maxlen)
                trim_history(cats_hist, args.history_maxlen)

            if counts["total_rows"] % 1000000 == 0:
                print(
                    f"  processed {counts['total_rows']:,} rows | "
                    f"train {counts['train_samples']:,} | test {counts['test_samples']:,}"
                )

    print("Writing item-info ...")
    with item_info_path.open("w", encoding="utf-8") as f:
        for mid in sorted(observed_mids):
            cat = int(video_to_cat[mid]) if mid < len(video_to_cat) else 0
            f.write(f"{mid}\t{cat}\n")

    print("Building vocabularies ...")
    uid_vocab = build_vocab(sorted(observed_uids), "default_uid")
    mid_vocab = build_vocab(sorted(observed_mids), "default_mid")
    cat_vocab = build_vocab(sorted(observed_cats), "default_cat")

    with uid_voc_path.open("wb") as f:
        pickle.dump(uid_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with mid_voc_path.open("wb") as f:
        pickle.dump(mid_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with cat_voc_path.open("wb") as f:
        pickle.dump(cat_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nDone.")
    print(f"  total log rows:          {counts['total_rows']:,}")
    print(f"  clicked rows:            {counts['clicked_rows']:,}")
    print(f"  skipped no history:      {counts['skipped_no_history']:,}")
    print(f"  train samples:           {counts['train_samples']:,}")
    print(f"  train click+deep:        {counts['train_click_deep']:,}")
    print(f"  train click-only:        {counts['train_click_only']:,}")
    print(f"  train click+hate:        {counts['train_click_hate']:,}")
    print(f"  train non-click:         {counts['train_non_click']:,}")
    print(f"  test samples:            {counts['test_samples']:,}")
    print(f"  test click+deep:         {counts['test_click_deep']:,}")
    print(f"  test click-only:         {counts['test_click_only']:,}")
    print(f"  test click+hate:         {counts['test_click_hate']:,}")
    print(f"  test non-click:          {counts['test_non_click']:,}")
    print(f"  unique users:            {len(observed_uids):,}")
    print(f"  unique videos:           {len(observed_mids):,}")
    print(f"  unique categories:       {len(observed_cats):,}")
    print(f"  output dir:              {output_dir}")


if __name__ == "__main__":
    main()
