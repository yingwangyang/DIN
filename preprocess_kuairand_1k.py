import argparse
import csv
import pickle
from array import array
from collections import defaultdict
from pathlib import Path


SEQ_SEP = "\x02"


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(
        description="Convert KuaiRand-1K standard logs into the text/pickle files expected by trainer.py."
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
        default=script_dir.parent / "data",
        help="Output directory. trainer.py expects ../data relative to code/DIN.",
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


def first_tag_to_int(tag_value):
    if not tag_value:
        return 0

    first = tag_value.split(",")[0].strip()
    if not first or first.lower() == "nan":
        return 0

    try:
        return int(float(first))
    except ValueError:
        return 0


def build_video_cat_array(video_features_path):
    print(f"Loading video tags from {video_features_path} ...")
    video_to_cat = array("I", [0])

    with video_features_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            video_id = int(row["video_id"])
            cat_id = first_tag_to_int(row.get("tag", ""))

            if video_id >= len(video_to_cat):
                video_to_cat.extend([0] * (video_id + 1 - len(video_to_cat)))
            video_to_cat[video_id] = cat_id

            if idx % 500000 == 0:
                print(f"  loaded {idx:,} video rows")

    print(f"Finished loading {len(video_to_cat):,} video slots")
    return video_to_cat


def iter_standard_logs(data_dir, max_rows_per_file=None):
    filenames = [
        ("train", "log_standard_4_08_to_4_21_1k.csv"),
        ("test", "log_standard_4_22_to_5_08_1k.csv"),
    ]

    for split_name, filename in filenames:
        path = data_dir / filename
        print(f"Reading {path} ...")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                yield split_name, row
                if max_rows_per_file is not None and idx >= max_rows_per_file:
                    break


def trim_history(history, maxlen):
    if len(history) > maxlen:
        del history[:-maxlen]


def build_vocab(sorted_values, unk_token):
    vocab = {unk_token: 0}
    for idx, value in enumerate(sorted_values, start=1):
        vocab[str(value)] = idx
    return vocab


def main():
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_features_path = data_dir / "video_features_basic_1k.csv"
    train_path = output_dir / "local_train_splitByUser"
    test_path = output_dir / "local_test_splitByUser"
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

    train_samples = 0
    test_samples = 0
    skipped_no_history = 0
    total_rows = 0
    click_rows = 0

    print("Building DIN samples ...")
    with train_path.open("w", encoding="utf-8") as train_f, \
            test_path.open("w", encoding="utf-8") as test_f, \
            reviews_info_path.open("w", encoding="utf-8") as reviews_f:

        for split_name, row in iter_standard_logs(data_dir, args.max_rows_per_file):
            total_rows += 1

            uid = int(row["user_id"])
            mid = int(row["video_id"])
            label = int(row["is_click"])
            cat = int(video_to_cat[mid]) if mid < len(video_to_cat) else 0

            observed_uids.add(uid)
            observed_mids.add(mid)
            observed_cats.add(cat)
            reviews_f.write(f"{uid}\t{mid}\n")

            mids_hist = history_mid[uid]
            cats_hist = history_cat[uid]

            if mids_hist:
                line = "\t".join(
                    [
                        str(label),
                        str(uid),
                        str(mid),
                        str(cat),
                        SEQ_SEP.join(str(x) for x in mids_hist),
                        SEQ_SEP.join(str(x) for x in cats_hist),
                    ]
                )

                if split_name == "test":
                    test_f.write(line + "\n")
                    test_samples += 1
                else:
                    train_f.write(line + "\n")
                    train_samples += 1
            else:
                skipped_no_history += 1

            if label == 1:
                click_rows += 1
                mids_hist.append(mid)
                cats_hist.append(cat)
                trim_history(mids_hist, args.history_maxlen)
                trim_history(cats_hist, args.history_maxlen)

            if total_rows % 1000000 == 0:
                print(
                    f"  processed {total_rows:,} rows | "
                    f"train {train_samples:,} | test {test_samples:,}"
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
    print(f"  total log rows:        {total_rows:,}")
    print(f"  clicked rows:          {click_rows:,}")
    print(f"  skipped no history:    {skipped_no_history:,}")
    print(f"  train samples:         {train_samples:,}")
    print(f"  test samples:          {test_samples:,}")
    print(f"  unique users:          {len(observed_uids):,}")
    print(f"  unique videos:         {len(observed_mids):,}")
    print(f"  unique categories:     {len(observed_cats):,}")
    print(f"  output dir:            {output_dir}")


if __name__ == "__main__":
    main()
