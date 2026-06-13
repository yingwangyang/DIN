import pickle
from collections import defaultdict

from preprocess_kuairand_1k import (
    SEQ_SEP,
    build_video_cat_array,
    build_vocab,
    is_click_deep,
    iter_standard_logs,
    parse_args,
    row_int,
    trim_history,
)


def format_sample(label, uid, mid, cat, mids_hist, cats_hist):
    return "\t".join(
        [
            str(label),
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
    train_click_deep_samples = 0
    train_click_only_as_negative = 0
    train_non_click_samples = 0
    test_samples = 0
    test_click_deep_samples = 0
    test_non_click_samples = 0
    test_click_only_ignored = 0
    skipped_no_history = 0
    total_rows = 0
    click_rows = 0

    print("Building Group B DIN samples ...")
    print("  train label: click+deep positive, non-click and click-only negative")
    print("  eval label: click+deep vs non-click, click-only ignored")
    with train_path.open("w", encoding="utf-8") as train_f, \
            test_path.open("w", encoding="utf-8") as test_f, \
            reviews_info_path.open("w", encoding="utf-8") as reviews_f:

        for split_name, row in iter_standard_logs(data_dir, args.max_rows_per_file):
            total_rows += 1

            uid = int(row["user_id"])
            mid = int(row["video_id"])
            click = row_int(row, "is_click")
            click_deep = is_click_deep(row)
            cat = int(video_to_cat[mid]) if mid < len(video_to_cat) else 0

            observed_uids.add(uid)
            observed_mids.add(mid)
            observed_cats.add(cat)
            reviews_f.write(f"{uid}\t{mid}\n")

            mids_hist = history_mid[uid]
            cats_hist = history_cat[uid]

            if mids_hist:
                if split_name == "test":
                    if click_deep:
                        label = 1
                        test_click_deep_samples += 1
                    elif click == 0:
                        label = 0
                        test_non_click_samples += 1
                    else:
                        label = None
                        test_click_only_ignored += 1

                    if label is not None:
                        test_f.write(format_sample(label, uid, mid, cat, mids_hist, cats_hist) + "\n")
                        test_samples += 1
                else:
                    if click_deep:
                        label = 1
                        train_click_deep_samples += 1
                    else:
                        label = 0
                        if click == 1:
                            train_click_only_as_negative += 1
                        else:
                            train_non_click_samples += 1

                    train_f.write(format_sample(label, uid, mid, cat, mids_hist, cats_hist) + "\n")
                    train_samples += 1
            else:
                skipped_no_history += 1

            if click == 1:
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
    print(f"  total log rows:               {total_rows:,}")
    print(f"  clicked rows:                 {click_rows:,}")
    print(f"  skipped no history:           {skipped_no_history:,}")
    print(f"  train samples:                {train_samples:,}")
    print(f"  train click+deep positives:   {train_click_deep_samples:,}")
    print(f"  train click-only negatives:   {train_click_only_as_negative:,}")
    print(f"  train non-click negatives:    {train_non_click_samples:,}")
    print(f"  test samples:                 {test_samples:,}")
    print(f"  test click+deep:              {test_click_deep_samples:,}")
    print(f"  test non-click:               {test_non_click_samples:,}")
    print(f"  test click-only skip:         {test_click_only_ignored:,}")
    print(f"  unique users:                 {len(observed_uids):,}")
    print(f"  unique videos:                {len(observed_mids):,}")
    print(f"  unique categories:            {len(observed_cats):,}")
    print(f"  output dir:                   {output_dir}")


if __name__ == "__main__":
    main()
