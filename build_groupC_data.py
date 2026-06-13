import argparse
import shutil
from collections import Counter
from pathlib import Path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build Group C DIN data from double-head click-only scores.")
    parser.add_argument("--doublehead-dir", type=Path, default=script_dir.parent / "data_doublehead")
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "data")
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--pref-low", type=float, default=0.35)
    parser.add_argument("--pref-high", type=float, default=0.65)
    parser.add_argument("--hes-low", type=float, default=0.35)
    parser.add_argument("--hes-high", type=float, default=0.65)
    parser.add_argument(
        "--uncertainty-high-quantile",
        type=float,
        default=0.75,
        help="Click-only samples at or above this uncertainty quantile are dropped as model blind spots.",
    )
    return parser.parse_args()


def read_scores(scores_file):
    scores = {}
    uncertainties = []
    with scores_file.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: idx for idx, name in enumerate(header)}
        for line in f:
            fields = line.rstrip("\n").split("\t")
            sample_id = int(fields[col["sample_id"]])
            p_pref = float(fields[col["p_pref"]])
            p_hes = float(fields[col["p_hes"]])
            uncertainty = float(fields[col["uncertainty"]])
            scores[sample_id] = {
                "p_pref": p_pref,
                "p_hes": p_hes,
                "uncertainty": uncertainty,
            }
            uncertainties.append(uncertainty)
    return scores, uncertainties


def quantile(values, q):
    if not values:
        return float("inf")
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def parse_doublehead_line(line):
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 11:
        raise ValueError(f"Expected 11 fields, got {len(fields)}: {line[:120]}")
    return {
        "sample_id": int(fields[0]),
        "behavior": fields[5],
        "uid": fields[6],
        "mid": fields[7],
        "cat": fields[8],
        "hist_mids": fields[9],
        "hist_cats": fields[10],
    }


def format_din_line(label, record):
    return "\t".join(
        [
            str(label),
            record["uid"],
            record["mid"],
            record["cat"],
            record["hist_mids"],
            record["hist_cats"],
        ]
    )


def classify_click_only(record, score, args, uncertainty_high):
    if score is None:
        return "drop_missing_score"

    high_uncertainty = score["uncertainty"] >= uncertainty_high
    if high_uncertainty:
        return "drop_model_blind_spot"

    if score["p_pref"] >= args.pref_high and score["p_hes"] >= args.hes_high:
        return "drop_deterministic_hesitation"

    if score["p_pref"] <= args.pref_low and score["p_hes"] <= args.hes_low:
        return "negative_real_dislike"

    return "drop_ambiguous"


def copy_metadata(doublehead_dir, output_dir):
    for filename in ["uid_voc.pkl", "mid_voc.pkl", "cat_voc.pkl", "item-info", "reviews-info"]:
        src = doublehead_dir / filename
        if src.exists():
            shutil.copyfile(src, output_dir / filename)


def build_train(args, scores, uncertainty_high, summary_path):
    input_path = args.doublehead_dir / "doublehead_train.tsv"
    output_path = args.output_dir / "local_train_splitByUser"
    counts = Counter()

    with input_path.open("r", encoding="utf-8") as src, \
            output_path.open("w", encoding="utf-8") as dst, \
            summary_path.open("w", encoding="utf-8") as summary:
        src.readline()
        summary.write("sample_id\tbehavior\tdecision\tp_pref\tp_hes\tuncertainty\n")

        for line in src:
            record = parse_doublehead_line(line)
            behavior = record["behavior"]

            if behavior == "click_deep":
                dst.write(format_din_line(1, record) + "\n")
                counts["train_positive_click_deep"] += 1
                continue

            if behavior == "non_click":
                dst.write(format_din_line(0, record) + "\n")
                counts["train_negative_non_click"] += 1
                continue

            if behavior == "click_hate":
                dst.write(format_din_line(0, record) + "\n")
                counts["train_negative_click_hate"] += 1
                continue

            if behavior != "click_only":
                counts[f"train_unknown_{behavior}"] += 1
                continue

            score = scores.get(record["sample_id"])
            decision = classify_click_only(record, score, args, uncertainty_high)
            counts[decision] += 1

            if score is None:
                summary.write(f"{record['sample_id']}\t{behavior}\t{decision}\tNA\tNA\tNA\n")
            else:
                summary.write(
                    f"{record['sample_id']}\t{behavior}\t{decision}\t"
                    f"{score['p_pref']:.8f}\t{score['p_hes']:.8f}\t{score['uncertainty']:.8f}\n"
                )

            if decision == "negative_real_dislike":
                dst.write(format_din_line(0, record) + "\n")
                counts["train_negative_click_only_real_dislike"] += 1

    return counts


def build_test(args):
    input_path = args.doublehead_dir / "doublehead_test.tsv"
    output_path = args.output_dir / "local_test_splitByUser"
    counts = Counter()

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        src.readline()
        for line in src:
            record = parse_doublehead_line(line)
            behavior = record["behavior"]
            if behavior == "click_deep":
                dst.write(format_din_line(1, record) + "\n")
                counts["test_positive_click_deep"] += 1
            elif behavior == "non_click":
                dst.write(format_din_line(0, record) + "\n")
                counts["test_negative_non_click"] += 1
            else:
                counts[f"test_skip_{behavior}"] += 1

    return counts


def main():
    args = parse_args()
    scores_file = args.scores_file or args.doublehead_dir / "click_only_scores.tsv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores, uncertainties = read_scores(scores_file)
    uncertainty_high = quantile(uncertainties, args.uncertainty_high_quantile)

    print("Building Group C DIN data ...")
    print(f"  scores file: {scores_file}")
    print(f"  uncertainty high threshold: {uncertainty_high:.8f}")
    print(f"  output dir: {args.output_dir}")

    copy_metadata(args.doublehead_dir, args.output_dir)
    summary_path = args.output_dir / "groupC_click_only_decisions.tsv"
    train_counts = build_train(args, scores, uncertainty_high, summary_path)
    test_counts = build_test(args)

    print("\nDone.")
    for key in sorted(train_counts):
        print(f"  {key}: {train_counts[key]:,}")
    for key in sorted(test_counts):
        print(f"  {key}: {test_counts[key]:,}")
    print(f"  click-only decision summary: {summary_path}")


if __name__ == "__main__":
    main()
