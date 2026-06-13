import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data_iterator import load_dict
from model_doublehead import DoubleHeadDIN
from preprocess_kuairand_1k import SEQ_SEP
from utils import calc_auc


EMBEDDING_DIM = 12
HIDDEN_DIM = [108, 200, 80]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "data_doublehead"
    default_model_path = script_dir / "best_model" / "doublehead.pt"

    parser = argparse.ArgumentParser(description="Train and score the double-head click-only classifier.")
    parser.add_argument("--mode", choices=["train", "score", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--uid-voc", type=Path, default=None)
    parser.add_argument("--mid-voc", type=Path, default=None)
    parser.add_argument("--cat-voc", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--mc-samples", type=int, default=20)
    return parser.parse_args()


class DoubleHeadDataIterator:
    def __init__(self, source, uid_voc, mid_voc, cat_voc, batch_size=128):
        self.source_path = source
        self.source = open(source, "r", encoding="utf-8")
        self.header = self.source.readline()
        self.source_dicts = [load_dict(uid_voc), load_dict(mid_voc), load_dict(cat_voc)]
        self.batch_size = batch_size
        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def reset(self):
        self.source.seek(0)
        self.header = self.source.readline()

    def next(self):
        records = []
        while len(records) < self.batch_size:
            line = self.source.readline()
            if line == "":
                break
            records.append(parse_record(line.rstrip("\n"), self.source_dicts))

        if not records:
            self.reset()
            raise StopIteration
        return records


def map_id(value, vocab):
    key = value.encode("UTF-8")
    return vocab[key] if key in vocab else 0


def parse_record(line, vocabs):
    fields = line.split("\t")
    if len(fields) != 11:
        raise ValueError(f"Expected 11 fields, got {len(fields)}: {line[:120]}")

    hist_mids = [map_id(x, vocabs[1]) for x in fields[9].split(SEQ_SEP) if x]
    hist_cats = [map_id(x, vocabs[2]) for x in fields[10].split(SEQ_SEP) if x]
    return {
        "sample_id": int(fields[0]),
        "pref_label": float(fields[1]),
        "pref_mask": float(fields[2]),
        "hes_label": float(fields[3]),
        "hes_mask": float(fields[4]),
        "behavior": fields[5],
        "uid": map_id(fields[6], vocabs[0]),
        "mid": map_id(fields[7], vocabs[1]),
        "cat": map_id(fields[8], vocabs[2]),
        "hist_mids": hist_mids,
        "hist_cats": hist_cats,
    }


def prepare_batch(records, maxlen):
    lengths = [len(r["hist_mids"]) for r in records]
    maxlen_x = min(max(lengths), maxlen)
    batch_size = len(records)

    mid_his = np.zeros((batch_size, maxlen_x), dtype="int64")
    cat_his = np.zeros((batch_size, maxlen_x), dtype="int64")
    mid_mask = np.zeros((batch_size, maxlen_x), dtype="float32")

    for idx, record in enumerate(records):
        mids = record["hist_mids"][-maxlen:]
        cats = record["hist_cats"][-maxlen:]
        seq_len = min(len(mids), maxlen_x)
        mid_his[idx, :seq_len] = mids[-seq_len:]
        cat_his[idx, :seq_len] = cats[-seq_len:]
        mid_mask[idx, :seq_len] = 1.0

    batch = {
        "sample_ids": np.array([r["sample_id"] for r in records], dtype="int64"),
        "behaviors": [r["behavior"] for r in records],
        "uids": np.array([r["uid"] for r in records], dtype="int64"),
        "mids": np.array([r["mid"] for r in records], dtype="int64"),
        "cats": np.array([r["cat"] for r in records], dtype="int64"),
        "mid_his": mid_his,
        "cat_his": cat_his,
        "mid_mask": mid_mask,
        "pref_label": np.array([r["pref_label"] for r in records], dtype="float32"),
        "pref_mask": np.array([r["pref_mask"] for r in records], dtype="float32"),
        "hes_label": np.array([r["hes_label"] for r in records], dtype="float32"),
        "hes_mask": np.array([r["hes_mask"] for r in records], dtype="float32"),
    }
    return batch


def to_device(batch):
    tensor_keys = [
        "uids",
        "mids",
        "cats",
        "mid_his",
        "cat_his",
        "mid_mask",
        "pref_label",
        "pref_mask",
        "hes_label",
        "hes_mask",
    ]
    return {key: torch.from_numpy(batch[key]).to(device) for key in tensor_keys}


def masked_bce(logits, labels, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return torch.sum(loss * mask) / torch.clamp(torch.sum(mask), min=1.0)


def safe_auc(raw_arr):
    if not raw_arr:
        return None
    labels = {record[1] for record in raw_arr}
    if len(labels) < 2:
        return None
    return calc_auc(raw_arr)


def run_eval(data, model, maxlen):
    data.reset()
    model.eval()
    pref_auc_arr = []
    hes_auc_arr = []
    loss_sum = 0.0
    batches = 0

    with torch.no_grad():
        while True:
            try:
                records = data.next()
            except StopIteration:
                break

            raw_batch = prepare_batch(records, maxlen)
            batch = to_device(raw_batch)
            pref_logits, hes_logits = model(
                batch["uids"],
                batch["mids"],
                batch["cats"],
                batch["mid_his"],
                batch["cat_his"],
                batch["mid_mask"],
            )
            pref_loss = masked_bce(pref_logits, batch["pref_label"], batch["pref_mask"])
            hes_loss = masked_bce(hes_logits, batch["hes_label"], batch["hes_mask"])
            loss_sum += (pref_loss + hes_loss).item()
            batches += 1

            pref_probs = torch.sigmoid(pref_logits).detach().cpu().numpy()
            hes_probs = torch.sigmoid(hes_logits).detach().cpu().numpy()
            for prob, label, mask in zip(pref_probs, raw_batch["pref_label"], raw_batch["pref_mask"]):
                if mask > 0:
                    pref_auc_arr.append([float(prob), float(label)])
            for prob, label, mask in zip(hes_probs, raw_batch["hes_label"], raw_batch["hes_mask"]):
                if mask > 0:
                    hes_auc_arr.append([float(prob), float(label)])

    return {
        "loss": loss_sum / max(1, batches),
        "pref_auc": safe_auc(pref_auc_arr),
        "hes_auc": safe_auc(hes_auc_arr),
    }


def train(args):
    train_file = args.train_file or args.data_dir / "doublehead_train.tsv"
    test_file = args.test_file or args.data_dir / "doublehead_test.tsv"
    uid_voc = args.uid_voc or args.data_dir / "uid_voc.pkl"
    mid_voc = args.mid_voc or args.data_dir / "mid_voc.pkl"
    cat_voc = args.cat_voc or args.data_dir / "cat_voc.pkl"

    train_data = DoubleHeadDataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size=args.batch_size)
    test_data = DoubleHeadDataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size=args.batch_size)
    n_uid, n_mid, n_cat = train_data.get_n()

    model = DoubleHeadDIN(
        n_uid=n_uid,
        n_mid=n_mid,
        n_cat=n_cat,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=args.dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    os.makedirs(args.model_path.parent, exist_ok=True)
    best_score = -float("inf")

    for epoch in range(1, args.epochs + 1):
        train_data.reset()
        model.train()
        iter_num = 0
        loss_sum = 0.0

        while True:
            try:
                records = train_data.next()
            except StopIteration:
                break

            raw_batch = prepare_batch(records, args.maxlen)
            batch = to_device(raw_batch)

            optimizer.zero_grad()
            pref_logits, hes_logits = model(
                batch["uids"],
                batch["mids"],
                batch["cats"],
                batch["mid_his"],
                batch["cat_his"],
                batch["mid_mask"],
            )
            pref_loss = masked_bce(pref_logits, batch["pref_label"], batch["pref_mask"])
            hes_loss = masked_bce(hes_logits, batch["hes_label"], batch["hes_mask"])
            loss = pref_loss + hes_loss
            loss.backward()
            optimizer.step()

            iter_num += 1
            loss_sum += loss.item()
            if iter_num % 500 == 0:
                print(f"epoch {epoch} iter {iter_num}, train loss {loss_sum / 500:.4f}")
                loss_sum = 0.0

        metrics = run_eval(test_data, model, args.maxlen)
        pref_auc = metrics["pref_auc"]
        hes_auc = metrics["hes_auc"]
        print(
            f"epoch {epoch} eval loss {metrics['loss']:.4f}, "
            f"pref_auc {pref_auc if pref_auc is not None else 'NA'}, "
            f"hes_auc {hes_auc if hes_auc is not None else 'NA'}"
        )

        score = pref_auc if pref_auc is not None else -metrics["loss"]
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, args.model_path)


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def score_click_only(args):
    train_file = args.train_file or args.data_dir / "doublehead_train.tsv"
    uid_voc = args.uid_voc or args.data_dir / "uid_voc.pkl"
    mid_voc = args.mid_voc or args.data_dir / "mid_voc.pkl"
    cat_voc = args.cat_voc or args.data_dir / "cat_voc.pkl"
    scores_file = args.scores_file or args.data_dir / "click_only_scores.tsv"

    data = DoubleHeadDataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size=args.batch_size)
    n_uid, n_mid, n_cat = data.get_n()
    model = DoubleHeadDIN(
        n_uid=n_uid,
        n_mid=n_mid,
        n_cat=n_cat,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=args.dropout_rate,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    enable_dropout(model)

    scores_file.parent.mkdir(parents=True, exist_ok=True)
    data.reset()
    with scores_file.open("w", encoding="utf-8") as f:
        f.write("sample_id\tp_pref\tp_hes\tuncertainty\tpref_var\thes_var\tbehavior\n")
        while True:
            try:
                records = data.next()
            except StopIteration:
                break

            raw_batch = prepare_batch(records, args.maxlen)
            click_only_idx = [idx for idx, behavior in enumerate(raw_batch["behaviors"]) if behavior == "click_only"]
            if not click_only_idx:
                continue

            batch = to_device(raw_batch)
            pref_samples = []
            hes_samples = []
            with torch.no_grad():
                for _ in range(args.mc_samples):
                    pref_logits, hes_logits = model(
                        batch["uids"],
                        batch["mids"],
                        batch["cats"],
                        batch["mid_his"],
                        batch["cat_his"],
                        batch["mid_mask"],
                    )
                    pref_samples.append(torch.sigmoid(pref_logits).detach().cpu().numpy())
                    hes_samples.append(torch.sigmoid(hes_logits).detach().cpu().numpy())

            pref_samples = np.stack(pref_samples, axis=0)
            hes_samples = np.stack(hes_samples, axis=0)
            pref_mean = pref_samples.mean(axis=0)
            hes_mean = hes_samples.mean(axis=0)
            pref_var = pref_samples.var(axis=0)
            hes_var = hes_samples.var(axis=0)

            for idx in click_only_idx:
                sample_id = int(raw_batch["sample_ids"][idx])
                f.write(
                    f"{sample_id}\t"
                    f"{pref_mean[idx]:.8f}\t"
                    f"{hes_mean[idx]:.8f}\t"
                    f"{pref_var[idx]:.8f}\t"
                    f"{pref_var[idx]:.8f}\t"
                    f"{hes_var[idx]:.8f}\t"
                    f"click_only\n"
                )

    print(f"Wrote click-only scores to {scores_file}")


def main():
    args = parse_args()
    if args.mode in ["train", "all"]:
        train(args)
    if args.mode in ["score", "all"]:
        score_click_only(args)


if __name__ == "__main__":
    main()
