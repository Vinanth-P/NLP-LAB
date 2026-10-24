"""Train the four Hindi POS taggers and evaluate them on dev and test."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from pos_models import POSModels, load_sents, prf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
METHODS = ("naive_bayes", "hmm_forward", "hmm_backward", "hmm_forward_backward")
LABELS = {
    "naive_bayes": "Naive Bayes",
    "hmm_forward": "HMM forward",
    "hmm_backward": "HMM backward",
    "hmm_forward_backward": "HMM fwd-bwd",
}


def evaluate(model: POSModels, sents) -> dict:
    gold = {m: [] for m in METHODS}
    pred = {m: [] for m in METHODS}
    for sent in sents:
        words = [w for w, _ in sent]
        decoded = model.decode(words)
        g = [t for _, t in sent]
        for m in METHODS:
            gold[m].extend(g)
            pred[m].extend(decoded[m])
    return {m: prf(gold[m], pred[m], model.tags) for m in METHODS}


def self_check(model: POSModels) -> None:
    """Forward Viterbi must match an exhaustive search on a short sentence."""
    words = ["एक", "बड़ा", "मकान"]
    rows = model._emit_rows(words)
    best, best_seq = -1e300, None
    for seq in itertools.product(range(model.T), repeat=len(words)):
        score = model.log_tfwd[model.T, seq[0]] + rows[0, seq[0]]
        for i in range(1, len(words)):
            score += model.log_tfwd[seq[i - 1], seq[i]] + rows[i, seq[i]]
        score += model.log_end[seq[-1]]
        if score > best:
            best, best_seq = score, seq
    got = tuple(model.hmm_forward(words))
    if got != best_seq:
        raise SystemExit(f"Viterbi self-check failed: {got} != {best_seq}")
    print("Forward Viterbi matches exhaustive search on a 3-word sentence.")


def write_svg(path: Path, tags: list[str], series: list[tuple[str, list[float]]], title: str) -> None:
    colors = ["#1f4e79", "#c45911", "#548235", "#7030a0"]
    row_h, left, top, plot_w, right = 22, 150, 46, 640, 24
    height = top + row_h * len(tags) + 36
    width = left + plot_w + right
    n = len(series)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="16">{title}</text>',
    ]
    max_v = max(max(vals) for _, vals in series) or 1
    max_v = min(1.0, max_v * 1.05) if max_v <= 1 else max_v
    for i, tag in enumerate(tags):
        y = top + i * row_h
        parts.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" font-size="11">{tag}</text>')
        bw = (row_h - 6) / n
        for s, (name, vals) in enumerate(series):
            length = 0 if max_v == 0 else plot_w * vals[i] / max_v
            parts.append(
                f'<rect x="{left}" y="{y + 3 + s * bw:.1f}" width="{length:.1f}" '
                f'height="{bw - 1:.1f}" fill="{colors[s % len(colors)]}"/>'
            )
    legend_y = height - 16
    x = left
    for s, (name, _) in enumerate(series):
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="12" height="12" fill="{colors[s % len(colors)]}"/>')
        parts.append(f'<text x="{x + 16}" y="{legend_y + 2}" font-size="11">{name}</text>')
        x += 130
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_summary_svg(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """rows: (label, dev accuracy, test accuracy) as percentages."""
    colors = ("#1f4e79", "#c45911")
    width, height = 760, 380
    left, right, top, bottom = 70, 20, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="Helvetica, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="16">'
        "Token accuracy on dev and test</text>",
    ]
    gap = plot_w / len(rows)
    for i, (label, dev, test) in enumerate(rows):
        x = left + i * gap + gap * 0.2
        bw = gap * 0.25
        for s, val in enumerate((dev, test)):
            h = plot_h * val / 100
            y = top + plot_h - h
            parts.append(
                f'<rect x="{x + s * (bw + 6):.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                f'fill="{colors[s]}"/>'
            )
            parts.append(
                f'<text x="{x + s * (bw + 6) + bw / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-size="10">{val:.1f}</text>'
            )
        parts.append(
            f'<text x="{x + bw:.1f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11">{label}</text>'
        )
    parts.append(f'<rect x="{left}" y="{height - 28}" width="12" height="12" fill="{colors[0]}"/>')
    parts.append(f'<text x="{left + 18}" y="{height - 17}" font-size="12">Dev</text>')
    parts.append(f'<rect x="{left + 70}" y="{height - 28}" width="12" height="12" fill="{colors[1]}"/>')
    parts.append(f'<text x="{left + 88}" y="{height - 17}" font-size="12">Test</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    print("Loading data...", flush=True)
    train = load_sents(DATA / "hindi-train-pos.txt")
    dev = load_sents(DATA / "hindi-dev-pos.txt")
    test = load_sents(DATA / "hindi-test-pos.txt")
    print(f"  train {len(train):,}  dev {len(dev):,}  test {len(test):,} sentences", flush=True)
    model = POSModels(train)
    print(f"  {model.T} tags, {model.V:,} word types (including <UNK>, <s>, </s>)", flush=True)
    self_check(model)

    print("Evaluating dev...", flush=True)
    dev_m = evaluate(model, dev)
    print("Evaluating test...", flush=True)
    test_m = evaluate(model, test)

    by_freq = np.argsort(-model.nb_tag)
    tag_order = [model.tags[i] for i in by_freq]
    lines = [
        f"{'Method':<22}{'Dev acc':>10}{'Dev macroF1':>14}{'Test acc':>10}{'Test macroF1':>14}",
        "-" * 70,
    ]
    summary_rows = []
    for m in METHODS:
        d, t = dev_m[m], test_m[m]
        lines.append(
            f"{LABELS[m]:<22}{d['accuracy'] * 100:9.2f}%{d['macro_f1'] * 100:13.2f}%"
            f"{t['accuracy'] * 100:9.2f}%{t['macro_f1'] * 100:13.2f}%"
        )
        summary_rows.append((LABELS[m], d["accuracy"] * 100, t["accuracy"] * 100))
        print(lines[-1], flush=True)
    table = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pos_results_table.txt").write_text(table)

    per_tag_lines = ["F1 per tag on the test set (percent)", ""]
    header = f"{'Tag':<12}" + "".join(f"{LABELS[m]:>16}" for m in METHODS) + f"{'Support':>10}"
    per_tag_lines += [header, "-" * len(header)]
    for tag in tag_order:
        cells = [f"{test_m[m]['per_tag'][tag]['f1'] * 100:15.2f}" for m in METHODS]
        support = test_m["hmm_forward"]["per_tag"][tag]["support"]
        per_tag_lines.append(f"{tag:<12}" + "".join(cells) + f"{support:10d}")
    (OUT / "pos_f1_per_tag.txt").write_text("\n".join(per_tag_lines) + "\n")

    payload = {
        "train_sentences": len(train),
        "dev_sentences": len(dev),
        "test_sentences": len(test),
        "tags": model.tags,
        "smoothing": "add-one",
        "naive_bayes_features": [
            "length MORE if len(word)>4 else LESS",
            "word",
            "previous word",
            "next word",
            "previous tag (predicted left to right at test time)",
        ],
        "dev": dev_m,
        "test": test_m,
    }
    (OUT / "pos_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    series = []
    for m in METHODS:
        series.append((LABELS[m], [test_m[m]["per_tag"][tag]["f1"] for tag in tag_order]))
    write_svg(OUT / "f1_per_tag_test.svg", tag_order, series, "Test F1 by tag")
    write_summary_svg(OUT / "accuracy_comparison.svg", summary_rows)

    # A short tagged sample so the decode is visible.
    sample = test[0]
    words = [w for w, _ in sample]
    decoded = model.decode(words)
    with (OUT / "sample_sentence.txt").open("w", encoding="utf-8") as f:
        f.write("word\tgold\t" + "\t".join(LABELS[m] for m in METHODS) + "\n")
        for i, (w, g) in enumerate(sample):
            cols = "\t".join(decoded[m][i] for m in METHODS)
            f.write(f"{w}\t{g}\t{cols}\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
