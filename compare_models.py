#!/usr/bin/env python3
"""
Compare fine-tuned Qwen3-VL-8B (LoRA) vs base model on strawberry QA.

Runs locally on Apple Silicon using mlx-vlm (4-bit quantized).

Usage:
    # 1. First convert your PEFT adapter to MLX format:
    python convert_adapter_to_mlx.py \
        --peft_path output/v2-20260308-003328/checkpoint-861

    # 2. Run the comparison:
    python compare_models.py \
        --adapter_path output/v2-20260308-003328/checkpoint-861-mlx \
        --val_data qa_pipeline/output/val.jsonl \
        --crops_dir qa_pipeline/crops \
        --output_dir comparison_results \
        --num_samples 30
"""

import argparse
import gc
import json
import os
import re
import random
import time
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Ripeness helpers
# ---------------------------------------------------------------------------
RIPENESS_CATEGORIES = [
    "green", "white", "turning", "light red",
    "nearly ripe", "ripe", "fully ripe", "overripe",
]

HARVEST_KEYWORDS = {
    "harvest_now": ["ready to harvest", "harvest now", "harvest immediately",
                    "ready for harvest", "ready for picking", "can be harvested",
                    "pick now", "harvest today"],
    "wait":        ["wait", "not yet ready", "not ready", "needs more time",
                    "allow more time", "requires more time", "too early",
                    "not yet mature", "not yet ripe"],
}

HEALTH_KEYWORDS = {
    "healthy":  ["healthy", "no visible signs", "no signs of disease",
                 "no visible disease", "appears healthy", "no abnormalities",
                 "no fungal", "no rot"],
    "diseased": ["powdery mildew", "grey mold", "gray mold", "botrytis",
                 "anthracnose", "leaf spot", "rot", "lesion", "fungal",
                 "infection", "disease", "mold"],
}


def extract_ripeness(text: str) -> str:
    text_lower = text.lower()
    for cat in ["fully ripe", "nearly ripe", "light red", "overripe",
                "turning", "green", "white", "ripe"]:
        if cat in text_lower:
            if cat in ("nearly ripe", "light red"):
                return "nearly ripe"
            return cat
    return "unknown"


def extract_harvest_rec(text: str) -> str:
    text_lower = text.lower()
    for kw in HARVEST_KEYWORDS["harvest_now"]:
        if kw in text_lower:
            return "harvest_now"
    for kw in HARVEST_KEYWORDS["wait"]:
        if kw in text_lower:
            return "wait"
    return "unknown"


def extract_health(text: str) -> str:
    text_lower = text.lower()
    for kw in HEALTH_KEYWORDS["diseased"]:
        if kw in text_lower:
            pattern = rf"no\s+(?:visible\s+)?(?:signs?\s+of\s+)?{re.escape(kw)}"
            if re.search(pattern, text_lower):
                continue
            return "diseased"
    for kw in HEALTH_KEYWORDS["healthy"]:
        if kw in text_lower:
            return "healthy"
    return "unknown"


def score_structure(text: str) -> float:
    text_lower = text.lower()
    found = 0
    if any(w in text_lower for w in ["ripeness", "ripe", "unripe", "green", "mature"]):
        found += 1
    if any(w in text_lower for w in ["color", "rgb", "red", "green", "white",
                                      "pink", "crimson", "yellow"]):
        found += 1
    if any(w in text_lower for w in ["health", "disease", "mildew", "mold",
                                      "healthy", "infection", "rot"]):
        found += 1
    if any(w in text_lower for w in ["harvest", "pick", "wait", "ready"]):
        found += 1
    return found / 4.0


def rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_f1(reference: str, hypothesis: str) -> float:
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "out", "off",
            "over", "under", "again", "further", "then", "once", "and", "but", "or",
            "nor", "not", "so", "yet", "both", "either", "neither", "each", "every",
            "all", "any", "few", "more", "most", "other", "some", "such", "no",
            "only", "own", "same", "than", "too", "very", "just", "because",
            "this", "that", "these", "those", "it", "its", "i", "me", "my",
            "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "they", "them", "their", "what", "which", "who", "whom"}
    ref_words = set(reference.lower().split()) - stop
    hyp_words = set(hypothesis.lower().split()) - stop
    if not ref_words or not hyp_words:
        return 0.0
    overlap = ref_words & hyp_words
    precision = len(overlap) / len(hyp_words)
    recall = len(overlap) / len(ref_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# MLX-VLM inference
# ---------------------------------------------------------------------------
def load_model(model_path: str, adapter_path: str = None):
    """Load model via mlx-vlm, optionally with LoRA adapter."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    label = "fine-tuned" if adapter_path else "base"
    print(f"Loading {label} model: {model_path}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")

    model, processor = load(model_path, adapter_path=adapter_path)
    config = load_config(model_path)
    return model, processor, config


def run_inference(model, processor, config, image_path: str, question: str,
                  system_prompt: str, max_tokens: int = 512) -> str:
    """Run a single VLM inference with mlx-vlm."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # Build message list for chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    formatted_prompt = apply_chat_template(
        processor, config, messages, num_images=1
    )

    result = generate(
        model, processor, formatted_prompt,
        image=[image_path],
        max_tokens=max_tokens,
        verbose=False,
    )

    # generate() returns a GenerationResult; extract text
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", str(result))
    return text.strip()


def unload_model():
    """Free MLX model memory."""
    import mlx.core as mx
    gc.collect()
    mx.metal.clear_cache()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate_model(model, processor, config, samples, crops_dir: str, label: str):
    results = []
    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        system_prompt = msgs[0]["content"]
        question = msgs[1]["content"].replace("<image>", "").strip()
        ground_truth = msgs[2]["content"]
        image_rel = sample["images"][0]

        # Resolve image path
        if os.path.isabs(image_rel):
            image_path = image_rel
        else:
            img_name = image_rel.replace("crops/", "", 1) if image_rel.startswith("crops/") else image_rel
            image_path = os.path.join(crops_dir, img_name)

        if not os.path.exists(image_path):
            print(f"  [{label}] SKIP {i+1}/{len(samples)}: image not found: {image_path}")
            continue

        t0 = time.time()
        try:
            prediction = run_inference(model, processor, config,
                                       image_path, question, system_prompt)
        except Exception as e:
            print(f"  [{label}] ERROR {i+1}/{len(samples)}: {e}")
            prediction = ""
        elapsed = time.time() - t0

        gt_ripeness = extract_ripeness(ground_truth)
        pred_ripeness = extract_ripeness(prediction)
        gt_harvest = extract_harvest_rec(ground_truth)
        pred_harvest = extract_harvest_rec(prediction)
        gt_health = extract_health(ground_truth)
        pred_health = extract_health(prediction)

        r = {
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "image": image_path,
            "inference_time": elapsed,
            "gt_ripeness": gt_ripeness,
            "pred_ripeness": pred_ripeness,
            "ripeness_correct": gt_ripeness == pred_ripeness,
            "gt_harvest": gt_harvest,
            "pred_harvest": pred_harvest,
            "harvest_correct": gt_harvest == pred_harvest,
            "gt_health": gt_health,
            "pred_health": pred_health,
            "health_correct": gt_health == pred_health,
            "structure_score": score_structure(prediction),
            "rouge_l": rouge_l(ground_truth, prediction),
            "keyword_f1": keyword_f1(ground_truth, prediction),
            "answer_length": len(prediction.split()),
        }
        results.append(r)

        status = "correct" if r["ripeness_correct"] else "wrong"
        print(f"  [{label}] {i+1}/{len(samples)} "
              f"ripeness={pred_ripeness}({status}) "
              f"rouge={r['rouge_l']:.2f} "
              f"time={elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(base_results, ft_results, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    def agg(results):
        if not results:
            return {}
        return {
            "ripeness_acc": np.mean([r["ripeness_correct"] for r in results]),
            "harvest_acc": np.mean([r["harvest_correct"] for r in results]),
            "health_acc": np.mean([r["health_correct"] for r in results]),
            "structure_score": np.mean([r["structure_score"] for r in results]),
            "rouge_l": np.mean([r["rouge_l"] for r in results]),
            "keyword_f1": np.mean([r["keyword_f1"] for r in results]),
            "avg_length": np.mean([r["answer_length"] for r in results]),
            "avg_time": np.mean([r["inference_time"] for r in results]),
        }

    base_agg = agg(base_results)
    ft_agg = agg(ft_results)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"base": "#FF6B6B", "finetuned": "#4ECDC4"}

    # ── Plot 1: Main accuracy metrics ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["ripeness_acc", "harvest_acc", "health_acc"]
    labels = ["Ripeness\nClassification", "Harvest\nRecommendation", "Health\nAssessment"]
    x = np.arange(len(metrics))
    width = 0.35

    base_vals = [base_agg[m] * 100 for m in metrics]
    ft_vals = [ft_agg[m] * 100 for m in metrics]

    bars1 = ax.bar(x - width/2, base_vals, width, label="Base Qwen3-VL-8B",
                   color=colors["base"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-tuned (LoRA)",
                   color=colors["finetuned"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy: Base vs Fine-tuned", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Text quality metrics ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics2 = ["rouge_l", "keyword_f1", "structure_score"]
    labels2 = ["ROUGE-L", "Keyword F1", "Answer\nCompleteness"]
    x2 = np.arange(len(metrics2))

    base_vals2 = [base_agg[m] * 100 for m in metrics2]
    ft_vals2 = [ft_agg[m] * 100 for m in metrics2]

    bars1 = ax.bar(x2 - width/2, base_vals2, width, label="Base Qwen3-VL-8B",
                   color=colors["base"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x2 + width/2, ft_vals2, width, label="Fine-tuned (LoRA)",
                   color=colors["finetuned"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Answer Quality: Base vs Fine-tuned", fontsize=14, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "quality_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: Overall radar chart ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_metrics = ["ripeness_acc", "harvest_acc", "health_acc",
                     "rouge_l", "keyword_f1", "structure_score"]
    radar_labels = ["Ripeness Acc", "Harvest Acc", "Health Acc",
                    "ROUGE-L", "Keyword F1", "Completeness"]

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    base_radar = [base_agg[m] for m in radar_metrics] + [base_agg[radar_metrics[0]]]
    ft_radar = [ft_agg[m] for m in radar_metrics] + [ft_agg[radar_metrics[0]]]

    ax.plot(angles, base_radar, "o-", linewidth=2, color=colors["base"],
            label="Base Qwen3-VL-8B")
    ax.fill(angles, base_radar, alpha=0.15, color=colors["base"])
    ax.plot(angles, ft_radar, "o-", linewidth=2, color=colors["finetuned"],
            label="Fine-tuned (LoRA)")
    ax.fill(angles, ft_radar, alpha=0.15, color=colors["finetuned"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Model Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "radar_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Plot 4: Ripeness category breakdown ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (results, label, color) in enumerate([
        (base_results, "Base Qwen3-VL-8B", colors["base"]),
        (ft_results, "Fine-tuned (LoRA)", colors["finetuned"]),
    ]):
        gt_cats = Counter(r["gt_ripeness"] for r in results)
        correct_cats = Counter(r["gt_ripeness"] for r in results if r["ripeness_correct"])

        cats = sorted(gt_cats.keys())
        totals = [gt_cats[c] for c in cats]
        corrects = [correct_cats.get(c, 0) for c in cats]

        x_pos = np.arange(len(cats))
        axes[idx].bar(x_pos, totals, color="#DDDDDD", edgecolor="white", label="Total")
        axes[idx].bar(x_pos, corrects, color=color, edgecolor="white", label="Correct")
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Count")
        axes[idx].legend(fontsize=9)

    fig.suptitle("Ripeness Classification by Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ripeness_breakdown.png"), dpi=150)
    plt.close(fig)

    # ── Plot 5: Per-sample ROUGE-L scatter ────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = range(len(base_results))
    base_rouges = [r["rouge_l"] for r in base_results]
    ft_rouges = [r["rouge_l"] for r in ft_results]

    ax.scatter(indices, base_rouges, alpha=0.5, s=30, color=colors["base"],
               label=f"Base (avg={np.mean(base_rouges):.3f})")
    ax.scatter(indices, ft_rouges, alpha=0.5, s=30, color=colors["finetuned"],
               label=f"Fine-tuned (avg={np.mean(ft_rouges):.3f})")
    ax.axhline(y=np.mean(base_rouges), color=colors["base"], linestyle="--",
               alpha=0.7, linewidth=1)
    ax.axhline(y=np.mean(ft_rouges), color=colors["finetuned"], linestyle="--",
               alpha=0.7, linewidth=1)

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("ROUGE-L Score", fontsize=12)
    ax.set_title("Per-Sample ROUGE-L: Base vs Fine-tuned", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "rouge_scatter.png"), dpi=150)
    plt.close(fig)

    # ── Plot 6: Summary table as image ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table_data = [
        ["Metric", "Base Qwen3-VL-8B", "Fine-tuned (LoRA)", "Delta"],
    ]
    for metric, label, fmt in [
        ("ripeness_acc", "Ripeness Accuracy", "{:.1%}"),
        ("harvest_acc", "Harvest Rec. Accuracy", "{:.1%}"),
        ("health_acc", "Health Assessment Acc.", "{:.1%}"),
        ("rouge_l", "ROUGE-L", "{:.3f}"),
        ("keyword_f1", "Keyword F1", "{:.3f}"),
        ("structure_score", "Answer Completeness", "{:.1%}"),
        ("avg_length", "Avg Answer Length (words)", "{:.0f}"),
        ("avg_time", "Avg Inference Time (s)", "{:.2f}"),
    ]:
        b = base_agg[metric]
        f = ft_agg[metric]
        delta = f - b
        sign = "+" if delta >= 0 else ""
        table_data.append([
            label,
            fmt.format(b),
            fmt.format(f),
            f"{sign}{fmt.format(delta)}",
        ])

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.32, 0.22, 0.22, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(4):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(table_data)):
        delta_text = table_data[i][3]
        if delta_text.startswith("+"):
            table[i, 3].set_facecolor("#D5F5E3")
        elif delta_text.startswith("-"):
            table[i, 3].set_facecolor("#FADBD8")

    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")
    return base_agg, ft_agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare fine-tuned vs base Qwen3-VL on strawberry QA (Apple Silicon)"
    )
    parser.add_argument("--model_name", default="mlx-community/Qwen3-VL-8B-Instruct-4bit",
                        help="Base model (MLX 4-bit quantized)")
    parser.add_argument("--adapter_path", required=True,
                        help="Path to MLX-format LoRA adapter directory")
    parser.add_argument("--val_data", required=True,
                        help="Path to val.jsonl")
    parser.add_argument("--crops_dir", required=True,
                        help="Path to crops/ directory with images")
    parser.add_argument("--output_dir", default="./comparison_results",
                        help="Where to save results and plots")
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of val samples to evaluate (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load validation data ──────────────────────────────────
    print("Loading validation data...")
    with open(args.val_data) as f:
        all_samples = [json.loads(line) for line in f]
    print(f"  Total validation samples: {len(all_samples)}")

    if args.num_samples > 0 and args.num_samples < len(all_samples):
        samples = random.sample(all_samples, args.num_samples)
        print(f"  Sampled {args.num_samples} for evaluation")
    else:
        samples = all_samples
        print(f"  Using all {len(all_samples)} samples")

    # ── Phase 1: Base model ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1: Evaluating BASE model")
    print("=" * 60)

    base_model, base_proc, base_config = load_model(args.model_name)

    base_results = evaluate_model(
        base_model, base_proc, base_config, samples, args.crops_dir, "BASE"
    )

    # Free memory before loading fine-tuned model
    del base_model, base_proc, base_config
    unload_model()
    print("  Base model unloaded.")

    # ── Phase 2: Fine-tuned model ─────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 2: Evaluating FINE-TUNED model")
    print("=" * 60)

    ft_model, ft_proc, ft_config = load_model(args.model_name, args.adapter_path)

    ft_results = evaluate_model(
        ft_model, ft_proc, ft_config, samples, args.crops_dir, "FT"
    )

    del ft_model, ft_proc, ft_config
    unload_model()

    # ── Generate plots ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating comparison plots")
    print("=" * 60)
    base_agg, ft_agg = generate_plots(base_results, ft_results, args.output_dir)

    # ── Save raw results ──────────────────────────────────────
    results_path = os.path.join(args.output_dir, "raw_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": vars(args),
            "base_aggregate": base_agg,
            "finetuned_aggregate": ft_agg,
            "base_results": base_results,
            "finetuned_results": ft_results,
        }, f, indent=2, default=str)
    print(f"Raw results saved to {results_path}")

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<28} {'Base':>10} {'Fine-tuned':>10} {'Delta':>10}")
    print("  " + "-" * 58)
    for metric, label, fmt in [
        ("ripeness_acc", "Ripeness Accuracy", "{:.1%}"),
        ("harvest_acc", "Harvest Rec. Accuracy", "{:.1%}"),
        ("health_acc", "Health Assessment Acc.", "{:.1%}"),
        ("rouge_l", "ROUGE-L", "{:.3f}"),
        ("keyword_f1", "Keyword F1", "{:.3f}"),
        ("structure_score", "Answer Completeness", "{:.1%}"),
        ("avg_length", "Avg Length (words)", "{:.0f}"),
        ("avg_time", "Avg Inference Time (s)", "{:.2f}"),
    ]:
        b = base_agg[metric]
        f_ = ft_agg[metric]
        d = f_ - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<28} {fmt.format(b):>10} {fmt.format(f_):>10} "
              f"{sign}{fmt.format(d):>9}")

    print("\n  Plots saved to: " + args.output_dir)
    print("  Done!")


if __name__ == "__main__":
    main()