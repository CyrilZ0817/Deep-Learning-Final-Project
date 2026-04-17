import argparse
import ast
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_training_log(filepath):
    metrics = {"loss": [], "grad_norm": [], "learning_rate": [], "epoch": []}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Match dict-like log lines
            match = re.match(r"^\{.*\}$", line)
            if not match:
                continue
            try:
                # Safely parse the dict (values are strings, so use ast.literal_eval)
                entry = ast.literal_eval(line)
                for key in metrics:
                    if key in entry:
                        metrics[key].append(float(entry[key]))
            except (ValueError, SyntaxError):
                continue

    return metrics


def plot_metrics(metrics, output_path):
    epochs = metrics["epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Fine-Tuning Training Metrics", fontsize=16, fontweight="bold", y=0.98)

    # --- Loss ---
    ax1 = axes[0]
    ax1.plot(epochs, metrics["loss"], color="#e05c5c", linewidth=2, marker="o", markersize=4, label="Loss")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # --- Grad Norm ---
    ax2 = axes[1]
    ax2.plot(epochs, metrics["grad_norm"], color="#5c9ee0", linewidth=2, marker="s", markersize=4, label="Grad Norm")
    ax2.set_ylabel("Gradient Norm", fontsize=12)
    ax2.set_title("Gradient Norm", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # --- Learning Rate ---
    ax3 = axes[2]
    ax3.plot(epochs, metrics["learning_rate"], color="#5ce07a", linewidth=2, marker="^", markersize=4, label="Learning Rate")
    ax3.set_ylabel("Learning Rate", fontsize=12)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_title("Learning Rate Schedule", fontsize=13)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from a .out log file.")
    parser.add_argument("-f", "--file", required=True, help="Path to the training output file (e.g. train.out)")
    args = parser.parse_args()

    filepath = os.path.abspath(args.file)
    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}")
        return

    metrics = parse_training_log(filepath)

    if not metrics["epoch"]:
        print("No valid training log entries found in the file.")
        return

    output_dir = os.path.dirname(filepath)
    output_path = os.path.join(output_dir, "loss_graph.png")

    plot_metrics(metrics, output_path)


if __name__ == "__main__":
    main()