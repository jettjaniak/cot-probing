import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_histogram(accuracies, title, color="orange"):
    """Plot a histogram of accuracies with median line."""
    plt.figure(figsize=(8, 6))
    plt.hist(accuracies, bins=20, alpha=0.7, color=color)
    plt.axvline(
        x=np.median(accuracies),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(accuracies):.2f}",
    )
    plt.title(title)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_binary_histogram(labels, title, colors=["#2ecc71", "#e74c3c"]):
    """Plot a histogram for binary labels with custom colors."""
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(labels, bins=2, alpha=0.7)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_triple_histogram(labels, title, colors=["#2ecc71", "#e74c3c", "#3498db"]):
    """Plot a histogram for three-category labels with custom colors."""
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(labels, bins=3, alpha=0.7)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy_difference_histogram(
    accuracies1_by_qid,
    accuracies2_by_qid,
    title,
    labels=["Set 1", "Set 2"],
    color="orange",
):
    """Plot a histogram of differences between two sets of accuracies with median line.

    Args:
        accuracies1: First list of accuracies
        accuracies2: Second list of accuracies (must be same length as accuracies1)
        title: Plot title
        labels: Labels for the two accuracy sets
        color: Color for the histogram bars
    """
    if accuracies1_by_qid.keys() != accuracies2_by_qid.keys():
        # Take intersection of keys
        common_keys = set(accuracies1_by_qid.keys()) & set(accuracies2_by_qid.keys())
        accuracies1_by_qid = {k: accuracies1_by_qid[k] for k in common_keys}
        accuracies2_by_qid = {k: accuracies2_by_qid[k] for k in common_keys}

    differences = [
        accuracies1_by_qid[q_id] - accuracies2_by_qid[q_id]
        for q_id in accuracies1_by_qid.keys()
    ]
    median_diff = np.median(differences)
    print(f"Median difference: {median_diff}")

    mean_diff = np.mean(differences)
    print(f"Mean difference: {mean_diff}")

    # Sort values to make plot prettier
    differences = sorted(differences)
    print(f"Middle value: {differences[len(differences) // 2]}")

    # Count value strictly higher than zero
    count_strictly_higher_than_zero = len([d for d in differences if d > 0])
    print(f"Count strictly higher than zero: {count_strictly_higher_than_zero}")

    # Count value strictly lower than zero
    count_strictly_lower_than_zero = len([d for d in differences if d < 0])
    print(f"Count strictly lower than zero: {count_strictly_lower_than_zero}")

    # Count value equal to zero
    count_equal_to_zero = len([d for d in differences if d == 0])
    print(f"Count equal to zero: {count_equal_to_zero}")

    plt.figure(figsize=(8, 6))
    plt.hist(differences, bins=20, alpha=0.7, color=color)
    plt.axvline(
        x=median_diff,
        color="red",
        linestyle="--",
        label=f"Median diff: {median_diff:.2f}",
    )
    plt.title(title)
    plt.xlabel(f"Accuracy Difference ({labels[0]} - {labels[1]})")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_heatmap(
    accuracies1_by_qid, accuracies2_by_qid, title, xlabel, ylabel, bins=20
):
    """Plot a heatmap comparing two sets of accuracies.

    Args:
        acc1: First list of accuracies (X-axis)
        acc2: Second list of accuracies (Y-axis)
        title: Plot title
        xlabel: Label for X-axis
        ylabel: Label for Y-axis
        bins: Number of bins for the 2D histogram
    """
    plt.figure(figsize=(10, 8))

    if accuracies1_by_qid.keys() != accuracies2_by_qid.keys():
        # Take intersection of keys
        common_keys = set(accuracies1_by_qid.keys()) & set(accuracies2_by_qid.keys())
        accuracies1_by_qid = {k: accuracies1_by_qid[k] for k in common_keys}
        accuracies2_by_qid = {k: accuracies2_by_qid[k] for k in common_keys}

    # Sort by keys
    sorted_accs1 = sorted(list(accuracies1_by_qid.items()), key=lambda x: x[0])
    sorted_accs2 = sorted(list(accuracies2_by_qid.items()), key=lambda x: x[0])

    acc1 = [acc for _, acc in sorted_accs1]
    acc2 = [acc for _, acc in sorted_accs2]

    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(acc1, acc2, bins=bins)

    # Create heatmap
    plt.imshow(
        hist.T,
        origin="lower",
        aspect="auto",
        extent=[min(acc1), max(acc1), min(acc2), max(acc2)],
        cmap="YlOrRd",
    )

    plt.colorbar(label="Count")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add correlation coefficient
    correlation = np.corrcoef(acc1, acc2)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()
