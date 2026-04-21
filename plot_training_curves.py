import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def setup_publication_style():
    """Configures matplotlib for publication-quality plots."""
    sns.set_theme(style="ticks", context="paper")
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,  # Ensures fonts are embedded in PDFs
        "ps.fonttype": 42
    })

def plot_with_smoothing(ax, x, y, color, title, ylabel):
    """Plots raw data with high transparency and a smoothed line over it."""
    # Raw noisy data
    ax.plot(x, y, alpha=0.2, color=color, label="Raw")
    
    # Smoothed trendline (Exponential Moving Average)
    smoothed = y.ewm(span=50, adjust=False).mean()
    ax.plot(x, smoothed, color=color, linewidth=2, label="Smoothed (EMA)")
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=True, edgecolor='black')
    sns.despine(ax=ax)

def main():
    # ==========================================
    # ARGUMENT PARSING
    # ==========================================
    parser = argparse.ArgumentParser(description="Plot training and validation logs.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing train_log.csv and val_log.csv"
    )
    parser.add_argument(
        "--reg_lambda", 
        type=float, 
        default=0.7, 
        help="Regularization lambda value (used in plot titles)"
    )
    parser.add_argument(
        "--all_plots", 
        action="store_true", 
        help="If set, plots all 4 metrics (Total, Task, Reg, Val). Otherwise, plots only Train Task and Val Loss."
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    reg_lambda = args.reg_lambda
    all_plots = args.all_plots

    train_csv = os.path.join(output_dir, "train_log.csv")
    val_csv = os.path.join(output_dir, "val_log.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"Could not find CSV logs in {output_dir}. Please check the path.")

    # Load data
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    setup_publication_style()
    palette = sns.color_palette("colorblind")

    if all_plots:
        # 4 Plots (2x2 Grid)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        # 1. Total Train Loss
        plot_with_smoothing(
            ax=axes[0], 
            x=df_train["step"], 
            y=df_train["total_loss"], 
            color=palette[0], # Blue
            title=f"Total Training Loss ($\lambda = ${reg_lambda})", 
            ylabel="Total Loss"
        )

        # 2. Train Task Loss
        plot_with_smoothing(
            ax=axes[1], 
            x=df_train["step"], 
            y=df_train["task_loss"], 
            color=palette[2], # Green
            title="Training Task Loss ($\mathcal{L}_{task}$)", 
            ylabel="Task Loss"
        )

        # 3. Regularization Loss
        plot_with_smoothing(
            ax=axes[2], 
            x=df_train["step"], 
            y=df_train["reg_loss"], 
            color=palette[1], # Orange
            title="Regularization Loss", 
            ylabel="Reg Loss"
        )
        
        val_ax = axes[3]
        
    else:
        # 2 Plots (1x2 Grid)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes = axes.flatten()
        
        # 1. Train Task Loss
        plot_with_smoothing(
            ax=axes[0], 
            x=df_train["step"], 
            y=df_train["task_loss"], 
            color=palette[2], # Green
            title="Training Task Loss", 
            ylabel="Loss"
        )
        
        val_ax = axes[1]

    # Validation Loss Plot (Shared logic for both modes)
    val_ax.plot(df_val["step"], df_val["val_loss"], marker='o', markersize=4, 
                 color=palette[3], linewidth=2, label="Validation Loss")
    val_ax.plot(df_val["step"], df_val["best_val_loss"], linestyle='--', 
                 color='black', alpha=0.7, linewidth=1.5, label="Best Val Loss")
    
    val_ax.set_title("Validation Loss", fontweight='bold')
    val_ax.set_xlabel("Training Steps")
    val_ax.set_ylabel("Loss")
    val_ax.legend(loc="upper right", frameon=True, edgecolor='black')
    sns.despine(ax=val_ax)

    plt.tight_layout()

    # Save format slightly altered to distinguish the plots if you run both
    plot_name = "training_metrics_all.png" if all_plots else "training_metrics_basic.png"
    save_path = os.path.join(output_dir, plot_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)
    print(f"Publication-ready plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()