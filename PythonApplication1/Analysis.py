import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
RESULTS_FILE = "results_script6_1b_FIXED.jsonl"  
OUTPUT_DIR = "analysis_output"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_and_preprocess_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run the benchmark script first.")
        return None

    print(f"Loading data from {filename}...")
    df = pd.read_json(filename, lines=True)

    # --- Preprocessing / Safety Checks ---

    # 1. Ensure 'model' column exists (default to 'unknown' if missing)
    if "model" not in df.columns:
        df["model"] = "default_model"

    # 2. Calculate Input Length if missing (based on 'prompt' column)
    if "input_length" not in df.columns and "prompt" in df.columns:
        print("Note: 'input_length' not found in data. Calculating from 'prompt'...")
        df["input_length"] = df["prompt"].apply(len)

    # 3. Calculate Response Length if missing (requires 'response' text to be present)
    if "response_length" not in df.columns:
        if "response" in df.columns:
            print("Note: 'response_length' not found. Calculating from 'response'...")
            df["response_length"] = df["response"].apply(len)
        else:
            print("WARNING: 'response_length' and 'response' text missing.")
            print("   -> You cannot analyze 'Metrics vs Output Length' without this.")
            print("   -> Fix: Update your benchmark script to save len(response).")
            # Create dummy column to prevent crash, but fill with NaNs
            df["response_length"] = None

    return df


def analyze_and_plot(df):
    # Set the visual style
    sns.set_theme(style="whitegrid")

    # --- 1. STATISTICS SUMMARY ---
    print("\n=== BENCHMARK STATISTICS ===")
    # Group by Model and calculate mean/std for key metrics
    stats = df.groupby("model")[["TTFT", "ITL", "end_to_end_latency"]].agg(
        ["mean", "std", "min", "max"]
    )
    print(stats)

    # Save stats to CSV
    stats.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv")
    print(f"\nStatistics saved to {OUTPUT_DIR}/summary_statistics.csv")

    # --- 2. VARIABILITY ANALYSIS (Box Plots) ---
    # Question: "Do results vary when executing multiple times?"

    # Plot TTFT Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="TTFT", hue="model", palette="Set2")
    plt.title("Variability of Time To First Token (TTFT)")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/variability_ttft.png")
    plt.show()

    # Plot ITL Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="ITL", hue="model", palette="Set2")
    plt.title("Variability of Inter-Token Latency (ITL)")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/variability_itl.png")
    plt.show()

    # --- 3. INPUT PROMPT vs METRICS ---
    # Question: "Is there a relationship between metrics and input prompt?"

    if "input_length" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="input_length", y="TTFT", hue="model", alpha=0.7)
        plt.title("Input Length vs. Time To First Token (TTFT)")
        plt.xlabel("Prompt Length (characters)")
        plt.ylabel("TTFT (s)")

        # Add trendline if possible (simple linear regression)
        # Note: robust linear regression usually requires statsmodels or distinct handling per group
        # We will stick to visual scatter for now.
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_input_ttft.png")
        plt.show()

    # --- 4. OUTPUT LENGTH vs METRICS ---
    # Question: "Is there a relationship between metrics and output length?"

    if df["response_length"].notnull().any():
        # E2E vs Output Length (Should be linear)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x="response_length", y="end_to_end_latency", hue="model", alpha=0.7
        )
        plt.title("Output Length vs. End-to-End Latency")
        plt.xlabel("Response Length (characters)")
        plt.ylabel("Total Time (s)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_output_e2e.png")
        plt.show()

        # ITL vs Output Length (Should be flat/constant)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="response_length", y="ITL", hue="model", alpha=0.7)
        plt.title("Output Length vs. Inter-Token Latency (ITL)")
        plt.xlabel("Response Length (characters)")
        plt.ylabel("ITL (s)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_output_itl.png")
        plt.show()
    else:
        print("\nSkipping Output vs. Metrics plots (Response length data missing).")


def main():
    df = load_and_preprocess_data(RESULTS_FILE)
    if df is not None and not df.empty:
        analyze_and_plot(df)
        print(f"\nAnalysis complete. Check the '{OUTPUT_DIR}' folder for images.")
    else:
        print("No data to analyze.")


if __name__ == "__main__":
    main()
