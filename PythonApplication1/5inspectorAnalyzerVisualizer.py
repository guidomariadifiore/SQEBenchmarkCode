# import libs
import requests
import jsonlines
import time
import json
import pandas as pd

# define the Ollama endpoint for the generation
OLLAMA_GENERATE_URL = f"http://localhost:11434/api/generate"

# use a sample prompt to test the inference function
prompt = "Write the dijkstra algorithm in Javascript. Output only the source code, no explaination."

# define the model to use
OLLAMA_MODEL = "gemma3:1b" # example

def generate_with_stream(model: str, prompt: str, options : dict = None):

    # in this case we measure inside the generation function
    start = time.time()

    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True, 
            "options": options,
        },
    )

    # check if any error occurred during the request
    resp.raise_for_status()

    first_token_time = None

    chunks = []

    # the iter_lines automatically checks when the stream ends
    for line in resp.iter_lines():
        if not line:
            continue
        # this time we have to use json to decode
        data = json.loads(line.decode("utf-8"))
        token = data.get("response", "")
        if not token:
            continue

        # TTFT
        now = time.time()
        if first_token_time is None:
            first_token_time = now

        chunks.append(token)

    end = time.time()
    total_gen_time = end - start

    text = "".join(chunks)

    # the 'data' variables contains now the last json object
    ollama_total_duration = data['total_duration'] * 10**-9
    
    ttft = (first_token_time - start) if first_token_time else 0.0
    itl = (total_gen_time - ttft) / max(len(chunks) - 1, 1)


    # in this case the response is composed of multiple json objects (you can check by directly calling the endpoint)

    return text, total_gen_time, ttft, itl, ollama_total_duration


response, e2e, ttft, itl, ollama_total_duration = generate_with_stream(
    model=OLLAMA_MODEL,
    prompt=prompt,
)

print(f"Ollama total duration (from JSON response): {ollama_total_duration:.6f} s")
print(f"End-to-end generation time: {e2e:.6f} s")
print(f"Time to first token (TTFT): {ttft:.6f} s")
print(f"Inter-token latency (ITL): {itl:.6f} s")

GENERATION_OPTIONS = { 
    "num_predict": 256, # we can limit the maximum number of generated tokens
    "temperature": 0 # handle randomness of the generation process
}
NUMBER_OF_ITERATIONS = 30 # define the number of iterations

from tqdm import tqdm

results = []

for it in tqdm(range(NUMBER_OF_ITERATIONS)):

    response, e2e, ttft, itl, ollama_total_duration = generate_with_stream(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options=GENERATION_OPTIONS
    )

    results.append({
        "prompt": prompt,
        "iteration": it,
        "response": response,
        "ollama_total_gen_time": ollama_total_duration,
        "end_to_end_latency": e2e,
        "TTFT": ttft,
        "ITL": itl
    })

results = pd.DataFrame(results)

# Dump results for further inspection and analysis 
with jsonlines.open("results.jsonl", "w") as out_file:
    out_file.write_all(results.to_dict('records'))

results.describe()

import numpy as np

### ===> perform an in-depth examination of the value distribution

def print_stats(stats):
    for k, v in stats.items():
        if isinstance(v, tuple):
            print(f"{k}: ({v[0]:.3f}, {v[1]:.3f})")
        else:
            print(f"{k}: {v:.3f}")

def summary_stats(data):
    mu = data.mean()
    range_ = data.max() - data.min()
    max_dist_mean = np.abs(data - mu).max()
    variance = data.var(ddof=1) if len(data) > 1 else 0.0
    sigma = data.std(ddof=1) if len(data) > 1 else 0.0
    cv = sigma / mu if mu != 0 else np.nan
    std_error = sigma / np.sqrt(len(data)) if len(data) > 1 else 0.0
    statistics = {
        "Mean (μ)": mu,
        "Range": range_,
        "Max distance from mean": max_dist_mean,
        "Variance (σ²)": variance,
        "Standard deviation (σ)": sigma,
        "Coefficient of variation (CV)": cv,
        "Standard error of the mean (SEM)": std_error
    }
    return statistics

# example on TTFT
print_stats(summary_stats(results.TTFT))

import matplotlib.pyplot as plt

# visualize the results (example)

latency_measurements = results.TTFT.values

plt.figure(figsize=(10, 2))
plt.plot(latency_measurements, color="black") 

plt.axhline(min(latency_measurements), color="lightgrey", ls="--")
plt.axhline(max(latency_measurements), color="lightgrey", ls="--")

plt.xlabel("Iteration")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.show()