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