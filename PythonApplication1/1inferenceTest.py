# import libs
import requests
import jsonlines
import time
import json
import pandas as pd

# define the Ollama endpoint for the generation
OLLAMA_GENERATE_URL = f"http://localhost:11434/api/generate"

def generate(model : str, prompt : str, options : dict = None):
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False, 
            "options": options,
        },
    )

    # check if any error occurred during the request
    resp.raise_for_status()

    data = resp.json()

    return data["response"], data["total_duration"] * 10**-9

# use a sample prompt to test the inference function
prompt = "Write the dijkstra algorithm in Javascript. Output only the source code, no explaination."

# run "ollama list" to print all the models that have been downloaded

# define the model to use
OLLAMA_MODEL = "gemma3:1b" # example

start = time.time()

response, ollama_total_duration = generate(
    model=OLLAMA_MODEL,
    prompt=prompt,
)

end = time.time()

e2e = end - start

print(f"Ollama total duration (from JSON response): {ollama_total_duration:.6f} s")
print(f"End-to-end generation time: {e2e:.6f} s")

# Formatted output
print(f"=== PROMPT ===")
print(f"{prompt[:200]} ... \n")

print(f"=== RESPONSE ===")
print(f"{response[:200]} ... ")