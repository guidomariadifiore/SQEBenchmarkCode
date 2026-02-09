import jsonlines

# Define your input and output filenames
input_files = [
    "results_script6_1b.jsonl", 
]

def fix_row(row):
    # capture the values currently stored in the WRONG keys
    # based on the unpacking bug: response, ttft, itl, e2e, ollama_total_duration
    
    val_stored_in_ttft = row.get("TTFT")             # This is actually End-to-End Total Time
    val_stored_in_itl = row.get("ITL")               # This is actually TTFT
    val_stored_in_e2e = row.get("end_to_end_latency") # This is actually ITL

    # Assign them to the CORRECT keys
    row["end_to_end_latency"] = val_stored_in_ttft
    row["TTFT"] = val_stored_in_itl
    row["ITL"] = val_stored_in_e2e
    
    return row

for filename in input_files:
    output_filename = filename.replace(".jsonl", "_FIXED.jsonl")
    print(f"Processing {filename} -> {output_filename}...")
    
    try:
        with jsonlines.open(filename, "r") as reader:
            with jsonlines.open(output_filename, "w") as writer:
                for row in reader:
                    fixed_row = fix_row(row)
                    writer.write(fixed_row)
        print("Done!")
    except FileNotFoundError:
        print(f"File {filename} not found, skipping.")

print("\nAll files fixed. Please update your Analysis.py to load the '_FIXED' files.")