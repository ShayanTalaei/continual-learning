import json
import os
from tqdm import tqdm

dataset_path = "/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_250_triplets_false_1000_reps_temp_0.7/dataset.jsonl"

# Create output directory
output_dir = os.path.dirname(dataset_path)
output_path = os.path.join(output_dir, "dataset_filtered.jsonl")

# Read and filter dataset
filtered_samples = []
total_samples = 0

# First pass: count total samples for progress bar
print("Counting total samples...")
with open(dataset_path, 'r') as f:
    for line in tqdm(f, desc="Counting"):
        total_samples += 1

# Second pass: filter samples
print(f"Filtering {total_samples} samples...")
with open(dataset_path, 'r') as f:
    for line in tqdm(f, total=total_samples, desc="Filtering"):
        row = json.loads(line.strip())
        
        # Check if evaluation score is 1
        if row.get('evaluation', {}).get('score') == 1:
            filtered_samples.append(row)

# Save filtered dataset
print(f"Saving {len(filtered_samples)} filtered samples...")
with open(output_path, 'w') as f:
    for sample in tqdm(filtered_samples, desc="Writing"):
        f.write(json.dumps(sample) + '\n')

print(f"Filtered dataset saved to: {output_path}")
print(f"Original samples: {total_samples}")
print(f"Filtered samples: {len(filtered_samples)}")
print(f"Filter rate: {len(filtered_samples)/total_samples:.2%}")