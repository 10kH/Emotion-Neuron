import json
import os
import re

# Load the JSON file
input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/fits_sum.json')
with open(input_file, 'r') as file:
    data = json.load(file)

# Count the number of keys before deduplication
initial_key_count = len(data)

# Function to normalize keys by removing spaces and special characters, and making lowercase
def normalize_key(key):
    # Remove all non-alphanumeric characters and make lowercase
    return re.sub(r'\W+', '', key).lower()

# Remove duplicates by using the normalized keys
deduped_data = {}
for key, value in data.items():
    normalized_key = normalize_key(key)
    deduped_data[normalized_key] = value  # Overwrite duplicates with the same normalized key

# Count the number of keys after deduplication
final_key_count = len(deduped_data)

# Save the deduplicated JSON back to a file
output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/fits_sum_deduped.json')
with open(output_file, 'w') as file:
    json.dump(deduped_data, file, ensure_ascii=False, indent=4)

print(f"Number of keys before deduplication: {initial_key_count}")
print(f"Number of keys after deduplication: {final_key_count}")
print("Duplicate keys removed, and file saved as 'fits_sum_deduped.json'")
