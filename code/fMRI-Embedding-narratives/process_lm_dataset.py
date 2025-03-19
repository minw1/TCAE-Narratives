import h5py
import spacy
import numpy as np
import random
import os
from datasets import load_dataset
from utils import pos_tags  # Import the POS tag mapping dictionary

'''
# Define the output directory and filename
output_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/h5_lm"
output_file = os.path.join(output_dir, "wikitext_udpos_vectors_new.h5")

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Load English SpaCy model with UDPOS tagging
nlp = spacy.load("en_core_web_sm")

# Load the dataset (limit to first 500k lines)
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
max_lines = 100  # Process only first 500k lines
pad_token = pos_tags["PAD"]  # Padding token

# Create an HDF5 file to store the processed data
with h5py.File(output_file, "w") as h5f:
    group = h5f.create_group("pos_vectors")
    
    for idx, example in enumerate(dataset):
        print(f"{example['text']}, {idx}")
        if idx >= max_lines:  # Stop after 500k lines
            break

        text = example["text"].strip()  # Extract and clean text
        if not text:  # Skip empty lines
            continue

        # Process text with spaCy
        doc = nlp(text)
        
        # Convert tokens to UDPOS integers, excluding punctuation
        udpos_vector = [pos_tags[token.pos_] for token in doc if token.pos_ in pos_tags and token.pos_ != "PUNCT"]

        # Step 2b: Truncate to a random length j between 20 and 70
        j = random.randint(20, 70)
        udpos_vector = udpos_vector[:j]  # Truncate if needed

        # Step 2c: Pad to length 100 with PAD tokens
        if len(udpos_vector) < 100:
            udpos_vector += [pad_token] * (100 - len(udpos_vector))
        else:
            udpos_vector = udpos_vector[:100]  # Ensure max length of 100

        # Convert to numpy array
        udpos_vector = np.array(udpos_vector, dtype=np.int32)

        # Store each processed line as a dataset in HDF5
        group.create_dataset(f"line_{idx}", data=udpos_vector)

print(f"Processing complete. Data saved to {output_file}")
'''

'''
import h5py
from utils import pos_tags  # Import the POS tag mapping dictionary

# Reverse the pos_tags dictionary to map integers back to POS tags
int_to_pos = {v: k for k, v in pos_tags.items()}

# Open the HDF5 file
h5_file = "/home/wsm32/project/wsm_thesis_scratch/narratives/h5_lm/wikitext_udpos_vectors.h5"

with h5py.File(h5_file, "r") as h5f:
    group = h5f["pos_vectors"]
    print(len(group.keys()))
    
    # Get first 10 datasets (lines)
    first_10_keys = sorted(group.keys(), key=lambda x: int(x.split('_')[1]))[:10]
    
    for key in first_10_keys:
        pos_vector = group[key][()]  # Load stored integer vector
        pos_tags_list = [int_to_pos[i] for i in pos_vector]  # Convert back to POS tags
        print(f"{key}: {pos_tags_list}")  # Print POS tag sequence
    
'''


import h5py
import spacy
import numpy as np
import random
import os
from datasets import load_dataset
from utils import pos_tags  # Import the POS tag mapping dictionary

# Define the output directory
output_dir = "/home/wsm32/project/wsm_thesis_scratch/narratives/h5_lm"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Load English SpaCy model with UDPOS tagging
nlp = spacy.load("en_core_web_sm")

# Define function to process and save dataset
def process_and_save(split_name):
    print(f"Processing {split_name} split...")
    
    # Load dataset split
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split_name)

    # Define output file name
    output_file = os.path.join(output_dir, f"wikitext_udpos_vectors_{split_name}.h5")

    pad_token = pos_tags["PAD"]  # Padding token

    # Create HDF5 file
    with h5py.File(output_file, "w") as h5f:
        group = h5f.create_group("pos_vectors")

        for idx, example in enumerate(dataset):
            text = example["text"].strip()  # Extract and clean text
            if not text:  # Skip empty lines
                continue

            # Process text with spaCy
            doc = nlp(text)

            # Convert tokens to UDPOS integers, excluding punctuation
            udpos_vector = [pos_tags[token.pos_] for token in doc if token.pos_ in pos_tags and token.pos_ != "PUNCT"]

            # Step 2b: Truncate to a random length j between 20 and 70
            j = random.randint(20, 70)
            udpos_vector = udpos_vector[:j]  # Truncate if needed

            # Step 2c: Pad to length 100 with PAD tokens
            if len(udpos_vector) < 100:
                udpos_vector += [pad_token] * (100 - len(udpos_vector))
            else:
                udpos_vector = udpos_vector[:100]  # Ensure max length of 100

            # Convert to numpy array
            udpos_vector = np.array(udpos_vector, dtype=np.int32)

            # Store each processed line as a dataset in HDF5
            group.create_dataset(f"line_{idx}", data=udpos_vector)

    print(f"Processing complete. Data saved to {output_file}")

# Process validation and test splits
process_and_save("train")
process_and_save("validation")
process_and_save("test")
