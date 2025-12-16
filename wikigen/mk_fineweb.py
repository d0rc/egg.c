#!/usr/bin/env python3
"""
Script to convert the Salesforce/wikitext dataset to a single text file.
"""
from datasets import load_dataset

def convert_wikitext_to_single_file():
    # Load the wikitext dataset (using the 103 version which is commonly used)
    print("Loading the fineweb dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb")
    
    # Create output file
    output_file = "fineweb_combined.txt"
    
    print(f"Writing all text data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        # Process train, validation, and test splits
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                print(f"Processing {split_name} split...")
                for i, example in enumerate(dataset[split_name]):
                    # Write the text content with a separator
                    text = example.get("text", "")
                    if text.strip():  # Only write non-empty text
                        f.write(text + "\n\n")
                    
                    # Print progress every 1000 examples
                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i + 1} examples from {split_name} split...")
    
    print(f"All done! The combined text has been written to {output_file}")

if __name__ == "__main__":
    convert_wikitext_to_single_file()