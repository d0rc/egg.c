import os
import struct
import array
from transformers import AutoTokenizer

def main():
    print("starting up...")

    model_id = "PleIAs/Monad"
    input_file = "../input.txt"
    output_file = "input.bin"
    decoding_file = "decoding.bin"

    print(f"Loading tokenizer: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Tokenizer loaded.")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Is fast tokenizer: {tokenizer.is_fast}")

    # --- Part 1: Tokenize input.txt ---
    print(f"Reading input file: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Tokenizing {len(text)} characters...")
    tokens = tokenizer.encode(text, add_special_tokens=False)

    print(f"Generated {len(tokens)} tokens.")

    print(f"Writing to {output_file}...")
    try:
        # Write as uint32 (I)
        token_array = array.array('I', tokens)
        with open(output_file, 'wb') as f:
            token_array.tofile(f)

        print(f"Successfully wrote {len(tokens)} tokens to {output_file}")

    except Exception as e:
        print(f"Error writing output file: {e}")

    # --- Part 2: Generate decoding.bin ---
    print(f"Generating {decoding_file}...")
    try:
        vocab_size = tokenizer.vocab_size
        with open(decoding_file, 'wb') as f:
            # Write vocab size first
            f.write(struct.pack('I', vocab_size))

            for i in range(vocab_size):
                # Get string representation
                # Note: decode([i]) might be slow for 150k tokens loop, but acceptable for a one-off script.
                # We use clean_up_tokenization_spaces=False to get exact mapping if possible,
                # though for Qwen/BPE, decode is usually what we want for reconstruction.
                s = tokenizer.decode([i])
                b = s.encode('utf-8')

                # Write length (uint32) and bytes
                f.write(struct.pack('I', len(b)))
                f.write(b)

                if i % 10000 == 0:
                    print(f"Processed {i}/{vocab_size} tokens...", end='\r')

            print(f"\nSuccessfully wrote {vocab_size} entries to {decoding_file}")

    except Exception as e:
        print(f"Error writing decoding file: {e}")

if __name__ == '__main__':
    main()
