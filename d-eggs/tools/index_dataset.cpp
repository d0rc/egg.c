#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>

// Include config to get TokenType definition
#include "../include/config.h"
#include "../include/model/definitions.h"

void print_usage(const char* name) {
    std::cerr << "Usage: " << name << " [options]\n"
              << "Options:\n"
              << "  --input <file>      Input binary file (default: input.bin)\n"
              << "  --output <file>     Output binary file (default: aligned-{SEQ_LEN}.bin)\n"
              << "  --bos <id>          BOS token ID (required)\n"
              << "  --eos <id>          EOS token ID (required)\n"
              << "  --pad <id>          PAD token ID (required)\n"
              << "  --seq-len <len>     Sequence length (default: " << SEQ_LEN << ")\n"
              << "  --help              Show this help\n";
}

int main(int argc, char** argv) {
    std::string input_path = "input.bin";
    std::string output_path = "";
    TokenType bos_token = 0;
    TokenType eos_token = 0;
    TokenType pad_token = 0;
    int seq_len = SEQ_LEN;
    
    bool bos_set = false;
    bool eos_set = false;
    bool pad_set = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input") {
            if (i + 1 < argc) input_path = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--bos") {
            if (i + 1 < argc) { bos_token = (TokenType)std::stoul(argv[++i]); bos_set = true; }
        } else if (arg == "--eos") {
            if (i + 1 < argc) { eos_token = (TokenType)std::stoul(argv[++i]); eos_set = true; }
        } else if (arg == "--pad") {
            if (i + 1 < argc) { pad_token = (TokenType)std::stoul(argv[++i]); pad_set = true; }
        } else if (arg == "--seq-len") {
            if (i + 1 < argc) seq_len = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!bos_set || !eos_set || !pad_set) {
        std::cerr << "Error: --bos, --eos, and --pad are required.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (output_path.empty()) {
        output_path = "aligned-" + std::to_string(seq_len) + ".bin";
    }

    std::cout << "Settings:\n"
              << "  Input: " << input_path << "\n"
              << "  Output: " << output_path << "\n"
              << "  BOS: " << (uint32_t)bos_token << "\n"
              << "  EOS: " << (uint32_t)eos_token << "\n"
              << "  PAD: " << (uint32_t)pad_token << "\n"
              << "  SEQ_LEN: " << seq_len << "\n";

    // Open Input
    std::ifstream in_file(input_path, std::ios::binary | std::ios::ate);
    if (!in_file) {
        std::cerr << "Error: Could not open input file " << input_path << "\n";
        return 1;
    }
    std::streamsize file_size = in_file.tellg();
    in_file.seekg(0, std::ios::beg);

    size_t num_tokens = file_size / sizeof(TokenType);
    std::cout << "Reading " << num_tokens << " tokens (" << file_size << " bytes)...\n";

    std::vector<TokenType> data(num_tokens);
    if (!in_file.read((char*)data.data(), file_size)) {
        std::cerr << "Error: Failed to read input file.\n";
        return 1;
    }
    in_file.close();

    // Open Output
    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Could not open output file " << output_path << "\n";
        return 1;
    }

    size_t mined_sequences = 0;
    size_t mined_tokens = 0;
    size_t ignored_tokens = 0;

    std::vector<TokenType> buffer;
    buffer.reserve(seq_len * 2); // Reserve enough

    auto last_print = std::chrono::steady_clock::now();
    auto start_time = last_print;

    size_t i = 0;
    while (i < num_tokens) {
        // Progress Update
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() >= 1000) {
            double progress = (double)i / num_tokens * 100.0;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            std::cout << "\r[Progress " << std::fixed << std::setprecision(1) << progress << "%] "
                      << "Mined: " << mined_sequences << " seqs (" << mined_tokens << " toks), "
                      << "Ignored: " << ignored_tokens << " toks, "
                      << "Time: " << elapsed << "s" << std::flush;
            last_print = now;
        }

        // Find BOS
        if (data[i] != bos_token) {
            // Skip until BOS
            // ignored_tokens++; // Should we count tokens before first BOS as ignored? Yes.
            // But usually we just scan for BOS.
            i++;
            continue;
        }

        // Found BOS at i
        buffer.clear();
        buffer.push_back(data[i]); // Add BOS
        i++;

        bool eos_found = false;
        bool next_bos_found = false;

        // Collect sequence
        while (i < num_tokens) {
            TokenType t = data[i];
            if (t == bos_token) {
                next_bos_found = true;
                break; // Stop, don't consume next BOS
            }
            
            buffer.push_back(t);
            i++;

            if (t == eos_token) {
                eos_found = true;
                break; // Stop after EOS
            }
            
            if (buffer.size() > seq_len) {
                // Too long already
                break;
            }
        }

        // Check if valid
        // If we stopped because of length > seq_len
        if (buffer.size() > seq_len) {
            ignored_tokens += buffer.size();
            // If we didn't hit BOS/EOS, we need to skip until we find one to resync?
            // The loop above consumes until BOS or EOS or Length.
            // If we broke due to length, 'i' is at the token that caused overflow.
            // We should probably skip until next BOS to reset.
            while (i < num_tokens && data[i] != bos_token) {
                ignored_tokens++;
                i++;
            }
            continue;
        }

        // If we stopped because of EOS or Next BOS or End of File
        // We need to ensure we have an EOS.
        if (!eos_found) {
            if (buffer.size() < seq_len) {
                buffer.push_back(eos_token);
                eos_found = true;
            } else {
                // Can't fit EOS
                ignored_tokens += buffer.size();
                continue;
            }
        }

        // Now we have a sequence ending in EOS (or we added it) and size <= seq_len
        mined_sequences++;
        mined_tokens += buffer.size();

        // Pad
        while (buffer.size() < seq_len) {
            buffer.push_back(pad_token);
        }

        // Write
        out_file.write((char*)buffer.data(), seq_len * sizeof(TokenType));
    }

    out_file.close();

    std::cout << "\nDone.\n"
              << "  Mined Sequences: " << mined_sequences << "\n"
              << "  Mined Tokens:    " << mined_tokens << "\n"
              << "  Ignored Tokens:  " << ignored_tokens << "\n"
              << "  Output Size:     " << (mined_sequences * seq_len * sizeof(TokenType)) << " bytes\n";

    return 0;
}
