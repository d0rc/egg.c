#ifndef EGG_UTILS_LOSS_FORMATTER_H
#define EGG_UTILS_LOSS_FORMATTER_H

#include <string>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "../config.h"

// Constants
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

namespace egg {
namespace utils {

inline double loss_to_nats(double reported_loss) {
    return reported_loss / (double)(1 << FIXED_POINT);
}

inline double loss_to_bits(double nats) {
    return nats / M_LN2;
}

inline std::string get_loss_description(double bits) {
    if (bits > 13.5) return "Initialization / Diverged";
    if (bits > 12.5) return "Random Guessing";
    if (bits > 7.0)  return "Unigram Range";
    if (bits > 5.0)  return "Bigram Range";
    if (bits > 2.0)  return "Trained";
    return "Overfitting / Memorization";
}

inline std::string format_loss_info(double reported_loss) {
    double nats = loss_to_nats(reported_loss);
    double bits = loss_to_bits(nats);
    std::string desc = get_loss_description(bits);
    
    char buf[256];
    snprintf(buf, sizeof(buf), "       Stats: %.4f nats | %.4f bits | %s", nats, bits, desc.c_str());
    return std::string(buf);
}

} // namespace utils
} // namespace egg

#endif // EGG_UTILS_LOSS_FORMATTER_H
