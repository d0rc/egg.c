#ifndef EGG_WORKER_JOB_TYPES_H
#define EGG_WORKER_JOB_TYPES_H

#include <vector>
#include <cstdint>
#include "protocol.h"

// Job Item: Received from Coordinator
struct JobItem {
    EggJobResponseHeader header;
    std::vector<uint8_t> payload; // Contains model update data if model_size > 0
};

// Result Type Enum
enum class ResultType {
    COMPUTE,
    LOG_MESSAGE
};

// Result Item: To be sent to Coordinator
struct ResultItem {
    ResultType type;
    EggResultHeader header; // Valid if type == COMPUTE
    std::vector<uint8_t> payload; // Result data (packed fitness) or Log string
};

#endif // EGG_WORKER_JOB_TYPES_H
