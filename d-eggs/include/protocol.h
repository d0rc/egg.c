#ifndef EGG_PROTOCOL_H
#define EGG_PROTOCOL_H

#include <stdint.h>
#include <string.h>
#include <arpa/inet.h> // for htonl, ntohl

#define EGG_PROTO_MAGIC 0x4A4F4253 // "JOBS"
#define EGG_PROTO_VERSION 2

// Opcodes
#define OP_JOB_REQUEST  0
#define OP_RESULT       1
#define OP_JOB_RESPONSE 2
#define OP_LOG_MESSAGE  3

// Header Size
#define EGG_HEADER_SIZE 10

// 64-bit Endianness Helpers (guard against macOS system macros)
#ifndef htonll
static inline uint64_t htonll(uint64_t value) {
    if (htonl(1) == 1) return value; // Big-endian system
    return ((uint64_t)htonl(value & 0xFFFFFFFFUL) << 32) | htonl(value >> 32);
}
#endif

#ifndef ntohll
static inline uint64_t ntohll(uint64_t value) {
    if (htonl(1) == 1) return value;
    return ((uint64_t)ntohl(value & 0xFFFFFFFFUL) << 32) | ntohl(value >> 32);
}
#endif

// Header Parsing/Writing
static inline void egg_write_header(uint8_t *buf, uint8_t opcode, uint32_t payload_len) {
    uint32_t magic = htonl(EGG_PROTO_MAGIC);
    memcpy(buf, &magic, 4);
    buf[4] = EGG_PROTO_VERSION;
    buf[5] = opcode;
    uint32_t len = htonl(payload_len);
    memcpy(buf + 6, &len, 4);
}

static inline int egg_parse_header(const uint8_t *buf, uint8_t *opcode, uint32_t *payload_len) {
    uint32_t magic;
    memcpy(&magic, buf, 4);
    if (ntohl(magic) != EGG_PROTO_MAGIC) return -1;
    if (buf[4] != EGG_PROTO_VERSION) return -2;
    
    *opcode = buf[5];
    uint32_t len;
    memcpy(&len, buf + 6, 4);
    *payload_len = ntohl(len);
    return 0;
}

// Payload Structs (Helpers for serialization, NOT for direct casting)

// JOB_REQUEST (24 bytes)
typedef struct {
    uint64_t seed;
    uint64_t last_step;
    uint64_t data_position;
} EggJobRequest;

static inline void egg_serialize_job_request(uint8_t *buf, const EggJobRequest *req) {
    uint64_t s = htonll(req->seed);
    uint64_t l = htonll(req->last_step);
    uint64_t d = htonll(req->data_position);
    memcpy(buf, &s, 8);
    memcpy(buf + 8, &l, 8);
    memcpy(buf + 16, &d, 8);
}

static inline void egg_deserialize_job_request(const uint8_t *buf, EggJobRequest *req) {
    uint64_t s, l, d;
    memcpy(&s, buf, 8);
    memcpy(&l, buf + 8, 8);
    memcpy(&d, buf + 16, 8);
    req->seed = ntohll(s);
    req->last_step = ntohll(l);
    req->data_position = ntohll(d);
}

// JOB_RESPONSE Header (32 bytes + model_data)
typedef struct {
    uint64_t seed;
    uint64_t last_step;
    uint64_t data_position;
    uint32_t model_size;
    float learning_rate; // Added in V2
    // Followed by model_data
} EggJobResponseHeader;

static inline void egg_serialize_job_response_header(uint8_t *buf, const EggJobResponseHeader *res) {
    uint64_t s = htonll(res->seed);
    uint64_t l = htonll(res->last_step);
    uint64_t d = htonll(res->data_position);
    uint32_t m = htonl(res->model_size);
    
    uint32_t lr_int;
    memcpy(&lr_int, &res->learning_rate, 4);
    lr_int = htonl(lr_int);
    
    memcpy(buf, &s, 8);
    memcpy(buf + 8, &l, 8);
    memcpy(buf + 16, &d, 8);
    memcpy(buf + 24, &m, 4);
    memcpy(buf + 28, &lr_int, 4);
}

static inline void egg_deserialize_job_response_header(const uint8_t *buf, EggJobResponseHeader *res) {
    uint64_t s, l, d;
    uint32_t m, lr_int;
    memcpy(&s, buf, 8);
    memcpy(&l, buf + 8, 8);
    memcpy(&d, buf + 16, 8);
    memcpy(&m, buf + 24, 4);
    memcpy(&lr_int, buf + 28, 4);
    
    res->seed = ntohll(s);
    res->last_step = ntohll(l);
    res->data_position = ntohll(d);
    res->model_size = ntohl(m);
    
    lr_int = ntohl(lr_int);
    memcpy(&res->learning_rate, &lr_int, 4);
}

// RESULT Header (44 bytes + result_data)
typedef struct {
    uint64_t seed;
    uint64_t last_step;
    uint64_t data_position;
    uint64_t updates_count; // Added field
    uint64_t sum_loss;      // Added field (int64_t packed as uint64_t)
    uint32_t result_size;
    // Followed by result_data
} EggResultHeader;

static inline void egg_serialize_result_header(uint8_t *buf, const EggResultHeader *res) {
    uint64_t s = htonll(res->seed);
    uint64_t l = htonll(res->last_step);
    uint64_t d = htonll(res->data_position);
    uint64_t u = htonll(res->updates_count);
    uint64_t sl = htonll(res->sum_loss);
    uint32_t r = htonl(res->result_size);
    memcpy(buf, &s, 8);
    memcpy(buf + 8, &l, 8);
    memcpy(buf + 16, &d, 8);
    memcpy(buf + 24, &u, 8);
    memcpy(buf + 32, &sl, 8);
    memcpy(buf + 40, &r, 4);
}

static inline void egg_deserialize_result_header(const uint8_t *buf, EggResultHeader *res) {
    uint64_t s, l, d, u, sl;
    uint32_t r;
    memcpy(&s, buf, 8);
    memcpy(&l, buf + 8, 8);
    memcpy(&d, buf + 16, 8);
    memcpy(&u, buf + 24, 8);
    memcpy(&sl, buf + 32, 8);
    memcpy(&r, buf + 40, 4);
    res->seed = ntohll(s);
    res->last_step = ntohll(l);
    res->data_position = ntohll(d);
    res->updates_count = ntohll(u);
    res->sum_loss = ntohll(sl);
    res->result_size = ntohl(r);
}

#endif // EGG_PROTOCOL_H
