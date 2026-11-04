// The wire format between strata-dsv41's CPU lane and an
// expert daemon (k40_expertd.cu); the message order is in
// README.md. One TCP connection, little-endian.
#pragma once
#include <cstdint>

namespace strata::dsv41::expertd {

constexpr uint32_t kMagic = 0x3144584Bu;   // "KXD1"
constexpr uint32_t kVersion = 3;
constexpr int kPort = 30060;
constexpr int kMaxT = 16;        // (the lane's window, kWin)
constexpr int kMaxPairs = 128;   // (kWin x kPicks)

#pragma pack(push, 1)
struct Hello {
    uint32_t magic, version, dim, ff;
    float lim;       // the SwiGLU clamp (0: none)
    uint32_t role;   // kCompute, or kPlacer (swaps only)
};
constexpr uint32_t kCompute = 0, kPlacer = 1;
struct HelloReply {
    uint32_t magic, version;
    uint64_t free_bytes;
};
struct Id {
    uint16_t blk, e;
};
// placer -> daemon: replace a held expert; daemon -> ReplyHdr
// (mask 1: done, 2: refused and the old kept, 3: both held
// already, 0: both gone)
struct Swap {
    uint32_t seq;
    Id out, in;
};
struct ComputeHdr {
    uint32_t seq;
    uint16_t blk, T, np, pad;
};
struct Pair {
    uint16_t e, tok;
    float w;   // the route weight (routed scaling included)
};
struct ReplyHdr {
    uint32_t seq, mask;
    uint32_t us;   // the daemon's time on it (receive to reply)
};
#pragma pack(pop)

}  // namespace strata::dsv41::expertd
