// k40_expertd: DeepSeek-V4.1 routed experts on a CUDA GPU
// (built for a Tesla K40), computed for strata-dsv41's CPU
// lane over TCP. See README.md and expertd_proto.hpp.
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "../ds4_iq.cuh"
#include "expertd_proto.hpp"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"   // iq1s_grid_gpu (bp_convert)

namespace iq = strata::ds4::gguf::iq;
namespace xp = strata::dsv41::expertd;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "k40_expertd: %s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// ------------------------------------------------ GGUF tensor index
struct Tensor {
    std::string file;
    int type = -1;
    uint64_t dims[4] = {0, 0, 0, 0};
    uint64_t off = 0;   // absolute file offset of the data
};

static bool rd(FILE* f, void* p, size_t n) { return std::fread(p, 1, n, f) == n; }
static bool rstr(FILE* f, std::string& s) {
    uint64_t n = 0;
    if (!rd(f, &n, 8) || n > (1u << 20)) return false;
    s.resize(n);
    return rd(f, s.data(), n);
}
static bool skip_val(FILE* f, uint32_t t) {
    static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t <= 12 && t != 8 && t != 9) return std::fseek(f, sz[t], SEEK_CUR) == 0;
    if (t == 8) { std::string s; return rstr(f, s); }
    if (t == 9) {
        uint32_t at = 0;
        uint64_t n = 0;
        if (!rd(f, &at, 4) || !rd(f, &n, 8)) return false;
        for (uint64_t i = 0; i < n; ++i)
            if (!skip_val(f, at)) return false;
        return true;
    }
    return false;
}
static bool index_gguf(const std::string& path, std::map<std::string, Tensor>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char magic[4];
    uint32_t ver = 0, align = 32;
    uint64_t nt = 0, nkv = 0;
    bool ok = rd(f, magic, 4) && std::memcmp(magic, "GGUF", 4) == 0 && rd(f, &ver, 4) && rd(f, &nt, 8) && rd(f, &nkv, 8);
    for (uint64_t i = 0; ok && i < nkv; ++i) {
        std::string k;
        uint32_t t = 0;
        ok = rstr(f, k) && rd(f, &t, 4);
        if (ok && k == "general.alignment" && t == 4) ok = rd(f, &align, 4);
        else if (ok) ok = skip_val(f, t);
    }
    std::vector<std::pair<std::string, Tensor>> ts;
    for (uint64_t i = 0; ok && i < nt; ++i) {
        std::string name;
        uint32_t nd = 0, type = 0;
        Tensor tn;
        ok = rstr(f, name) && rd(f, &nd, 4) && nd <= 4;
        for (uint32_t d = 0; ok && d < nd; ++d) ok = rd(f, &tn.dims[d], 8);
        ok = ok && rd(f, &type, 4) && rd(f, &tn.off, 8);
        tn.type = (int) type;
        tn.file = path;
        ts.emplace_back(name, tn);
    }
    const uint64_t base = ((uint64_t) std::ftell(f) + align - 1) / align * align;
    std::fclose(f);
    if (!ok) return false;
    for (auto& [name, tn] : ts) {
        tn.off += base;
        out[name] = tn;
    }
    return true;
}

// ------------------------------------------------ kernels
// h = clamped silu(g) u (the engine's swiglu), int8 a 32
__global__ void swiglu_q8(const float* gu, int ff, float lim, int8_t* hq, float* hd) {
    const int p = blockIdx.y, b = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (b * 32 >= ff) return;
    float g = gu[(size_t) p * 2 * ff + b * 32 + lane];
    float u = gu[(size_t) p * 2 * ff + ff + b * 32 + lane];
    if (lim > 0.0f) {
        g = fminf(g, lim);
        u = fminf(fmaxf(u, -lim), lim);
    }
    const float h = g / (1.0f + expf(-g)) * u;
    float a = fabsf(h);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float sc = a / 127.0f;
    hq[(size_t) p * ff + b * 32 + lane] = (int8_t) (a > 0.0f ? __float2int_rn(h / sc) : 0);
    if (lane == 0) hd[(size_t) p * (ff / 32) + b] = sc;
}
// IQ1_S rows -> bit planes: a sub-block's +1 and -1 masks,
// meta (3-bit scale | 8: negative delta), the block's fp16 d
__global__ void bp_convert(const uint8_t* raw, size_t ld, int k, int n, uint2* pl, uint16_t* mt, uint16_t* dh) {
    const int nsb = k / 32;
    const size_t s = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= (size_t) n * nsb) return;
    const int row = (int) (s / nsb), kb = (int) (s % nsb), ib = kb & 7;
    const uint8_t* b = raw + (size_t) row * ld + (size_t) (kb >> 3) * 50;
    const uint16_t qh = (uint16_t) (b[34 + 2 * ib] | (b[35 + 2 * ib] << 8));
    uint32_t pm = 0, mm = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t g = iq1s_grid_gpu[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8)];
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo = (g >> (8 * m)) & 15u, hi = (g >> (8 * m + 4)) & 15u;   // (0: -1, 2: +1)
            if (lo == 2) pm |= 1u << (8 * l + m);
            if (lo == 0) mm |= 1u << (8 * l + m);
            if (hi == 2) pm |= 1u << (8 * l + 4 + m);
            if (hi == 0) mm |= 1u << (8 * l + 4 + m);
        }
    }
    pl[s] = make_uint2(pm, mm);
    mt[s] = (uint16_t) (((qh >> 12) & 7u) | ((qh & 0x8000u) ? 8u : 0u));
    if (ib == 0) dh[(size_t) row * (nsb / 8) + (kb >> 3)] = (uint16_t) (b[0] | (b[1] << 8));
}
// IQ1_M rows -> the same planes; meta: s0 (bits 0-2), s1 (3-5),
// the 4 groups' negative deltas (6-9); d (iq1m_d's) a block
__global__ void bp_convert_m(const uint8_t* raw, size_t ld, int k, int n, uint2* pl, uint16_t* mt, uint16_t* dh) {
    const int nsb = k / 32;
    const size_t s = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= (size_t) n * nsb) return;
    const int row = (int) (s / nsb), kb = (int) (s % nsb), ib = kb & 7;
    const uint8_t* b = raw + (size_t) row * ld + (size_t) (kb >> 3) * 56;
    const uint8_t* qs = b + 4 * ib;
    const uint32_t qh0 = b[32 + 2 * ib], qh1 = b[33 + 2 * ib];
    const uint32_t sc = (uint32_t) b[48 + 2 * (ib >> 1)] | ((uint32_t) b[49 + 2 * (ib >> 1)] << 8);
    const int sh = 6 * (ib & 1);
    const uint32_t ii[4] = {qs[0] | ((qh0 << 8) & 0x700u), qs[1] | ((qh0 << 4) & 0x700u),
                            qs[2] | ((qh1 << 8) & 0x700u), qs[3] | ((qh1 << 4) & 0x700u)};
    const uint32_t ng = ((qh0 >> 3) & 1u) | ((qh0 >> 6) & 2u) | (((qh1 >> 3) & 1u) << 2) | (((qh1 >> 7) & 1u) << 3);
    uint32_t pm = 0, mm = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t g = iq1s_grid_gpu[ii[l]];
        for (int m = 0; m < 4; ++m) {
            const uint32_t lo = (g >> (8 * m)) & 15u, hi = (g >> (8 * m + 4)) & 15u;   // (0: -1, 2: +1)
            if (lo == 2) pm |= 1u << (8 * l + m);
            if (lo == 0) mm |= 1u << (8 * l + m);
            if (hi == 2) pm |= 1u << (8 * l + 4 + m);
            if (hi == 0) mm |= 1u << (8 * l + 4 + m);
        }
    }
    pl[s] = make_uint2(pm, mm);
    mt[s] = (uint16_t) (((sc >> sh) & 7u) | (((sc >> (sh + 3)) & 7u) << 3) | (ng << 6));
    if (ib == 0) {   // (iq1m_d: the scales' top nibbles)
        const uint32_t s0 = b[48] | (b[49] << 8), s1 = b[50] | (b[51] << 8), s2 = b[52] | (b[53] << 8),
                       s3 = b[54] | (b[55] << 8);
        dh[(size_t) row * (nsb / 8) + (kb >> 3)] =
            (uint16_t) ((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
    }
}
// rows of int8 (k each) -> 8 bit planes a sub-block, its sum,
// and its 4 groups' sums (s8, for IQ1_M's per-8 deltas)
__global__ void bp_xplanes(const int8_t* xq, int k, int rows, uint32_t* xp, int* xs, int* s8) {
    const int nsb = k / 32, lane = threadIdx.x & 31;
    const int s = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (s >= nsb * rows) return;
    const int v = xq[(size_t) s * 32 + lane];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const uint32_t m = __ballot_sync(0xffffffffu, (v >> t) & 1);
        if (lane == 0) xp[(size_t) s * 8 + t] = m;
    }
    int sum = v;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((lane & 7) == 0) s8[(size_t) s * 4 + (lane >> 3)] = sum;
#pragma unroll
    for (int o = 16; o > 4; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) xs[s] = sum;
}
struct BpJob {
    const uint2* pl;
    const uint16_t *mt, *dh;
    const uint32_t* xp;
    const int* xs;
    const float* xd;
    float* y;
    int k, n;
    const int* s8;   // (IQ1_M: the groups' sums)
};
__device__ __forceinline__ int bp_w(uint2 pm, uint4 xa, uint4 xb) {
    int W = (__popc(pm.x & xa.x) - __popc(pm.y & xa.x));
    W += (__popc(pm.x & xa.y) - __popc(pm.y & xa.y)) << 1;
    W += (__popc(pm.x & xa.z) - __popc(pm.y & xa.z)) << 2;
    W += (__popc(pm.x & xa.w) - __popc(pm.y & xa.w)) << 3;
    W += (__popc(pm.x & xb.x) - __popc(pm.y & xb.x)) << 4;
    W += (__popc(pm.x & xb.y) - __popc(pm.y & xb.y)) << 5;
    W += (__popc(pm.x & xb.z) - __popc(pm.y & xb.z)) << 6;
    W -= (__popc(pm.x & xb.w) - __popc(pm.y & xb.w)) << 7;
    return W;
}
// job blockIdx.y: y = rows . x over bit planes, P lanes a row;
// IQ1_S's integer dot exactly (the bits of iq::mv's Kepler path)
template <int P>
__global__ void __launch_bounds__(256) bp_mv(const BpJob* __restrict__ jobs) {
    const BpJob J = jobs[blockIdx.y];
    const int lane = threadIdx.x & (P - 1);
    const int row = blockIdx.x * (256 / P) + (int) threadIdx.x / P;
    const int nsb = J.k / 32;
    const bool live = row < J.n;
    float acc = 0.0f;
    if (live) {
        const uint2* p = J.pl + (size_t) row * nsb + lane;
        const uint16_t* m = J.mt + (size_t) row * nsb + lane;
        const uint16_t* dr = J.dh + (size_t) row * (nsb / 8);
        const uint4* xv = (const uint4*) J.xp + 2 * lane;
        const int* sv = J.xs + lane;
        const float* dv = J.xd + lane;
        for (int kb = lane; kb < nsb; kb += P) {
            const uint4 xa = __ldg(xv), xb = __ldg(xv + 1);
            const uint2 pm = __ldg(p);
            const uint32_t mt = __ldg(m);
            const int S = __ldg(sv);
            const float dx = __ldg(dv);
            const float d = __half2float(__ushort_as_half(__ldg(dr + (kb >> 3)))) * 0.125f;
            const int L = (bp_w(pm, xa, xb) << 3) + ((mt & 8u) ? -S : S);
            const float sc = (float) (2 * (mt & 7u) + 1);
            acc = fmaf(dx, __fmul_rn(d, sc * (float) L), acc);
            p += P; m += P; xv += 2 * P; sv += P; dv += P;
        }
    }
#pragma unroll
    for (int o = P / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0 && live) J.y[row] = acc;
}
// IQ1_M: mv's d (s0 lo + s1 hi), lo / hi the halves' integer
// dots, each 8 (W) plus its groups' signed sums
template <int P>
__global__ void __launch_bounds__(256) bp_mv_m(const BpJob* __restrict__ jobs) {
    const BpJob J = jobs[blockIdx.y];
    const int lane = threadIdx.x & (P - 1);
    const int row = blockIdx.x * (256 / P) + (int) threadIdx.x / P;
    const int nsb = J.k / 32;
    const bool live = row < J.n;
    float acc = 0.0f;
    if (live) {
        const uint2* p = J.pl + (size_t) row * nsb + lane;
        const uint16_t* m = J.mt + (size_t) row * nsb + lane;
        const uint16_t* dr = J.dh + (size_t) row * (nsb / 8);
        const uint4* xv = (const uint4*) J.xp + 2 * lane;
        const int4* sv = (const int4*) J.s8 + lane;
        const float* dv = J.xd + lane;
        for (int kb = lane; kb < nsb; kb += P) {
            const uint4 xa = __ldg(xv), xb = __ldg(xv + 1);
            const uint2 pm = __ldg(p);
            const uint32_t mt = __ldg(m);
            const int4 S = __ldg(sv);
            const float dx = __ldg(dv);
            const float d = __half2float(__ushort_as_half(__ldg(dr + (kb >> 3)))) * 0.125f;
            const int wl = bp_w(make_uint2(pm.x & 0xffffu, pm.y & 0xffffu), xa, xb);
            const int wh = bp_w(make_uint2(pm.x & 0xffff0000u, pm.y & 0xffff0000u), xa, xb);
            const int lo = (wl << 3) + ((mt & 64u) ? -S.x : S.x) + ((mt & 128u) ? -S.y : S.y);
            const int hi = (wh << 3) + ((mt & 256u) ? -S.z : S.z) + ((mt & 512u) ? -S.w : S.w);
            const int s0 = 2 * (int) (mt & 7u) + 1, s1 = 2 * (int) ((mt >> 3) & 7u) + 1;
            acc = fmaf(dx, __fmul_rn(d, (float) (s0 * lo + s1 * hi)), acc);
            p += P; m += P; xv += 2 * P; sv += P; dv += P;
        }
    }
#pragma unroll
    for (int o = P / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0 && live) J.y[row] = acc;
}
// y[t] = sum of w[p] part[p] over the pairs p of token t
__global__ void wsum(const float* part, const xp::Pair* pairs, int np, int n, float* y) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int p = 0; p < np; ++p)
        if (pairs[p].tok == t) s = fmaf(pairs[p].w, part[(size_t) p * n + i], s);
    y[(size_t) t * n + i] = s;
}

// ------------------------------------------------ socket I/O
static bool send_all(int fd, const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const ssize_t k = ::send(fd, c, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        c += k;
        n -= (size_t) k;
    }
    return true;
}
static bool recv_all(int fd, void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const ssize_t k = ::recv(fd, c, n, 0);
        if (k <= 0) return false;
        c += k;
        n -= (size_t) k;
    }
    return true;
}

// ------------------------------------------------ the server
struct LayerW {
    bool ok = false;   // gate, up and down all IQ1_S, n_embd x n_ff
    const Tensor *g = nullptr, *u = nullptr, *d = nullptr;
};
struct Mat {   // IQ1_S as bit planes (bp_convert's), else raw
    uint2* pl;
    uint16_t *mt, *dh;
    int type = iq::kIQ1_S;
    const uint8_t* raw = nullptr;   // (raw: for iq::mv)
    size_t ld = 0;
};
struct Held {
    Mat g, u, d;
    uint8_t* base;   // (its arena slot, reused by a swap)
};
// a loader's buffers: the raw expert on the host and the device
struct Loader {
    std::vector<uint8_t> host;
    uint8_t* stage = nullptr;
    cudaStream_t s = nullptr;
};
// bytes of an n x k matrix's planes, meta and d, in that order
static size_t bp_bytes(int k, int n) { return (size_t) n * (k / 32) * 10 + (size_t) n * (k / 256) * 2; }
static Mat bp_at(uint8_t* base, int k, int n) {
    const size_t s = (size_t) n * (k / 32);
    return {(uint2*) base, (uint16_t*) (base + s * 8), (uint16_t*) (base + s * 10)};
}
// the types kept as bit planes (the ternary ones)
static bool is_bp(int t) { return t == iq::kIQ1_S || t == iq::kIQ1_M; }
static Mat raw_at(uint8_t* base, int type, int k) {
    Mat m{nullptr, nullptr, nullptr};
    m.type = type;
    m.raw = base;
    m.ld = iq::row_bytes(type, k);
    return m;
}

constexpr int kJobRegions = 4, kJobsRegion = 2 * xp::kMaxPairs;

struct Server {
    std::map<std::string, Tensor> idx;
    std::vector<LayerW> layers;
    std::map<uint32_t, Held> held;   // (blk << 16 | e), under mu
    std::mutex mu;
    std::vector<uint8_t*> arenas, free_slots;
    Loader boot;   // (the first assignment's)
    int dim = 5120, ff = 2304;
    float lim = 0.0f;
    size_t ldg = 0, ldd = 0, mg = 0, md = 0;
    // scratch
    std::vector<uint8_t> hin;
    uint8_t* pin = nullptr;    // pinned request copy
    float* pout = nullptr;     // pinned reply
    uint8_t* din = nullptr;    // request on the device
    float *dgu = nullptr, *dhd = nullptr, *dpart = nullptr, *dy = nullptr;
    int8_t* dh = nullptr;
    uint8_t* stage = nullptr;  // an expert's IQ1_S, to convert
    uint32_t *xpl = nullptr, *hpl = nullptr;   // x / h bit planes
    int *xsum = nullptr, *hsum = nullptr, *xs8 = nullptr, *hs8 = nullptr;
    BpJob *hjobs = nullptr, *djobs = nullptr;   // pinned / device
    cudaStream_t s = nullptr;

    void setup_layers() {
        for (int b = 0; b < 256; ++b) {
            const std::string p = "blk." + std::to_string(b) + ".ffn_";
            auto gi = idx.find(p + "gate_exps.weight"), ui = idx.find(p + "up_exps.weight"),
                 di = idx.find(p + "down_exps.weight");
            if (gi == idx.end() || ui == idx.end() || di == idx.end()) continue;
            if ((int) layers.size() <= b) layers.resize((size_t) b + 1);
            LayerW& L = layers[(size_t) b];
            L.g = &gi->second; L.u = &ui->second; L.d = &di->second;
            // (IQ1_S: bit planes; the other types iq::mv takes: raw)
            L.ok = iq::supported(L.g->type) && iq::supported(L.u->type) && iq::supported(L.d->type) &&
                   !iq::is_tq(L.g->type) && !iq::is_tq(L.u->type) && !iq::is_tq(L.d->type) &&
                   L.g->dims[0] == (uint64_t) dim && L.g->dims[1] == (uint64_t) ff &&
                   L.d->dims[0] == (uint64_t) ff && L.d->dims[1] == (uint64_t) dim;
            if (!L.ok) continue;
            const Tensor* t3[3] = {L.g, L.u, L.d};
            for (int r = 0; r < 3; ++r) {
                const int k = r < 2 ? dim : ff, n = r < 2 ? ff : dim;
                slot_mat = std::max(slot_mat, is_bp(t3[r]->type) ? bp_bytes(k, n) : iq::row_bytes(t3[r]->type, k) * n);
                raw_max = std::max(raw_max, iq::row_bytes(t3[r]->type, k) * n);
            }
        }
        slot_mat = (slot_mat + 255) / 256 * 256;
    }
    size_t slot_mat = 0, raw_max = 0;   // (a matrix's slot, its largest raw form)
    // an expert from the files into dst (an arena slot, three
    // slot_mat parts): IQ1_S as bit planes, others raw; then held
    bool load_one(int blk, int e, uint8_t* dst, Loader& ld) {
        if (blk >= (int) layers.size() || !layers[(size_t) blk].ok || (uint64_t) e >= layers[(size_t) blk].g->dims[2])
            return false;
        const LayerW& L = layers[(size_t) blk];
        const Tensor* t3[3] = {L.g, L.u, L.d};
        Held h{};
        h.base = dst;
        Mat* m3[3] = {&h.g, &h.u, &h.d};
        ld.host.resize(raw_max);
        for (int r = 0; r < 3; ++r) {
            const int k = r < 2 ? dim : ff, n = r < 2 ? ff : dim, ty = t3[r]->type;
            const size_t ld_r = iq::row_bytes(ty, k), sz = ld_r * n;
            const int fd = open(t3[r]->file.c_str(), O_RDONLY);
            if (fd < 0) return false;
            const ssize_t got = pread(fd, ld.host.data(), sz, (off_t) (t3[r]->off + (uint64_t) e * sz));
            close(fd);
            if (got != (ssize_t) sz) return false;
            uint8_t* part = dst + (size_t) r * slot_mat;
            if (is_bp(ty)) {
                CK(cudaMemcpyAsync(ld.stage, ld.host.data(), sz, cudaMemcpyHostToDevice, ld.s));
                *m3[r] = bp_at(part, k, n);
                m3[r]->type = ty;
                const size_t ns = (size_t) n * (k / 32);
                if (ty == iq::kIQ1_S)
                    bp_convert<<<(unsigned) ((ns + 255) / 256), 256, 0, ld.s>>>(ld.stage, ld_r, k, n, m3[r]->pl,
                                                                                m3[r]->mt, m3[r]->dh);
                else
                    bp_convert_m<<<(unsigned) ((ns + 255) / 256), 256, 0, ld.s>>>(ld.stage, ld_r, k, n, m3[r]->pl,
                                                                                  m3[r]->mt, m3[r]->dh);
            } else {
                CK(cudaMemcpyAsync(part, ld.host.data(), sz, cudaMemcpyHostToDevice, ld.s));
                *m3[r] = raw_at(part, ty, k);
            }
            CK(cudaStreamSynchronize(ld.s));   // (ld.host is reused)
        }
        std::lock_guard<std::mutex> lk(mu);
        held[(uint32_t) blk << 16 | (uint32_t) e] = h;
        return true;
    }
    void loader_init(Loader& ld) {
        if (ld.stage != nullptr) return;
        CK(cudaMalloc(&ld.stage, raw_max));
        CK(cudaStreamCreateWithFlags(&ld.s, cudaStreamNonBlocking));
    }
    // the placer's connection: swaps, one at a time
    void place(int fd) {
        Loader ld;
        loader_init(ld);
        long done = 0;
        for (;;) {
            xp::Swap sw{};
            if (!recv_all(fd, &sw, sizeof(sw))) break;
            xp::ReplyHdr rh{sw.seq, 0u, 0u};
            const auto t0 = std::chrono::steady_clock::now();
            uint8_t* slot = nullptr;
            bool have_in = false;
            const bool can = sw.in.blk < layers.size() && layers[sw.in.blk].ok &&
                             sw.in.e < layers[sw.in.blk].g->dims[2];
            if (can) {
                std::lock_guard<std::mutex> lk(mu);
                auto it = held.find((uint32_t) sw.out.blk << 16 | sw.out.e);
                have_in = held.count((uint32_t) sw.in.blk << 16 | sw.in.e) != 0;
                if (it != held.end() && !have_in) {
                    slot = it->second.base;
                    held.erase(it);
                }
            }
            if (have_in) {
                rh.mask = 3;   // (both held already)
            } else if (slot == nullptr) {
                rh.mask = 2;   // refused: the old one is still held
            } else if (load_one(sw.in.blk, sw.in.e, slot, ld)) {
                rh.mask = 1;
                ++done;
            } else {
                free_slots.push_back(slot);   // (a read failed: both gone)
            }
            rh.us = (uint32_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            if (!send_all(fd, &rh, sizeof(rh))) break;
        }
        std::fprintf(stderr, "k40_expertd: placer gone after %ld swaps\n", done);
    }
    // the candidates in rank order, while memory lasts
    std::vector<xp::Id> assign(const std::vector<xp::Id>& want, size_t reserve) {
        std::vector<xp::Id> got;
        const size_t per = 3 * slot_mat;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t budget = fr > reserve ? fr - reserve : 0;
        const size_t n_max = budget / per;
        std::vector<xp::Id> ok;
        for (const xp::Id& w : want)
            if (w.blk < layers.size() && layers[w.blk].ok && w.e < layers[w.blk].g->dims[2] &&
                ok.size() < n_max)
                ok.push_back(w);
        // one arena per 256 experts (smaller allocations)
        loader_init(boot);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t a = 0; a < ok.size(); a += 256) {
            const size_t n = std::min<size_t>(256, ok.size() - a);
            uint8_t* ar = nullptr;
            if (cudaMalloc(&ar, n * per) != cudaSuccess) break;
            arenas.push_back(ar);
            for (size_t i = 0; i < n; ++i)
                if (load_one(ok[a + i].blk, ok[a + i].e, ar + i * per, boot)) got.push_back(ok[a + i]);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "k40_expertd: holding %zu of %zu candidates (%.2f GB) in %.1f s\n", got.size(),
                     want.size(), got.size() * per / 1e9, s);
        return got;
    }
    void alloc_scratch() {
        const size_t in_max = sizeof(xp::ComputeHdr) + (size_t) xp::kMaxT * (dim + dim / 8) +
                              (size_t) xp::kMaxPairs * sizeof(xp::Pair);
        CK(cudaMallocHost(&pin, in_max));
        CK(cudaMallocHost(&pout, (size_t) xp::kMaxT * dim * 4));
        CK(cudaMalloc(&din, in_max));
        CK(cudaMalloc(&dgu, (size_t) xp::kMaxPairs * 2 * ff * 4));
        CK(cudaMalloc(&dh, (size_t) xp::kMaxPairs * ff));
        CK(cudaMalloc(&dhd, (size_t) xp::kMaxPairs * ff / 8));
        CK(cudaMalloc(&dpart, (size_t) xp::kMaxPairs * dim * 4));
        CK(cudaMalloc(&dy, (size_t) xp::kMaxT * dim * 4));
        CK(cudaMalloc(&xpl, (size_t) xp::kMaxT * (dim / 32) * 32));
        CK(cudaMalloc(&xsum, (size_t) xp::kMaxT * (dim / 32) * 4));
        CK(cudaMalloc(&hpl, (size_t) xp::kMaxPairs * (ff / 32) * 32));
        CK(cudaMalloc(&hsum, (size_t) xp::kMaxPairs * (ff / 32) * 4));
        CK(cudaMalloc(&xs8, (size_t) xp::kMaxT * (dim / 32) * 16));
        CK(cudaMalloc(&hs8, (size_t) xp::kMaxPairs * (ff / 32) * 16));
        CK(cudaMallocHost(&hjobs, (size_t) kJobRegions * kJobsRegion * sizeof(BpJob)));
        CK(cudaMalloc(&djobs, (size_t) kJobRegions * kJobsRegion * sizeof(BpJob)));
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    // one request: false drops the connection
    bool compute(int fd, const xp::ComputeHdr& h) {
        if (h.T == 0 || h.T > xp::kMaxT || h.np == 0 || h.np > xp::kMaxPairs) return false;
        const size_t xbytes = (size_t) h.T * dim, dbytes = (size_t) h.T * (dim / 32) * 4;
        const size_t pbytes = (size_t) h.np * sizeof(xp::Pair);
        if (!recv_all(fd, pin, xbytes + dbytes + pbytes)) return false;
        const auto t0 = std::chrono::steady_clock::now();
        const xp::Pair* pairs = (const xp::Pair*) (pin + xbytes + dbytes);
        xp::ReplyHdr rh{h.seq, 0u, 0u};
        // jobs: bit-plane regions (gate/up IQ1_S, IQ1_M; down IQ1_S,
        // IQ1_M), and the raw ones for iq::mv (jg, jd)
        const int nsx = dim / 32, nsh = ff / 32;
        const float* xd_dev = (const float*) (din + xbytes);
        const int8_t* xq_dev = (const int8_t*) din;
        int nr[kJobRegions] = {};
        std::vector<iq::MvJob> jg, jd;
        auto put = [&](int reg, const BpJob& j) { hjobs[(size_t) reg * kJobsRegion + nr[reg]++] = j; };
        std::unique_lock<std::mutex> lk(mu);   // (the lookups only)
        for (int p = 0; p < h.np; ++p) {
            const auto it = held.find((uint32_t) h.blk << 16 | pairs[p].e);
            if (it == held.end() || pairs[p].tok >= h.T) return false;
            const int t = pairs[p].tok;
            rh.mask |= 1u << t;
            for (int r = 0; r < 2; ++r) {
                const Mat& M = r ? it->second.u : it->second.g;
                float* y = dgu + (size_t) p * 2 * ff + (size_t) r * ff;
                if (M.raw == nullptr) {
                    put(M.type == iq::kIQ1_S ? 0 : 1,
                        {M.pl, M.mt, M.dh, xpl + (size_t) t * nsx * 8, xsum + (size_t) t * nsx, xd_dev + (size_t) t * nsx,
                         y, dim, ff, xs8 + (size_t) t * nsx * 4});
                } else {
                    iq::MvJob j;
                    j.w = M.raw; j.ld = M.ld; j.type = M.type; j.k = dim; j.n = ff;
                    j.xq = xq_dev + (size_t) t * dim; j.xd = xd_dev + (size_t) t * nsx; j.y = y; j.ycs = ff;
                    jg.push_back(j);
                }
            }
            const Mat& D = it->second.d;
            float* y = dpart + (size_t) p * dim;
            if (D.raw == nullptr) {
                put(D.type == iq::kIQ1_S ? 2 : 3,
                    {D.pl, D.mt, D.dh, hpl + (size_t) p * nsh * 8, hsum + (size_t) p * nsh, dhd + (size_t) p * nsh, y, ff,
                     dim, hs8 + (size_t) p * nsh * 4});
            } else {
                iq::MvJob j;
                j.w = D.raw; j.ld = D.ld; j.type = D.type; j.k = ff; j.n = dim;
                j.xq = dh + (size_t) p * ff; j.xd = dhd + (size_t) p * nsh; j.y = y; j.ycs = dim;
                jd.push_back(j);
            }
        }
        lk.unlock();
        CK(cudaMemcpyAsync(din, pin, xbytes + dbytes + pbytes, cudaMemcpyHostToDevice, s));
        for (int g = 0; g < kJobRegions; ++g)
            if (nr[g] > 0)
                CK(cudaMemcpyAsync(djobs + (size_t) g * kJobsRegion, hjobs + (size_t) g * kJobsRegion,
                                   (size_t) nr[g] * sizeof(BpJob), cudaMemcpyHostToDevice, s));
        if (nr[0] + nr[1] > 0) bp_xplanes<<<(unsigned) ((nsx * h.T + 7) / 8), 256, 0, s>>>(xq_dev, dim, h.T, xpl, xsum, xs8);
        if (nr[0] > 0) bp_mv<32><<<dim3((unsigned) ((ff + 7) / 8), nr[0]), 256, 0, s>>>(djobs);
        if (nr[1] > 0) bp_mv_m<32><<<dim3((unsigned) ((ff + 7) / 8), nr[1]), 256, 0, s>>>(djobs + kJobsRegion);
        for (size_t a = 0; a < jg.size(); a += 64)
            if (!iq::mv(jg.data() + a, (int) std::min<size_t>(64, jg.size() - a), s)) return false;
        swiglu_q8<<<dim3((unsigned) (ff / 32 / 8), h.np), dim3(32, 8), 0, s>>>(dgu, ff, lim, dh, dhd);
        if (nr[2] + nr[3] > 0) bp_xplanes<<<(unsigned) ((nsh * h.np + 7) / 8), 256, 0, s>>>(dh, ff, h.np, hpl, hsum, hs8);
        if (nr[2] > 0) bp_mv<8><<<dim3((unsigned) ((dim + 31) / 32), nr[2]), 256, 0, s>>>(djobs + 2 * kJobsRegion);
        if (nr[3] > 0) bp_mv_m<8><<<dim3((unsigned) ((dim + 31) / 32), nr[3]), 256, 0, s>>>(djobs + 3 * kJobsRegion);
        for (size_t a = 0; a < jd.size(); a += 64)
            if (!iq::mv(jd.data() + a, (int) std::min<size_t>(64, jd.size() - a), s)) return false;
        const xp::Pair* dpairs = (const xp::Pair*) (din + xbytes + dbytes);
        wsum<<<dim3((unsigned) ((dim + 255) / 256), h.T), 256, 0, s>>>(dpart, dpairs, h.np, dim, dy);
        CK(cudaMemcpyAsync(pout, dy, (size_t) h.T * dim * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        rh.us = (uint32_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        if (!send_all(fd, &rh, sizeof(rh))) return false;
        for (int t = 0; t < h.T; ++t)
            if ((rh.mask >> t) & 1u)
                if (!send_all(fd, pout + (size_t) t * dim, (size_t) dim * 4)) return false;
        return true;
    }
    void serve(int fd) {
        xp::Hello hi{};
        if (!recv_all(fd, &hi, sizeof(hi)) || hi.magic != xp::kMagic || hi.version != xp::kVersion) {
            std::fprintf(stderr, "k40_expertd: a client with another protocol\n");
            return;
        }
        if (!held.empty() && (hi.dim != (uint32_t) dim || hi.ff != (uint32_t) ff)) return;
        if (hi.role == xp::kPlacer) {   // (after a compute client's assignment)
            if (held.empty()) return;
            xp::HelloReply hr{xp::kMagic, xp::kVersion, 0};
            if (send_all(fd, &hr, sizeof(hr))) place(fd);
            return;
        }
        dim = (int) hi.dim; ff = (int) hi.ff; lim = hi.lim;
        ldg = iq::row_bytes(iq::kIQ1_S, dim); ldd = iq::row_bytes(iq::kIQ1_S, ff);
        mg = ldg * ff; md = ldd * dim;
        hin.resize(2 * mg + md);
        if (layers.empty()) setup_layers();
        if (pin == nullptr) alloc_scratch();
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        xp::HelloReply hr{xp::kMagic, xp::kVersion, (uint64_t) fr};
        if (!send_all(fd, &hr, sizeof(hr))) return;
        uint32_t n = 0;
        if (!recv_all(fd, &n, 4) || n > (1u << 20)) return;
        std::vector<xp::Id> want(n);
        if (!recv_all(fd, want.data(), n * sizeof(xp::Id))) return;
        std::vector<xp::Id> got;
        if (held.empty()) {
            got = assign(want, (size_t) 256 << 20);
        } else {   // (a reconnect: every expert already held)
            std::lock_guard<std::mutex> lk(mu);
            for (const auto& kv : held) got.push_back({(uint16_t) (kv.first >> 16), (uint16_t) (kv.first & 0xffffu)});
        }
        const uint32_t ng = (uint32_t) got.size();
        if (!send_all(fd, &ng, 4) || !send_all(fd, got.data(), ng * sizeof(xp::Id))) return;
        long served = 0;
        for (;;) {
            xp::ComputeHdr h{};
            if (!recv_all(fd, &h, sizeof(h))) break;
            if (!compute(fd, h)) {
                std::fprintf(stderr, "k40_expertd: a bad request (layer %u, T %u, %u pairs)\n", h.blk, h.T, h.np);
                break;
            }
            ++served;
        }
        std::fprintf(stderr, "k40_expertd: client gone after %ld requests\n", served);
    }
};

// --selftest FILE OFFSET TYPE K N EXPERTS: a bit-plane type's
// kernel against iq::mv on the same rows (bits and time)
static int selftest(char** a) {
    const long long off = std::atoll(a[1]);
    const int type = std::atoi(a[2]), k = std::atoi(a[3]), n = std::atoi(a[4]), ne = std::atoi(a[5]);
    if (!is_bp(type)) { std::fprintf(stderr, "selftest: %s is kept raw\n", iq::name(type)); return 2; }
    const size_t ld = iq::row_bytes(type, k), mat = ld * (size_t) n, bpb = bp_bytes(k, n);
    const int nsb = k / 32;
    std::vector<uint8_t> w(mat * ne);
    const int fd = open(a[0], O_RDONLY);
    if (fd < 0 || pread(fd, w.data(), w.size(), off) != (ssize_t) w.size()) { std::perror("pread"); return 1; }
    close(fd);
    std::vector<int8_t> xq(k);
    std::vector<float> xd(nsb);
    uint32_t r = 12345u;
    for (int b = 0; b < nsb; ++b) {
        xd[b] = 0.001f * (float) (1 + b % 7);
        for (int i = 0; i < 32; ++i) {
            r = r * 1664525u + 1013904223u;
            xq[b * 32 + i] = (int8_t) ((int) (r >> 24) - 128);
        }
    }
    uint8_t *dw, *dbp; int8_t* dxq; float *dxd, *dya, *dyb; uint32_t* dxp; int *dxs, *ds8; BpJob* dj;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dbp, bpb * ne)); CK(cudaMalloc(&dxq, k)); CK(cudaMalloc(&dxd, nsb * 4));
    CK(cudaMalloc(&dya, (size_t) ne * n * 4)); CK(cudaMalloc(&dyb, (size_t) ne * n * 4));
    CK(cudaMalloc(&dxp, nsb * 32)); CK(cudaMalloc(&dxs, nsb * 4)); CK(cudaMalloc(&ds8, nsb * 16));
    CK(cudaMalloc(&dj, ne * sizeof(BpJob)));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxq, xq.data(), k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd, xd.data(), nsb * 4, cudaMemcpyHostToDevice));
    std::vector<iq::MvJob> js(ne);
    std::vector<BpJob> hj(ne);
    for (int e = 0; e < ne; ++e) {
        Mat m = bp_at(dbp + bpb * e, k, n);
        const size_t ns = (size_t) n * nsb;
        if (type == iq::kIQ1_S) bp_convert<<<(unsigned) ((ns + 255) / 256), 256>>>(dw + mat * e, ld, k, n, m.pl, m.mt, m.dh);
        else bp_convert_m<<<(unsigned) ((ns + 255) / 256), 256>>>(dw + mat * e, ld, k, n, m.pl, m.mt, m.dh);
        hj[e] = {m.pl, m.mt, m.dh, dxp, dxs, dxd, dyb + (size_t) e * n, k, n, ds8};
        iq::MvJob& j = js[e];
        j.w = dw + mat * e; j.ld = ld; j.type = type; j.k = k; j.n = n; j.xq = dxq; j.xd = dxd;
        j.y = dya + (size_t) e * n; j.ycs = n;
    }
    CK(cudaMemcpy(dj, hj.data(), ne * sizeof(BpJob), cudaMemcpyHostToDevice));
    bp_xplanes<<<(unsigned) ((nsb + 7) / 8), 256>>>(dxq, k, 1, dxp, dxs, ds8);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timed = [&](auto fn) {
        fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) fn();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms * 1e3 / (20.0 * ne);
    };
    const double tm = timed([&] { for (int e = 0; e < ne; e += 8) iq::mv(js.data() + e, std::min(8, ne - e), 0); });
    std::vector<float> ya((size_t) ne * n), yb((size_t) ne * n);
    CK(cudaMemcpy(ya.data(), dya, ya.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("%s %d x %d: iq::mv %.1f us a matrix\n", iq::name(type), n, k, tm);
    for (int P : {32, 8}) {
        const double tb = timed([&] {
            const dim3 g((unsigned) ((n + 256 / P - 1) / (256 / P)), ne);
            if (type == iq::kIQ1_S) { if (P == 32) bp_mv<32><<<g, 256>>>(dj); else bp_mv<8><<<g, 256>>>(dj); }
            else { if (P == 32) bp_mv_m<32><<<g, 256>>>(dj); else bp_mv_m<8><<<g, 256>>>(dj); }
        });
        CK(cudaMemcpy(yb.data(), dyb, yb.size() * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        double rel = 0.0;
        for (size_t i = 0; i < ya.size(); ++i) {
            diff += std::memcmp(&ya[i], &yb[i], 4) != 0;
            rel = std::max(rel, (double) std::fabs(ya[i] - yb[i]) / (std::fabs(ya[i]) + 1e-3));
        }
        std::printf("  bit planes, %2d lanes a row: %.1f us a matrix, %zu of %zu differ (max rel %.1e)\n", P, tb, diff,
                    ya.size(), rel);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 8 && std::strcmp(argv[1], "--selftest") == 0) return selftest(argv + 2);
    std::string dir;
    int port = xp::kPort;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--model") == 0) dir = argv[i + 1];
        else if (std::strcmp(argv[i], "--port") == 0) port = std::atoi(argv[i + 1]);
    }
    if (dir.empty()) {
        std::fprintf(stderr, "usage: k40_expertd --model DIR_WITH_GGUF_SHARDS [--port %d]\n", xp::kPort);
        return 2;
    }
    Server sv;
    DIR* dp = opendir(dir.c_str());
    if (dp == nullptr) { std::perror(dir.c_str()); return 1; }
    for (dirent* de; (de = readdir(dp)) != nullptr;) {
        const std::string nm = de->d_name;
        if (nm.size() > 5 && nm.compare(nm.size() - 5, 5, ".gguf") == 0)
            if (!index_gguf(dir + "/" + nm, sv.idx)) std::fprintf(stderr, "k40_expertd: cannot read %s\n", nm.c_str());
    }
    closedir(dp);
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, 0));
    std::fprintf(stderr, "k40_expertd: %s (sm_%d%d), %zu tensors indexed, port %d\n", pr.name, pr.major, pr.minor,
                 sv.idx.size(), port);
    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(ls, (sockaddr*) &a, sizeof(a)) != 0 || listen(ls, 4) != 0) { std::perror("bind"); return 1; }
    for (;;) {   // a thread a connection (compute, placer)
        const int fd = accept(ls, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::thread([&sv, fd] {
            sv.serve(fd);
            close(fd);
        }).detach();
    }
}
