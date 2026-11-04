// The CPU lane's side of an expert daemon (README.md): it
// connects and assigns on its own thread; until then, or after a
// failure, holds() is false and the lane computes every expert.
#pragma once
#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "expertd_proto.hpp"

namespace strata::dsv41::expertd {

class Client {
public:
    // addr "host:port"; cand: (layer, expert), hottest first
    void start(const std::string& addr, int dim, int ff, float lim, std::vector<Id> cand, int n_layers, int n_exp) {
        dim_ = dim;
        ff_ = ff;
        lim_ = lim;
        addr_ = addr;
        n_slots_ = (size_t) n_layers * (size_t) n_exp;
        held_.reset(new std::atomic<uint8_t>[n_slots_]);
        for (size_t i = 0; i < n_slots_; ++i) held_[i].store(0, std::memory_order_relaxed);
        n_exp_ = n_exp;
        cand_ = std::move(cand);
        th_ = std::thread([this] { connect_and_assign(cand_); });
    }
    // (the placer's thread) a lost daemon: dial again every 5 s;
    // a restarted one is assigned the first candidates again
    void reconnect() {
        if (ready() || connecting_.load()) return;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_dial_ < std::chrono::seconds(5)) return;
        last_dial_ = now;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        if (pfd_ >= 0) ::close(pfd_);
        pfd_ = -1;
        connect_and_assign(cand_);
    }
    ~Client() {
        if (th_.joinable()) th_.join();
        if (fd_ >= 0) ::close(fd_);
        if (pfd_ >= 0) ::close(pfd_);
    }
    bool ready() const { return ready_.load(std::memory_order_acquire); }
    bool holds(int layer, int e) const {
        if (!ready()) return false;
        const size_t i = (size_t) layer * (size_t) n_exp_ + (size_t) e;
        return i < n_slots_ && held_[i].load(std::memory_order_acquire) != 0;
    }
    size_t held() const { return n_held_; }
    // the placer (its own thread and connection): un-hold before
    // the lane can no longer send it, then swap
    void unhold(int layer, int e) { held_[(size_t) layer * n_exp_ + e].store(0, std::memory_order_release); }
    bool swap(Id out, Id in) {
        if (pfd_ < 0 && !connect_placer()) return false;
        Swap sw{++pseq_, out, in};
        ReplyHdr rh{};
        if (!io_all(pfd_, &sw, sizeof(sw), true) || !io_all(pfd_, &rh, sizeof(rh), false) || rh.seq != sw.seq) {
            ::close(pfd_);
            pfd_ = -1;
            return false;
        }
        if (rh.mask == 1 || rh.mask == 3) held_[(size_t) in.blk * n_exp_ + in.e].store(1, std::memory_order_release);
        if (rh.mask == 2 || rh.mask == 3) held_[(size_t) out.blk * n_exp_ + out.e].store(1, std::memory_order_release);
        last_swap_us = rh.us;
        return rh.mask == 1;
    }
    uint32_t last_swap_us = 0;
    // x: T rows of dim F32 (ldx apart), rounded as q8_quant does
    bool send(uint32_t seq, int layer, int T, const float* x, size_t ldx, const std::vector<Pair>& pairs) {
        if (!ready() || T < 1 || T > kMaxT || pairs.empty() || pairs.size() > (size_t) kMaxPairs) return false;
        const int nb = dim_ / 32;
        buf_.resize(sizeof(ComputeHdr) + (size_t) T * dim_ + (size_t) T * nb * 4 + pairs.size() * sizeof(Pair));
        ComputeHdr h{seq, (uint16_t) layer, (uint16_t) T, (uint16_t) pairs.size(), 0};
        std::memcpy(buf_.data(), &h, sizeof(h));
        int8_t* q = (int8_t*) (buf_.data() + sizeof(h));
        float* d = (float*) (q + (size_t) T * dim_);
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < nb; ++b) {
                const float* v = x + (size_t) t * ldx + (size_t) b * 32;
                float am = 0.0f;
                for (int i = 0; i < 32; ++i) am = std::fmax(am, std::fabs(v[i]));
                const float sc = am / 127.0f;
                d[t * nb + b] = sc;
                for (int i = 0; i < 32; ++i)
                    q[(size_t) t * dim_ + b * 32 + i] = (int8_t) (am > 0.0f ? std::nearbyint(v[i] / sc) : 0.0f);
            }
        std::memcpy(d + (size_t) T * nb, pairs.data(), pairs.size() * sizeof(Pair));
        if (!send_all(buf_.data(), buf_.size())) return fail();
        return true;
    }
    // the reply to seq: out[t] (dim F32, ldo apart) for t in mask
    // us: the daemon's own time on it
    bool recv(uint32_t seq, float* out, size_t ldo, uint32_t& mask, uint32_t& us) {
        ReplyHdr rh{};
        if (!recv_all(&rh, sizeof(rh)) || rh.seq != seq) return fail();
        for (int t = 0; t < kMaxT; ++t)
            if ((rh.mask >> t) & 1u)
                if (!recv_all(out + (size_t) t * ldo, (size_t) dim_ * 4)) return fail();
        mask = rh.mask;
        us = rh.us;
        return true;
    }

private:
    bool fail() {
        if (ready_.exchange(false)) std::fprintf(stderr, "ds4 expertd: lane down, its experts back on the CPU\n");
        for (size_t i = 0; i < n_slots_; ++i) held_[i].store(0, std::memory_order_relaxed);
        return false;   // (reconnect() closes and dials again)
    }
    static bool io_all(int fd, void* p, size_t n, bool out) {
        char* c = (char*) p;
        while (n > 0) {
            const ssize_t k = out ? ::send(fd, c, n, MSG_NOSIGNAL) : ::recv(fd, c, n, 0);
            if (k <= 0) return false;
            c += k;
            n -= (size_t) k;
        }
        return true;
    }
    bool send_all(const void* p, size_t n) { return io_all(fd_, (void*) p, n, true); }
    bool recv_all(void* p, size_t n) { return io_all(fd_, p, n, false); }
    // a connected socket to addr_, or -1
    int dial() {
        const size_t colon = addr_.rfind(':');
        const std::string host = colon == std::string::npos ? addr_ : addr_.substr(0, colon);
        const std::string port = colon == std::string::npos ? std::to_string(kPort) : addr_.substr(colon + 1);
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) return -1;
        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
            ::close(fd);
            fd = -1;
        }
        freeaddrinfo(res);
        if (fd >= 0) {
            const int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }
        return fd;
    }
    bool connect_placer() {
        pfd_ = dial();
        Hello hi{kMagic, kVersion, (uint32_t) dim_, (uint32_t) ff_, lim_, kPlacer};
        HelloReply hr{};
        if (pfd_ < 0 || !io_all(pfd_, &hi, sizeof(hi), true) || !io_all(pfd_, &hr, sizeof(hr), false) ||
            hr.magic != kMagic) {
            if (pfd_ >= 0) ::close(pfd_);
            pfd_ = -1;
            return false;
        }
        return true;
    }
    void connect_and_assign(const std::vector<Id>& cand) {
        connecting_.store(true);
        struct Done {
            std::atomic<bool>& f;
            ~Done() { f.store(false); }
        } done{connecting_};
        fd_ = dial();
        if (fd_ < 0) {
            if (!quiet_) std::fprintf(stderr, "ds4 expertd: no daemon at %s, every expert on the CPU lane\n", addr_.c_str());
            quiet_ = true;   // (said once; reconnect() keeps dialling)
            return;
        }
        quiet_ = false;
        Hello hi{kMagic, kVersion, (uint32_t) dim_, (uint32_t) ff_, lim_, kCompute};
        HelloReply hr{};
        uint32_t n = (uint32_t) cand.size(), m = 0;
        if (!send_all(&hi, sizeof(hi)) || !recv_all(&hr, sizeof(hr)) || hr.magic != kMagic || hr.version != kVersion ||
            !send_all(&n, 4) || !send_all(cand.data(), cand.size() * sizeof(Id)) || !recv_all(&m, 4)) {
            std::fprintf(stderr, "ds4 expertd: the daemon at %s did not answer\n", addr_.c_str());
            fail();
            return;
        }
        std::vector<Id> got(m);
        if (m > 0 && !recv_all(got.data(), m * sizeof(Id))) { fail(); return; }
        for (const Id& id : got) {
            const size_t i = (size_t) id.blk * (size_t) n_exp_ + id.e;
            if (id.e < n_exp_ && i < n_slots_) held_[i].store(1, std::memory_order_relaxed);
        }
        n_held_ = m;
        timeval tv{1, 0};   // (a late reply: the lane computes them)
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        std::fprintf(stderr, "ds4 expertd: %s holds %u experts\n", addr_.c_str(), m);
        ready_.store(m > 0, std::memory_order_release);
    }

    std::thread th_;
    std::atomic<bool> ready_{false};
    std::vector<Id> cand_;
    std::chrono::steady_clock::time_point last_dial_{};
    std::atomic<bool> connecting_{false};
    bool quiet_ = false;
    std::unique_ptr<std::atomic<uint8_t>[]> held_;
    std::vector<uint8_t> buf_;
    std::string addr_;
    size_t n_held_ = 0, n_slots_ = 0;
    int fd_ = -1, pfd_ = -1, dim_ = 0, ff_ = 0, n_exp_ = 0;
    float lim_ = 0.0f;
    uint32_t pseq_ = 0;
};

}  // namespace strata::dsv41::expertd
#endif
