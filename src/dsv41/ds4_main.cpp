// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_main.cpp - maya-ds4: DeepSeek-V4.1-Flash on Maya.
//
//   maya-ds4 --pack <model>/maya-ds4 --ids 0,128803,... --n 16          greedy tokens after the ids
//   maya-ds4 --pack ... --ids ... --dump-logits out.f32                every position's logits (the oracle's check)
//   maya-ds4 --pack ... --serve                                       Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `T <id>` lines and `DONE <generated> <prompt> <prompt ms>
// <decode ms> <stop|length|cancel> 0 0 <reused> 0 0`; `STOP` ends a request early, `QUIT` the engine.  A prompt that
// continues the state (the conversation so far) reuses it; one that only shares the last prompt's start goes back to
// that prompt's end; anything else starts over.  `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures
// (the file: see read_pics); the state is shared only where the same picture sits at the same place.  With
// STRATA_DS4_SLOT_KEEP=1 the conversations are kept when another one takes the model (a slot each: the state, the
// compressed KV included), and across runs in STRATA_DS4_SLOT_DIR - so a request that does not continue the state but
// does continue a kept conversation reads only its new tokens.
// `pin=N` on a GEN marks the first N prompt tokens as a start other requests share (a system prompt and its tools): the
// read stops there for a checkpoint, so a request behind the same start - another agent - goes back to it instead of
// reading it again (also without slots).  STRATA_DS4_REUSE=0 turns every reuse off (each request reads its whole prompt).
#if defined(__APPLE__)
#include <execinfo.h>
#include <csignal>
#include <unistd.h>
#include <libproc.h>   // proc_pid_rusage: the bytes this process had the storage read (io_read_bytes)
#endif
#include "ds4_launch.h"   // ds4_dbg
#include "ds4_model.hpp"
#include "ds4_experts_rule.hpp"
#include "ds4_os.hpp"
#include "ds4_packinfo.hpp"
#include "ds4_slotmeta.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the bytes this process had the storage read for it (Linux's /proc/self/io, macOS's rusage; -1 where there is none): the
// Monitor's disk reads a token, the experts read in place from the model's files included
long long io_read_bytes() {
#ifdef __linux__
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
#elif defined(__APPLE__)
    // (the same number `iotop`/Activity Monitor's "Bytes read" shows: every read the storage did for this process, the
    // page cache's misses of the experts read in place included; a page-cache hit counts no storage read, which is what
    // the 60 tps go/no-go needs)
    struct rusage_info_v4 ri;
    std::memset(&ri, 0, sizeof ri);
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*) &ri) != 0) return -1;
    return (long long) ri.ri_diskio_bytesread;
#else
    return -1;
#endif
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1,...] [--max-context N] [--reserve-mib N]\n"
                 "           [--vram-experts F] [--cpu-threads N] [--expert-profile FILE|none] [--profile-readonly] [--ssd-cache GB] [--ssd-policy lru|lfu|lfulru] [--no-expert-cache] [--verbose]\n"
                 "           [--tp N] [--threads on|off] [--prefill-chunk N] [--dspark on|off|none|auto] [--dspark-block N] [--dspark-conf F] [--experts N|auto]\n"
                 "           [--vision on|off] [--mmap-experts on|off|auto]\n"
                 "           (--ids I,J,... [--images FILE] [--n N] [--dump-logits FILE] | --serve\n"
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT\n"
                 "           | --pack-info)\n");
}

// stdin lines on a thread, so a request can see STOP while it decodes
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    void start() {
        std::thread([this] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                std::lock_guard<std::mutex> lk(mu);
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    continue;
                }
                if (l == "QUIT") stop.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
};

// The drafts between two stats: each draft position (1st, 2nd, ..) - how often it was verified with every draft
// before it kept, and how often it was kept then
std::string per_position(const strata::ds4::Stats& s0, const strata::ds4::Stats& s1) {
    std::string pp = "per position (kept / reached):";
    for (int i = 0; i < 8; ++i) {
        const long long r = s1.draft_reached[i] - s0.draft_reached[i], k = s1.draft_kept[i] - s0.draft_kept[i];
        if (r <= 0) break;
        char b[64];
        std::snprintf(b, sizeof b, " %d: %lld/%lld %.0f%%", i + 1, k, r, 100.0 * (double) k / (double) r);
        pp += b;
    }
    return pp;
}

static std::FILE* g_round_log = nullptr;
static std::string g_round_tag;
static int g_spec_depth = 64;   // --dspark-depth: at most this many drafts a round
static std::vector<float> g_round_conf;
static int g_round_nd = 0;

// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule): a draft is verified while the chance
// that it and every draft before it are kept stays at least this - a long window whose later drafts are likely lost is
// not verified whole.  STRATA_DS4_DSPARK_CUM=F, a request's dspark_cum key; 0: off, only the floor on each draft's own
// confidence (dspark_conf).  0.5 is the default on gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) only: measured on 4x R9700, DeepSeek 2.25 bpw,
// six 128-token decodes an arm in one load - 71.7 -> 76.9 tok/s (medians), the same ids.  Project Maya ran 0.5 on a Radeon 8065S (Strix Halo,
// gfx1151: +6.5% decode, the same tokens), but that is a single datapoint on a box the 0.2.7 defaults have not been tested on, so
// Strix keeps 0 until it is; the CUDA measurements in 0.2.7 had it flat, so CUDA keeps 0.  STRATA_DS4_DSPARK_CUM overrides everywhere.
float dspark_cum_default(const strata::ds4::Model& m) {
    static const char* const e = std::getenv("STRATA_DS4_DSPARK_CUM");
    return strata::ds4::dspark_cum_value(e != nullptr, e != nullptr ? (float) std::atof(e) : 0.0f, m.gpus_are_gfx12(),
                                         strata::ds4::defaults_026());
}

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them), verified together; else a plain step.  `out` gets the tokens that follow
// tok (the accepted drafts, then the trunk's next), the last of them the next round's tok.
struct RoundTimes {   // DS4_ROUND_TIMES: where a decode round's time goes (ms, summed), printed at exit
    double draft = 0, verify = 0, step = 0;
    unsigned long long draft_d = 0, verify_d = 0, step_d = 0;
    long rounds = 0, spec_rounds = 0, steps = 0;
    ~RoundTimes() {
        if (ds4_dbg("DS4_ROUND_TIMES") && rounds > 0)
            std::fprintf(stderr, "rounds: %ld (%ld speculative, %ld plain steps): draft %.1f ms (%.2f a round), verify %.1f ms (%.2f), plain step %.1f ms (%.2f)\n",
                         rounds, spec_rounds, steps, draft, spec_rounds ? draft / (double) spec_rounds : 0.0, verify, spec_rounds ? verify / (double) spec_rounds : 0.0,
                         step, steps ? step / (double) steps : 0.0);
        if (ds4_dbg("DS4_ROUND_TIMES") && rounds > 0)
            std::fprintf(stderr, "dispatches a round: draft %.0f, verify %.0f, plain step %.0f\n", spec_rounds ? (double) draft_d / (double) spec_rounds : 0.0, spec_rounds ? (double) verify_d / (double) spec_rounds : 0.0, steps ? (double) step_d / (double) steps : 0.0);
    }
} g_rt;
#if defined(STRATA_METAL)
extern "C" unsigned long long ds4_dispatch_seq();
#endif
#if defined(STRATA_METAL)
extern "C" void ds4_timeline_gate(int);
#endif
static void tl_gate(bool open) {
#if defined(STRATA_METAL)
    static const bool only_draft = std::getenv("DS4_TL_DRAFT") != nullptr;
    if (only_draft) ds4_timeline_gate(open ? 1 : 0);
#endif
}
static unsigned long long rt_seq() {
#if defined(STRATA_METAL)
    return ds4_dispatch_seq();
#else
    return 0;
#endif
}
static double rt_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f) {
    out.clear();
    offered = accepted = 0;
    if (spec && room > 1 && m.draft_block() > 0) {
        std::vector<int> drafts;
        std::vector<float> conf;
        ++g_rt.rounds;
        static const bool tl_init = (tl_gate(false), true);
        (void) tl_init;
        tl_gate(true);
        const double rt0 = rt_now();
        const unsigned long long rs0 = rt_seq();
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        g_rt.draft += rt_now() - rt0;
        tl_gate(false);
        g_rt.draft_d += rt_seq() - rs0;
        g_round_conf = conf;
        g_round_nd = (int) drafts.size();
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default(m);
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < room - 1 && keep < g_spec_depth && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            const double rt1 = rt_now();
            const unsigned long long rs1 = rt_seq();
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            g_rt.verify += rt_now() - rt1;
            g_rt.verify_d += rt_seq() - rs1;
            ++g_rt.spec_rounds;
            if (g_round_log != nullptr) {   // DS4_ROUND_LOG: tag, drafts offered, accepted, the confidences
                std::fprintf(g_round_log, "%s\t%d\t%d", g_round_tag.c_str(), keep, acc);
                for (int i = 0; i < keep; ++i) std::fprintf(g_round_log, "\t%.4f", g_round_conf[(size_t) i]);
                std::fprintf(g_round_log, "\n");
            }
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            return true;
        }
    }
    const double rt2 = rt_now();
    const unsigned long long rs2 = rt_seq();
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
    g_rt.step += rt_now() - rt2;
    g_rt.step_d += rt_seq() - rs2;
    ++g_rt.steps;
    return true;
}

// The pack's line-break tokens (vocab.json keys holding a newline).
static std::vector<char> g_newline;
// Turn markers (keys with a full-width bar, think and tool tags).
static std::vector<char> g_mark;
static bool is_mark(int t) { return t >= 0 && (size_t) t < g_mark.size() && g_mark[(size_t) t]; }
static bool is_nl_tok(int t) { return t >= 0 && (size_t) t < g_newline.size() && g_newline[(size_t) t]; }

// How a prompt lines up with a kept conversation `ref`: for each
// prompt token the ref position it matches (-1: new), the ref spans
// the prompt lacks (stale), and how far into ref the matching got.
struct Align {
    std::vector<int64_t> ref_of;
    std::vector<std::pair<size_t, size_t>> stale;   // (start, length) in ref
    size_t ref_end = 0, matched = 0;
};

static bool align_to(const std::vector<int32_t>& ref, const std::vector<int>& ids, Align& al) {
    constexpr size_t K = 32, kMaxSkip = 1 << 16;
    const size_t n = ids.size();
    size_t c0 = 0;
    while (c0 < ref.size() && c0 < n && ref[c0] == ids[c0]) ++c0;
    if (c0 == ref.size() || n < 2 * K || ref.size() < c0 + K) return false;
    auto gram = [](const auto* p) {
        uint64_t h = 1469598103934665603ull;
        for (size_t t = 0; t < K; ++t) h = (h ^ (uint32_t) p[t]) * 1099511628211ull;
        return h;
    };
    // Each run of ref is found wherever it sits in the prompt (a
    // prompt the engine reordered last turn comes back in the
    // client's order); a prompt token is used once, never the last.
    std::unordered_map<uint64_t, std::vector<uint32_t>> at;
    for (size_t q = 0; q + K <= n - 1; ++q) at[gram(ids.data() + q)].push_back((uint32_t) q);
    std::vector<char> used(n, 0);
    const auto free_run = [&](size_t q) {
        for (size_t t = q; t < q + K; ++t)
            if (used[t]) return false;
        return true;
    };
    const auto find_at = [&](size_t i) -> size_t {   // ref[i..i+K) at an unused place
        const auto hit = at.find(gram(ref.data() + i));
        if (hit == at.end()) return SIZE_MAX;
        for (uint32_t q : hit->second)
            if (free_run(q) && std::equal(ref.begin() + (long) i, ref.begin() + (long) (i + K), ids.begin() + q))
                return q;
        return SIZE_MAX;
    };
    // (a whole line shorter than a run is found by its tokens)
    const auto ends = [](const auto& v, size_t b) {
        return is_nl_tok(v[b]) || is_mark(v[b]) || (b + 1 < v.size() && is_mark(v[b + 1]));
    };
    const auto span_hash = [](const auto* p, size_t len) {
        uint64_t h = 1469598103934665603ull;
        for (size_t t = 0; t < len; ++t) h = (h ^ (uint32_t) p[t]) * 1099511628211ull;
        return h ^ len;
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> line_at;
    for (size_t a = 0; a + 1 < n;) {
        size_t b = a;
        while (b + 1 < n && ends(ids, b) == false) ++b;
        if (b + 1 < n && b - a + 1 >= 4) line_at[span_hash(ids.data() + a, b - a + 1)].push_back((uint32_t) a);
        a = b + 1;
    }
    const auto find_line = [&](size_t i) -> size_t {   // ref's line at i at an unused place
        if (i != 0 && ends(ref, i - 1) == false) return SIZE_MAX;
        size_t j = i;
        while (j + 1 < ref.size() && ends(ref, j) == false) ++j;
        const size_t len = j - i + 1;
        if (len < 4) return SIZE_MAX;
        const auto hit = line_at.find(span_hash(ref.data() + i, len));
        if (hit == line_at.end()) return SIZE_MAX;
        for (uint32_t q0 : hit->second) {
            bool ok = q0 + len < n;
            for (size_t t = 0; ok && t < len; ++t) ok = used[q0 + t] == 0 && ids[q0 + t] == ref[i + t];
            if (ok) return q0;
        }
        return SIZE_MAX;
    };
    const auto find_any = [&](size_t i) -> size_t {
        const size_t g = i + K <= ref.size() ? find_at(i) : SIZE_MAX;
        return g != SIZE_MAX ? g : find_line(i);
    };
    al = Align{};
    al.ref_of.assign(n, -1);
    size_t i = 0, q = 0;   // (q: where the current run continues)
    while (i < ref.size()) {
        if (q + 1 < n && used[q] == 0 && ref[i] == ids[q]) {   // (the run goes on)
            used[q] = 1;
            al.ref_of[q++] = (int64_t) i++;
            ++al.matched;
            continue;
        }
        size_t i2 = i, q2 = find_any(i);
        if (q2 == SIZE_MAX)   // (ref has tokens the prompt lacks)
            for (i2 = i + 1; i2 < ref.size() && i2 <= i + kMaxSkip; ++i2)
                if ((q2 = find_any(i2)) != SIZE_MAX) break;
        if (q2 == SIZE_MAX) break;
        if (i2 > i) al.stale.push_back({i, i2 - i});
        i = i2;
        q = q2;
    }
    al.ref_end = i;
    return al.matched >= c0 + 256;
}


static void load_newlines(const std::string& pack) {
    std::ifstream f(pack + "/tokenizer/vocab.json", std::ios::binary);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '"') continue;
        size_t j = i + 1;
        bool nl = false;
        for (; j < s.size() && s[j] != '"'; ++j) {
            if (s[j] == '\\') {
                nl = nl || s.compare(j, 6, "\\u010a") == 0 || s.compare(j, 6, "\\u010A") == 0;
                ++j;
            } else if ((unsigned char) s[j] == 0xC4 && j + 1 < s.size() && (unsigned char) s[j + 1] == 0x8A)
                nl = true;
        }
        size_t k = j + 1;
        while (k < s.size() && (s[k] == ' ' || s[k] == ':')) ++k;
        if (k >= s.size() || s[k] < '0' || s[k] > '9') {
            i = j;
            continue;
        }
        const long id = std::strtol(s.c_str() + k, nullptr, 10);
        if (id >= 0 && id < (1 << 24)) {
            if ((size_t) id >= g_newline.size()) g_newline.resize((size_t) id + 1, 0);
            if (nl) g_newline[(size_t) id] = 1;
            const std::string key = s.substr(i + 1, j - i - 1);
            const bool bar = key.find("\xEF\xBD\x9C") != std::string::npos ||
                             key.find("\\uff5c") != std::string::npos || key.find("\\uFF5C") != std::string::npos;
            const bool tag = key == "<think>" || key == "</think>" || key == "<tool_result>" || key == "</tool_result>";
            if (bar || tag) {
                if ((size_t) id >= g_mark.size()) g_mark.resize((size_t) id + 1, 0);
                g_mark[(size_t) id] = 1;
            }
        }
        i = k;
    }
}

// ---- KV splice (DS4_SPLICE_EDIT=1): a prompt changed in the middle of
// a kept state is spliced in place (Model::splice_edits), not re-read.
struct EditPlan {
    std::vector<strata::ds4::Model::Edit> eds;
    int64_t E = -1, changed = 0;   // (the state's tokens kept to E)
    size_t p_end = 0;              // (the prompt read on from here)
};

// The edits that make ref[0, E) into ids[0, p_end), from the alignment's
// forward matches; E the latest of `cks` where every edit starts and
// ends on the grid, fits its positions (room) and keeps W tokens after
static bool plan_edits(const std::vector<int32_t>& ref, const std::vector<int>& ids, const Align& al, std::vector<int64_t> cks,
                       int64_t SA, int64_t W, const std::function<int64_t(int64_t, int64_t)>& room, EditPlan& out) {
    std::vector<int64_t> p_of(ref.size(), -1);
    int64_t last = -1;
    for (size_t t = 0; t < ids.size() && t < al.ref_of.size(); ++t) {
        const int64_t r = al.ref_of[t];
        if (r > last && r < (int64_t) ref.size()) {
            p_of[(size_t) r] = (int64_t) t;
            last = r;
        }
    }
    std::sort(cks.rbegin(), cks.rend());
    for (int64_t E : cks) {
        if (E < 256 || E > (int64_t) ref.size() || p_of[(size_t) E - 1] < 0) continue;
        // the raw gaps between matched runs: ref [ra, rb) -> ids [pa, pb)
        struct G { int64_t ra, rb, pa, pb; };
        std::vector<G> gs;
        int64_t pr = -1, pt = -1;
        bool ok = true;
        for (int64_t r = 0; r < E && ok; ++r) {
            const int64_t t = p_of[(size_t) r];
            if (t < 0) continue;
            if (r != pr + 1 || t != pt + 1) {
                if (pr < 0) ok = false;   // (the starts must agree)
                gs.push_back({pr + 1, r, pt + 1, t});
            }
            pr = r;
            pt = t;
        }
        if (!ok || pr != E - 1) continue;
        // on the grid (widened over matched tokens), merged where they meet
        std::vector<G> m;
        for (G g : gs) {
            const int64_t lo = g.ra % SA, hi = (SA - g.rb % SA) % SA;
            g.ra -= lo;
            g.pa -= lo;
            g.rb += hi;
            g.pb += hi;
            if (!m.empty() && g.ra <= m.back().rb) {
                m.back().rb = std::max(m.back().rb, g.rb);
                m.back().pb = std::max(m.back().pb, g.pb);
            } else {
                m.push_back(g);
            }
        }
        // (any change of length: an odd one leaves the followers' paired
        // rows a token off, Model::splice_edits pairs the end anew)
        const std::vector<G>& f = m;
        EditPlan ep;
        for (const G& g : f) {
            if (!ok) break;
            if (g.ra < 0 || g.pa < 0 || g.rb + W > E || room(g.ra, g.rb - g.ra) < g.pb - g.pa) {
                ok = false;
                break;
            }
            strata::ds4::Model::Edit ed;
            ed.a = g.ra;
            ed.d = g.rb - g.ra;
            ed.rep.assign(ids.begin() + g.pa, ids.begin() + g.pb);
            ep.changed += g.pb - g.pa;
            ep.eds.push_back(std::move(ed));
        }
        if (!ok) continue;
        ep.E = E;
        ep.p_end = (size_t) p_of[(size_t) E - 1] + 1;
        out = std::move(ep);
        return true;
    }
    return false;
}

// ---- the block library (DS4_SPLICE_BLOCKS=1: a dry run for now):
// spans of what a prompt has left to read that a kept conversation
// (or the state's kept part) already holds, at any position
struct BlockHit {
    int src = 0;       // (index into the sources)
    int64_t s0 = 0;    // (where it sits there)
    size_t q = 0, n = 0;   // (where it goes in the prompt, its length)
};
static std::vector<BlockHit> find_blocks(const std::vector<int>& ids, size_t from,
                                         const std::vector<const std::vector<int32_t>*>& srcs, int64_t align,
                                         size_t min_run, int64_t& wrong_parity) {
    constexpr size_t K = 64;
    const auto h64 = [](const auto* p) {
        uint64_t h = 1469598103934665603ull;
        for (size_t t = 0; t < K; ++t) h = (h ^ (uint32_t) p[t]) * 1099511628211ull;
        return h;
    };
    std::unordered_map<uint64_t, std::vector<std::pair<int, int64_t>>> idx;
    for (size_t s = 0; s < srcs.size(); ++s)
        for (size_t o = 0; o + K <= srcs[s]->size(); o += K) idx[h64(srcs[s]->data() + o)].push_back({(int) s, (int64_t) o});
    std::vector<BlockHit> out;
    wrong_parity = 0;
    const size_t n = ids.size();
    size_t q = from, lo = from;   // (lo: the end of the last block)
    while (q + K <= n) {
        const auto it = idx.find(h64(ids.data() + q));
        BlockHit best;
        if (it != idx.end())
            for (size_t c = 0; c < it->second.size() && c < 16; ++c) {
                const auto& v = *srcs[(size_t) it->second[c].first];
                const int64_t o = it->second[c].second;
                if (!std::equal(v.begin() + o, v.begin() + o + (long) K, ids.begin() + (long) q)) continue;
                size_t b = 0, f = K;
                while (q - b > lo && o - (int64_t) b > 0 && v[(size_t) (o - (int64_t) b - 1)] == ids[q - b - 1]) ++b;
                while (q + f < n && (size_t) o + f < v.size() && v[(size_t) o + f] == ids[q + f]) ++f;
                if (b + f > best.n) best = {it->second[c].first, o - (int64_t) b, q - b, b + f};
            }
        if (best.n < min_run) {
            ++q;
            continue;
        }
        // (the compressed rows' groups must line up: same parity)
        if (((int64_t) best.q - best.s0) % align != 0) {
            wrong_parity += (int64_t) best.n;
        } else {
            const int64_t cut = (align - (int64_t) best.q % align) % align;
            best.q += (size_t) cut;
            best.s0 += cut;
            best.n = (best.n - (size_t) cut) / (size_t) align * (size_t) align;
            out.push_back(best);
        }
        lo = q = best.q + best.n;
    }
    return out;
}

// Changed lines move whole: a line holding any new or stale token is
// new (prompt) or stale (ref) as a whole, cuts are widened to the
// state's group size `align`, and the stale spans are rebuilt.
static void whole_lines(const std::vector<int32_t>& ref, const std::vector<int>& ids, Align& al, int64_t align) {
    const auto is_nl = [](int t) { return t >= 0 && (size_t) t < g_newline.size() && g_newline[(size_t) t]; };
    if (g_newline.empty()) return;
    // (a line ends at a newline; a marker is a line of its own)
    const auto ends_at = [&](const auto& v, size_t b) {
        return is_nl(v[b]) || is_mark(v[b]) || (b + 1 < v.size() && is_mark(v[b + 1]));
    };
    const size_t R = al.ref_end, n = ids.size();
    std::vector<char> st(R, 0);
    for (const auto& s : al.stale)
        for (size_t k = s.first; k < s.first + s.second && k < R; ++k) st[k] = 1;
    std::vector<int64_t> p_of(R, -1);
    size_t tmax = 0;   // (the last matched prompt token: changes before it only)
    for (size_t t = 0; t < n; ++t)
        if (al.ref_of[t] >= 0 && (size_t) al.ref_of[t] < R) {
            p_of[(size_t) al.ref_of[t]] = (int64_t) t;
            tmax = t;
        }
    const auto unmatch_ref = [&](size_t k) {
        st[k] = 1;
        if (p_of[k] >= 0) al.ref_of[(size_t) p_of[k]] = -1;
        p_of[k] = -1;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t a = 0; a <= tmax;) {   // (prompt lines with a new token)
            size_t b = a;
            while (b < n - 1 && ends_at(ids, b) == false) ++b;
            if (b <= tmax) {
                bool any_new = false, any_old = false;
                for (size_t t = a; t <= b; ++t) (al.ref_of[t] < 0 ? any_new : any_old) = true;
                if (any_new && any_old) {
                    for (size_t t = a; t <= b; ++t)
                        if (al.ref_of[t] >= 0 && (size_t) al.ref_of[t] < R) unmatch_ref((size_t) al.ref_of[t]);
                    changed = true;
                }
            }
            a = b + 1;
        }
        for (size_t a = 0; a < R;) {   // (ref lines with a stale token)
            size_t b = a;
            while (b < ref.size() - 1 && ends_at(ref, b) == false) ++b;
            if (b < R) {
                bool any_st = false, any_ok = false;
                for (size_t k = a; k <= b; ++k) (st[k] ? any_st : any_ok) = true;
                if (any_st && any_ok) {
                    for (size_t k = a; k <= b; ++k) unmatch_ref(k);
                    changed = true;
                }
            }
            a = b + 1;
        }
        for (size_t k = 0; k < R; ++k) {   // (cuts widened to the group size)
            if (st[k] == 0) continue;
            const bool starts = k == 0 || st[k - 1] == 0;
            const bool ends = k + 1 == R || st[k + 1] == 0;
            if (starts && k % (size_t) align != 0 && st[k - 1] == 0) {
                unmatch_ref(k - 1);
                changed = true;
            }
            if (ends && (k + 1) % (size_t) align != 0 && k + 1 < R && st[k + 1] == 0) {
                unmatch_ref(k + 1);
                changed = true;
            }
        }
    }
    al.stale.clear();
    for (size_t k = 0; k < R;) {
        if (st[k] == 0) {
            ++k;
            continue;
        }
        size_t e = k;
        while (e < R && st[e]) ++e;
        al.stale.push_back({k, e - k});
        k = e;
    }
    al.matched = 0;
    for (int64_t r : al.ref_of) al.matched += r >= 0;
}

// The prompt with its tokens matched to ref[0, upto) first (in ref
// order), then the rest in their own order.
static std::vector<int> alike_first(const std::vector<int>& ids, const Align& al, int64_t upto, size_t& n_alike) {
    std::vector<std::pair<int64_t, int>> early, late;   // (ref position, token)
    std::vector<int> fresh;
    for (size_t t = 0; t < ids.size(); ++t) {
        const int64_t r = al.ref_of[t];
        if (r < 0) fresh.push_back(ids[t]);
        else (r < upto ? early : late).push_back({r, ids[t]});
    }
    std::sort(early.begin(), early.end());
    std::sort(late.begin(), late.end());
    std::vector<int> out;
    out.reserve(ids.size());
    for (const auto& x : early) out.push_back(x.second);
    n_alike = out.size();
    for (const auto& x : late) out.push_back(x.second);
    out.insert(out.end(), fresh.begin(), fresh.end());
    return out;
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    long long pin = 0;      // pin=N: the first N prompt tokens are a start other requests share (a system prompt and its
                            // tools): the state at N is kept as a checkpoint, so another agent behind the same start
                            // goes back to it instead of reading it again
    float cum = -1.0f;      // dspark_cum=F: the cut on the product of the confidences (-1: the engine's, dspark_cum_default)
    std::string file;   // GENI: the pictures
    std::vector<int> ids;
};

// A prompt's picture: where its span starts, its pixels (the server's resize: RGB rows, both sides multiples of the
// patch size), a hash of them; once encoded, its span's rows
struct Pic {
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;   // the span (0: not known yet)
    std::vector<uint8_t> rgb;
};

// GENI's file: int32 'DSI1', int32 count, then per picture int64 start, int32 width, int32 height and its
// height x width x 3 RGB bytes, in prompt order
bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the pictures' file " + path;
        return false;
    }
    char magic[4] = {};
    int32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DSI1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 &&
              n >= 0 && n <= 4096;
    for (int i = 0; ok && i < n; ++i) {
        Pic p;
        int32_t wh[2] = {};
        ok = std::fread(&p.start, 8, 1, f) == 1 && std::fread(wh, 4, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 &&
             (int64_t) wh[0] * wh[1] <= (64 << 20);
        if (!ok) break;
        p.width = wh[0], p.height = wh[1];
        p.rgb.resize((size_t) p.width * p.height * 3);
        ok = std::fread(p.rgb.data(), 1, p.rgb.size(), f) == p.rgb.size();
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the size and the pixels
        for (int32_t x : wh)
            for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) (x >> (8 * b))) * 1099511628211ull;
        for (uint8_t b : p.rgb) h = (h ^ b) * 1099511628211ull;
        p.hash = h;
        v.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) err = path + ": not a pictures' file (DSI1)";
    return ok;
}

// the encoded spans, the latest few (a conversation sends its pictures with every turn)
struct SpanCache {
    std::deque<std::pair<uint64_t, std::shared_ptr<const std::vector<uint16_t>>>> q;
    std::shared_ptr<const std::vector<uint16_t>> get(strata::ds4::Model& m, const Pic& p, std::string& err) {
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i].first == p.hash) {
                auto r = q[i].second;
                q.erase(q.begin() + (long) i);
                q.emplace_front(p.hash, r);
                return r;
            }
        auto rows = std::make_shared<std::vector<uint16_t>>();
        if (!m.encode_image(p.rgb.data(), p.width, p.height, *rows, err)) return nullptr;
        q.emplace_front(p.hash, rows);
        if (q.size() > 8) q.pop_back();
        return rows;
    }
};

bool parse_gen(const std::string& line, Req& r, bool geni) {
    const char* p = line.c_str() + (geni ? 5 : 4);
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
        if (eq == std::string::npos) {
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "pin") r.pin = std::atoll(v.c_str());
        else if (k == "dspark_cum" || k == "spec_cum") r.cum = std::strtof(v.c_str(), nullptr);
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min, const std::string& pack) {
    std::vector<int> eos = parse_ids(eos_list);
    load_newlines(pack);
    {
        size_t nl = 0;
        for (char c : g_newline) nl += c;
        std::fprintf(stderr, "ds4 serve: %zu line-break tokens (changed lines move whole)\n", nl);
    }
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    std::string line;
    unsigned long long counter = 0;
    std::vector<Pic> held;   // the pictures the state holds (start, hash, span)
    SpanCache spans;
    const int img_tok = m.image_token();
    // CONVERSATION SLOTS: a request that does not continue the conversation in the model first sets that one aside
    // (Model::slot_save: its window rings, the compressors' partial groups and every source's compressed KV and index
    // keys), then takes back the kept conversation this prompt does continue (slot_load) - chats, and an agent's side
    // requests, that interleave no longer re-read each other's prompts (70k and 185k tokens alternating: 75-200 s a
    // switch at ~930 tok/s).  STRATA_DS4_SLOT_KEEP=1 turns them on and keeps them across runs, in STRATA_DS4_SLOT_DIR
    // (default <pack>/slots): a slot is a .bin (the state) and a .meta (this model's identity, the conversation's
    // tokens and its size), both written under a temporary name first; a slot whose .meta is missing, foreign or does
    // not match its file, or unused for STRATA_DS4_SLOT_DAYS (7) days, is removed at start.  STRATA_DS4_SLOT_CAP (4) is
    // the conversations kept in RAM; past it the least recently used is written to the folder and read back from there
    // (RAM first, the folder past the cap), and the engine's end writes the ones in RAM to the folder, so the next run
    // reads on from them.  STRATA_DS4_SLOT_GB (16) caps the folder's .bin files: at start, and after every write, the
    // folder's least recently used conversation is removed while they are over it (never the one just written, so a
    // conversation bigger than the cap stays).  STRATA_DS4_SLOT_MIN (1024) is the shortest conversation set aside (a
    // shorter one is read again in about the time its slot would take).  Text only: a conversation that holds pictures
    // is not set aside, and a prompt with pictures takes none back.
    const auto env_num = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    const bool slot_keep =
        std::getenv("STRATA_DS4_SLOT_KEEP") != nullptr && std::atoi(std::getenv("STRATA_DS4_SLOT_KEEP")) != 0;
    const bool reuse = std::getenv("STRATA_DS4_REUSE") == nullptr || std::atoi(std::getenv("STRATA_DS4_REUSE")) != 0;
    int slot_max = slot_keep ? std::max(0, (int) env_num("STRATA_DS4_SLOT_CAP", 4)) : 0;
    const size_t slot_min = (size_t) std::max(1.0, env_num("STRATA_DS4_SLOT_MIN", 1024));
    const double slot_days = env_num("STRATA_DS4_SLOT_DAYS", 7);
    const uint64_t slot_budget = (uint64_t) (env_num("STRATA_DS4_SLOT_GB", 16) * 1073741824.0);
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                            // empty: the state is in data (RAM)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        std::vector<int64_t> cks;                     // the positions short of its end it can be taken back at
    };
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    const std::string slot_identity = m.slot_identity();
    const auto note = [&](const std::string& msg) {
        std::fprintf(stderr, "ds4 slots: %s\n", msg.c_str());
        std::printf("NOTE sessions: %s\n", msg.c_str());
        std::fflush(stdout);
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes,
                                const std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::write(path, slot_identity, toks, bytes, cks);
    };
    // a .meta of this model -> its tokens and the slot's size (false: foreign, damaged or not this model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes,
                               std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::read(path, slot_identity, m.max_context(), toks, bytes, cks);
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)   // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
    };
    const auto slot_count = [&](int& ram, int& disk) {
        ram = disk = 0;
        for (const Slot& s : slots) (s.data ? ram : disk) += 1;
    };
    // the folder's conversations (their .bin; a .meta beside each is a few KB: identity and tokens)
    const auto slot_folder_bytes = [&]() {
        uint64_t total = 0;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0) {
                const uint64_t bytes = (uint64_t) std::filesystem::file_size(de.path(), ec);
                if (!ec) total += bytes;
            }
        }
        return total;
    };
    // the folder over STRATA_DS4_SLOT_GB: its least recently used conversation goes first, and never `keep` (the one just
    // written), so a conversation bigger than the cap stays
    const auto slot_trim = [&](const std::string& keep) {
        const uint64_t before = slot_folder_bytes();
        size_t dropped = 0;
        while (slot_folder_bytes() > slot_budget) {
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (!slots[i].data && slots[i].path != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used))
                    lru = i;
            if (lru == SIZE_MAX) break;
            slot_drop(lru);
            ++dropped;
        }
        if (dropped > 0) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "the folder held %s (over STRATA_DS4_SLOT_GB, %s): %zu least recently used "
                          "conversation(s) removed, %s left", gb(before).c_str(), gb(slot_budget).c_str(), dropped,
                          gb(slot_folder_bytes()).c_str());
            note(msg);
        }
    };
    // a RAM slot into the folder (its bytes, then its .meta); false: not written (its bytes go, then)
    const auto slot_write_file = [&](Slot& s) -> bool {
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < ((uint64_t) 8 << 30)) {   // (8 GB free, the GLM engine's guard)
            std::fprintf(stderr, "ds4 slots: under 8 GB free in %s - the conversation is not written\n",
                         slot_dir.string().c_str());
            return false;
        }
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path =
            (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) + ".bin")).string();
        {
            std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) s.data->data(), (std::streamsize) s.data->size());
            f.flush();
            if (!f) {
                std::filesystem::remove(path + ".tmp", ec);
                std::fprintf(stderr, "ds4 slots: writing %s.tmp failed (disk full?)\n", path.c_str());
                return false;
            }
        }
        fs::rename(path + ".tmp", path, ec);
        if (ec || !meta_write(meta_of(path), s.tokens, s.data->size(), s.cks)) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "ds4 slots: writing %s failed\n", path.c_str());
            return false;
        }
        s.path = path;
        s.bytes = s.data->size();
        s.data.reset();
        slot_trim(path);   // (the folder over its cap: the least recently used conversation goes, not this one)
        return true;
    };
    // the conversation in the model into a slot (the state is lost otherwise: another one takes the model)
    const auto slot_put = [&](const char* why) {
        if (slot_max <= 0) return;
        const std::vector<int32_t>& t = m.tokens();
        if (t.size() < slot_min) return;   // (a fresh engine's 0-token state is no conversation: nothing to set aside)
        const auto t0 = std::chrono::steady_clock::now();
        auto buf = std::make_shared<std::vector<uint8_t>>();
        std::string e;
        const uint64_t bytes = m.slot_save(*buf, e);   // (a state unchanged since it was taken refuses here: nothing
        if (bytes == 0) {                             // changes, its slot stays)
            if (e != "it is already set aside")
                std::fprintf(stderr, "ds4 slots: setting the conversation aside failed (%s) - it is read from token 0 "
                                     "when it comes back\n", e.c_str());
            return;
        }
        for (size_t i = slots.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (slots[i].tokens.size() <= t.size() && std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), t.begin()))
                slot_drop(i);
        slots.push_back({t, "", bytes, ++slot_clock, nullptr, m.live_positions()});
        slots.back().data = buf;
        while (true) {   // past the cap: the least recently used conversation in RAM to the folder, or gone
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            if (ram <= slot_max) break;
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (slots[i].data && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
            if (lru == SIZE_MAX) break;
            if (!slot_write_file(slots[lru])) slot_drop(lru);
        }
        int ram = 0, disk = 0;
        slot_count(ram, disk);
        char msg[256];
        std::snprintf(msg, sizeof msg, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %d in RAM, %d in %s",
                      t.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ram, disk,
                      slot_dir.string().c_str());
        note(msg);
    };
    // the kept conversation this prompt shares the most of - past `floor` tokens, what the model itself can go back
    // to - back in the model: the tokens it covers (0: none).  A conversation the prompt continues whole comes back
    // whole (`whole`); one it leaves part way - a retried prompt, a history the client renders again without the
    // earlier reasoning - comes back at its latest checkpoint the prompt still agrees with (the end of a prompt, the
    // position before it, the anchor where its run of continued prompts began).
    const auto slot_take = [&](const std::vector<int>& prompt, size_t floor, bool& whole) -> size_t {
        whole = false;
        if (slot_max <= 0 || prompt.size() < 2) return 0;
        std::vector<strata::ds4::slotmeta::Cand> cands;
        for (const Slot& sl : slots) cands.push_back({&sl.tokens, &sl.cks});
        size_t len = 0;
        const size_t best = strata::ds4::slotmeta::pick(cands, prompt, floor, &len);
        if (best == SIZE_MAX) return 0;
        const size_t whole_len = slots[best].tokens.size();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ram = slots[best].data != nullptr;
        std::string e;
        bool back = false;
        if (ram) back = m.slot_load(slots[best].data->data(), slots[best].data->size(), (int64_t) whole_len, e);
        else {
            std::ifstream f(slots[best].path, std::ios::binary);
            std::vector<uint8_t> file;
            if (f) {
                f.seekg(0, std::ios::end);
                const std::streamoff n = f.tellg();
                f.seekg(0, std::ios::beg);
                if (n > 0) {
                    file.resize((size_t) n);
                    if (f.read((char*) file.data(), n))
                        back = m.slot_load(file.data(), file.size(), (int64_t) whole_len, e);
                    else e = "reading " + slots[best].path + " failed";
                } else e = slots[best].path + " is empty";
            } else e = "cannot read " + slots[best].path;
        }
        if (!back) {
            note("taking a conversation back failed (" + (e.empty() ? std::string("it did not load") : e) +
                 ") - reading its prompt again");
            slot_drop(best);
            return 0;
        }
        if (len < whole_len && !(m.restore((int64_t) len) && (size_t) m.pos() == len)) {
            note("going back to token " + std::to_string(len) + " of a kept conversation failed - reading its prompt again");
            m.reset();
            return 0;
        }
        whole = len == whole_len;
        slots[best].used = ++slot_clock;
        if (!ram) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char msg[256];
        std::string part;
        if (!whole) part = " at token " + std::to_string(len) + " of " + std::to_string(whole_len);
        std::snprintf(msg, sizeof msg, "took back a %zu-token conversation%s from %s (%.1f s); %zu new tokens to read",
                      len, part.c_str(), ram ? "RAM" : "the folder",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(msg);
        return len;
    };
    if (slot_max > 0) {
        slot_dir = std::getenv("STRATA_DS4_SLOT_DIR") ? std::filesystem::path(std::getenv("STRATA_DS4_SLOT_DIR"))
                                                      : std::filesystem::path(pack) / "slots";
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "ds4 slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        } else {
            struct Found {
                std::string bin;
                std::vector<int32_t> toks;
                uint64_t bytes;
                std::vector<int64_t> cks;
                fs::file_time_type t;
            };
            std::vector<Found> found;
            size_t refused = 0;
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
                const std::string fn = de.path().filename().string(), p = de.path().string();
                if (fn.rfind("slot-", 0) != 0) continue;
                const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
                const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
                if (!is_bin && !is_meta) {   // a .tmp an interrupted write left
                    std::filesystem::remove(de.path(), ec);
                    continue;
                }
                if (!is_meta) continue;      // a .bin is judged by its .meta
                const std::string bin = p.substr(0, p.size() - 5) + ".bin";
                Found fd{bin, {}, 0, {}, fs::last_write_time(de.path(), ec)};
                const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
                if (meta_read(p, fd.toks, fd.bytes, fd.cks) && have == fd.bytes && fd.bytes > 0 && slot_age_days(p) <= slot_days)
                    found.push_back(std::move(fd));
                else {
                    ++refused;
                    std::filesystem::remove(p, ec);
                    std::filesystem::remove(bin, ec);
                }
            }
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a .bin without its .meta
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
            std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
            uint64_t total = 0;
            for (auto& fd : found) {
                total += fd.bytes;
                slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock, nullptr, std::move(fd.cks)});
            }
            std::string msg = "folder " + slot_dir.string() + ": " + std::to_string(slots.size()) +
                              " conversation(s) kept from earlier runs (" + gb(total) + ")";
            if (refused) msg += ", " + std::to_string(refused) + " not this model's or incomplete - removed";
            note(msg);
            slot_trim("");   // (the folder over STRATA_DS4_SLOT_GB: the least recently used conversation goes first)
        }
    }
    if (slot_max > 0) {
#if !defined(_WIN32)
        // (the server ends the engine with QUIT and waits for it; a SIGTERM to the whole process group - systemd
        // stopping a service, a supervisor killing the server's group - must not end it mid-write, and a server that
        // is already gone must not end it with SIGPIPE when it writes its last lines)
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
#endif
    }
    // WP-8's option (1): the post-prompt cache settle runs in the gaps between requests and gives way the moment one arrives
    // (DS4_SETTLE_IDLE=0 turns it off).  An agent's turn ends, its tool runs, and the swaps the next answer wants go in during
    // that idle - which is what the 800 ms prompt-side budget cannot do for a run that has no next request.
    std::atomic<bool> settle_cancel{false};
    std::thread settle_th;
    const bool settle_idle_on = [] {
        const char* v = std::getenv("DS4_SETTLE_IDLE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto settle_stop = [&] {
        if (!settle_th.joinable()) return;
        settle_cancel.store(true, std::memory_order_release);
        settle_th.join();
    };
    const auto settle_start = [&] {
        if (!settle_idle_on || settle_th.joinable()) return;
        settle_cancel.store(false, std::memory_order_release);
        settle_th = std::thread([&m, &settle_cancel] { m.settle_idle(settle_cancel); });
    };
    for (;;) {
        settle_start();   // (nothing in flight: this is the idle)
        const bool have_line = in.next(line);
        settle_stop();    // (a line arrived, or the input closed)
        if (!have_line) break;
        if (line == "QUIT") break;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        const double t0 = now_ms();
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
                std::fflush(stdout);
                continue;
            }
        }
        if (!reuse) {   // (STRATA_DS4_REUSE=0, for A/Bs and tests: every request is a fresh read of its whole prompt)
            m.reset();
            held.clear();
        }
        if (const char* dump = std::getenv("DS4_DUMP_IDS")) {
            // (debug: the client's prompt and the model's state)
            std::ofstream f(std::string(dump) + ".client", std::ios::binary | std::ios::trunc);
            f.write((const char*) r.ids.data(), (std::streamsize) (r.ids.size() * sizeof(int)));
            std::ofstream g(std::string(dump) + ".state", std::ios::binary | std::ios::trunc);
            g.write((const char*) m.tokens().data(), (std::streamsize) (m.tokens().size() * sizeof(int32_t)));
        }
        static const bool reorder_on = [] {
            const char* v = std::getenv("STRATA_DS4_REORDER");
            return v == nullptr || std::atoi(v) != 0;
        }();
        // DS4_SPLICE_EDIT=1: the kept state a prompt changed in the middle of
        // is edited in place (the changed spans and a seam DS4_SPLICE_SEAM
        // after each computed, the rest kept); else the reorder path below
        static const bool splice_edit_on = std::getenv("DS4_SPLICE_EDIT") && std::atoi(std::getenv("DS4_SPLICE_EDIT")) != 0;
        static const int64_t splice_seam = std::getenv("DS4_SPLICE_SEAM") ? std::atoll(std::getenv("DS4_SPLICE_SEAM")) : 2048;
        bool edited = false;
        if (splice_edit_on && reuse && pics.empty() && !r.ids.empty()) {
            const int64_t SA = m.splice_align(), SW = m.splice_window();
            long best = -2;
            EditPlan bp;
            int64_t bgain = 0;
            const auto try_one = [&](long k, const std::vector<int32_t>& t, const std::vector<int64_t>& cks, bool live) {
                if (t.empty() || t[0] != r.ids[0]) return;
                Align a2;
                if (!align_to(t, r.ids, a2)) return;
                whole_lines(t, r.ids, a2, SA);
                EditPlan ep;
                // (a longer span moves its followers on: any length fits)
                const auto room = [](int64_t, int64_t) { return (int64_t) 1 << 40; };
                (void) live;
                if (!plan_edits(t, r.ids, a2, cks, SA, SW, room, ep) || ep.eds.empty()) return;
                const int64_t gain = (int64_t) ep.p_end - ep.changed - splice_seam * (int64_t) ep.eds.size();
                if (gain > bgain) {
                    bgain = gain;
                    best = k;
                    bp = std::move(ep);
                }
            };
            try_one(-1, m.tokens(), m.live_positions(), true);
            for (size_t k = 0; k < slots.size(); ++k) try_one((long) k, slots[k].tokens, slots[k].cks, false);
            size_t c0 = 0;   // (worth it only past the plain shared prefix)
            for (const auto& st0 = m.tokens(); c0 < st0.size() && c0 < r.ids.size() && st0[c0] == r.ids[c0];) ++c0;
            if (best >= -1 && bgain > (int64_t) c0) {
                std::string e;
                bool ok = true, touched = false;
                if (best >= 0) {
                    const std::vector<int32_t> want = slots[(size_t) best].tokens;
                    if (m.tokens().empty() == false) slot_put("");   // (it can drop or move kept ones)
                    best = -2;
                    for (size_t k = 0; k < slots.size(); ++k)
                        if (slots[k].tokens == want) best = (long) k;
                    ok = best >= 0;
                    if (!ok) e = "the kept conversation went away";
                    if (ok) {
                        Slot& sl = slots[(size_t) best];
                        std::vector<uint8_t> file;
                        const uint8_t* bytes = nullptr;
                        size_t nb = 0;
                        if (sl.data) {
                            bytes = sl.data->data();
                            nb = sl.data->size();
                        } else {
                            std::ifstream f(sl.path, std::ios::binary);
                            file.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                            bytes = file.data();
                            nb = file.size();
                        }
                        touched = true;
                        ok = nb > 0 && m.slot_load(bytes, nb, (int64_t) sl.tokens.size(), e);
                    }
                }
                if (ok && bp.E < m.pos()) {
                    touched = true;
                    ok = m.restore(bp.E) && m.pos() == bp.E;
                    if (!ok) e = "no checkpoint at the cut point";
                }
                const double t0 = now_ms();
                int64_t nchanged = 0;
                for (const auto& ed : bp.eds) nchanged += (int64_t) ed.rep.size();
                if (ok) {
                    touched = true;
                    ok = m.splice_edits(bp.eds, splice_seam, e);
                }
                const auto& st1 = m.tokens();
                if (ok && !(st1.size() == bp.p_end && std::equal(st1.begin(), st1.end(), r.ids.begin()))) {
                    ok = false;
                    e = "the edited state differs from the prompt";
                }
                if (ok) {
                    edited = true;
                    std::fprintf(stderr, "ds4 serve: spliced in place: %zu edits (%lld tokens new), %zu tokens kept, %zu to read "
                                         "(%.0f ms)\n", bp.eds.size(), (long long) nchanged, bp.p_end, r.ids.size() - bp.p_end,
                                 now_ms() - t0);
                } else {
                    std::fprintf(stderr, "ds4 serve: splice-edit not done (%s)%s\n", e.c_str(),
                                 touched ? " - the prompt is read again from its start" : "");
                    if (touched) {
                        m.reset();
                        held.clear();
                    }
                }
            }
        }
        if (!edited && reorder_on && reuse && pics.empty()) {   // (every kept conversation: the one that reuses most)
            long ref_slot = -2;   // (-1: the state in the model)
            Align al;
            int64_t E = -1, best_gain = 0;
            const int64_t SA = m.splice_align(), SW = m.splice_window();
            // the cut point for a candidate: the latest checkpoint C at
            // which every stale span before C can be cut (none straddles
            // C, each leaves a window of kept tokens after it); -1: none
            const auto plan = [&](const std::vector<int32_t>& t, const Align& a2, std::vector<int64_t> cks,
                                  int64_t& cut_out) {
                cks.push_back((int64_t) t.size());
                std::sort(cks.begin(), cks.end());
                cks.erase(std::unique(cks.begin(), cks.end()), cks.end());
                static const bool dbg_p = std::getenv("DS4_DUMP_IDS") != nullptr;
                // (the prompt's turn markers: their ref positions in client
                // order, the matched ones sorted, and how long a prefix of
                // them is matched in increasing ref order)
                std::vector<int64_t> mref, kref;
                for (size_t q = 0; q < r.ids.size(); ++q)
                    if (is_mark(r.ids[q])) mref.push_back(a2.ref_of[q]);
                for (int64_t x : mref)
                    if (x >= 0) kref.push_back(x);
                std::sort(kref.begin(), kref.end());
                size_t inc = 0;
                while (inc < mref.size() && mref[inc] >= 0 && (inc == 0 || mref[inc] > mref[inc - 1])) ++inc;
                for (size_t i = cks.size(); i-- > 0;) {
                    const int64_t C = cks[i];
                    if (dbg_p && t.size() > 20000)
                        std::fprintf(stderr, "ds4 try: ref %zu C %lld ref_end %zu\n", t.size(), (long long) C,
                                     a2.ref_end);
                    if (C < 256 || C > (int64_t) a2.ref_end) continue;
                    // (the kept part ends on a line boundary: moved
                    // lines read after it start a line of their own)
                    const int last_tok = t[(size_t) C - 1];
                    if (is_nl_tok(last_tok) == false && is_mark(last_tok) == false) continue;
                    int64_t end = C, cut = 0;
                    bool ok = true;
                    // (the model must read the client's turn markers in
                    // the client's order: the kept ones are exactly the
                    // first k, in increasing ref order, all before C)
                    {
                        const size_t k = (size_t) (std::lower_bound(kref.begin(), kref.end(), C) - kref.begin());
                        ok = k <= inc && (k == 0 || mref[k - 1] < C);
                        if (ok == false && dbg_p && t.size() > 20000)
                            std::fprintf(stderr, "ds4 plan: ref %zu C %lld: %zu markers kept, %zu in order\n", t.size(),
                                         (long long) C, k, inc);
                    }
                    for (size_t k = a2.stale.size(); ok && k-- > 0;) {
                        const int64_t s = (int64_t) a2.stale[k].first, e = s + (int64_t) a2.stale[k].second;
                        if (s >= C) continue;   // (past the cut point: dropped with the tail)
                        if (e > C) {
                            ok = false;
                            break;
                        }
                        const int64_t a = (s + SA - 1) / SA * SA, b = e / SA * SA;
                        if (b <= a) continue;
                        if (b + SW > end) ok = false;
                        end -= b - a;
                        cut += b - a;
                    }
                    if (ok) {
                        // (the moved lines land at P, before the new turn's
                        // marker: they must end on a group boundary and start
                        // on one or after a plain token, or a later cut of
                        // them takes that marker along)
                        const int64_t P = C - cut;
                        int64_t L = 0;
                        for (size_t q = 0; q < r.ids.size(); ++q) {
                            if (a2.ref_of[q] >= 0 && a2.ref_of[q] < C) continue;
                            if (is_mark(r.ids[q])) break;
                            ++L;
                        }
                        if (L > 0 && ((P + L) % SA != 0 || (P % SA != 0 && is_mark(t[(size_t) C - 1])))) {
                            ok = false;
                            if (dbg_p && t.size() > 20000)
                                std::fprintf(stderr, "ds4 plan: ref %zu C %lld: moved %lld at %lld off the grid\n",
                                             t.size(), (long long) C, (long long) L, (long long) P);
                        }
                    }
                    if (dbg_p && t.size() > 20000)
                        std::fprintf(stderr, "ds4 try: ref %zu C %lld ok %d\n", t.size(), (long long) C, (int) ok);
                    if (ok) {
                        cut_out = cut;
                        return C;
                    }
                }
                return (int64_t) -1;
            };
            const auto consider = [&](long k, const std::vector<int32_t>& t, const std::vector<int64_t>& cks) {
                Align a2;
                // (a state that starts unlike the prompt is scrambled)
                if (t.empty() || r.ids.empty() || t[0] != r.ids[0]) return;
                static const bool dbg = std::getenv("DS4_DUMP_IDS") != nullptr;
                const bool aligned = align_to(t, r.ids, a2);
                if (dbg && t.size() > 20000)
                    std::fprintf(stderr, "ds4 cand: ref %zu aligned %d matched %zu ref_end %zu stale %zu cks %zu\n",
                                 t.size(), (int) aligned, a2.matched, a2.ref_end, a2.stale.size(), cks.size());
                if (aligned == false) return;
                whole_lines(t, r.ids, a2, SA);
                int64_t cut = 0;
                const int64_t C = plan(t, a2, cks, cut);
                if (dbg && t.size() > 20000)
                    std::fprintf(stderr, "ds4 cand: ref %zu after lines: matched %zu stale %zu; C %lld cut %lld\n",
                                 t.size(), a2.matched, a2.stale.size(), (long long) C, (long long) cut);
                if (C >= 256 && C - cut > best_gain) {
                    best_gain = C - cut;
                    E = C;
                    ref_slot = k;
                    al = std::move(a2);
                }
            };
            consider(-1, m.tokens(), m.live_positions());
            for (size_t k = 0; k < slots.size(); ++k) consider((long) k, slots[k].tokens, slots[k].cks);
            // the live state's plain prefix wins over a kept one's splice
            // that reuses less (an old copy kept being picked, its stale
            // span cut, the newer state's tail read again every turn)
            if (ref_slot >= 0 && !m.tokens().empty()) {
                const auto& lt = m.tokens();
                size_t lp = 0;
                while (lp < lt.size() && lp < r.ids.size() && lt[lp] == r.ids[lp]) ++lp;
                int64_t lc = 0;
                for (int64_t c : m.live_positions())
                    if (c <= (int64_t) lp) lc = std::max(lc, c);
                if (lc >= 256 && lc > best_gain) {
                    std::fprintf(stderr, "ds4 serve: the live state reuses %lld (a kept one's splice %lld): kept as is\n",
                                 (long long) lc, (long long) best_gain);
                    ref_slot = -2;
                    E = -1;
                }
            }
            const std::vector<int32_t>* ref = ref_slot >= 0 ? &slots[(size_t) ref_slot].tokens
                                              : ref_slot == -1 ? &m.tokens() : nullptr;
            if (ref != nullptr) {
                size_t n_alike = 0;
                if (E < 256) {
                    // (nothing kept to resume at: the prompt stays as sent)
                } else if (al.stale.empty()) {
                    r.ids = alike_first(r.ids, al, INT64_MAX, n_alike);
                } else {
                    std::string e;
                    if (ref_slot < 0) slot_put("");                       // (the conversation it was stays kept)
                    bool ok = true;
                    if (ref_slot >= 0 && m.tokens().empty() == false) {
                        const std::vector<int32_t> want = slots[(size_t) ref_slot].tokens;
                        slot_put("");   // (it can drop or move kept ones: find the target again)
                        ref_slot = -2;
                        for (size_t k = 0; k < slots.size(); ++k)
                            if (slots[k].tokens == want) ref_slot = (long) k;
                        ok = ref_slot >= 0;
                        if (ok == false) e = "the kept conversation went away";
                    }
                    if (ok && ref_slot >= 0) {
                        Slot& sl = slots[(size_t) ref_slot];
                        std::vector<uint8_t> file;
                        const uint8_t* bytes = nullptr;
                        size_t nb = 0;
                        if (sl.data) {
                            bytes = sl.data->data();
                            nb = sl.data->size();
                        } else {
                            std::ifstream f(sl.path, std::ios::binary);
                            file.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                            bytes = file.data();
                            nb = file.size();
                        }
                        ok = nb > 0 && m.slot_load(bytes, nb, (int64_t) sl.tokens.size(), e);
                    }
                    // (back to the cut point; the tail past it is read again)
                    if (ok && E < m.pos()) {
                        ok = m.restore(E) && m.pos() == E;
                        if (ok == false) e = "no checkpoint at the cut point";
                    }
                    int64_t cut = 0;
                    for (size_t k = al.stale.size(); ok && k-- > 0;) {   // (last first: earlier spans keep their place)
                        const int64_t s = (int64_t) al.stale[k].first;
                        if (s >= E) continue;
                        const int64_t a = (s + SA - 1) / SA * SA;
                        const int64_t b = (s + (int64_t) al.stale[k].second) / SA * SA;
                        if (b > a) {
                            ok = m.splice_out(a, b - a, e);
                            cut += ok ? b - a : 0;
                        }
                    }
                    // the kept state must equal the prompt's tokens matched
                    // before the cut point, in the state's order; the rest
                    // of the prompt follows in the client's order
                    std::vector<int> kept, rest;
                    if (ok) {
                        std::vector<std::pair<int64_t, int>> early;
                        for (size_t t = 0; t < r.ids.size(); ++t) {
                            const int64_t q = al.ref_of[t];
                            if (q >= 0 && q < E) early.push_back({q, r.ids[t]});
                            else rest.push_back(r.ids[t]);
                        }
                        std::sort(early.begin(), early.end());
                        for (const auto& x : early) kept.push_back(x.second);
                        const auto& stt = m.tokens();
                        ok = kept.size() == stt.size() && std::equal(kept.begin(), kept.end(), stt.begin());
                        if (ok == false) e = "the cut state differs from the prompt's kept tokens";
                    }
                    if (ok) {
                        m.checkpoint(3);   // (the next prompt behind this start goes back here)
                        n_alike = kept.size();
                        kept.insert(kept.end(), rest.begin(), rest.end());
                        r.ids.swap(kept);
                        std::fprintf(stderr, "ds4 serve: spliced %lld stale tokens, cut point %lld; "
                                     "%zu tokens reused, %zu to read\n", (long long) cut, (long long) E,
                                     n_alike, r.ids.size() - n_alike);
                    } else {
                        std::fprintf(stderr, "ds4 serve: splice failed (%s) - reading the prompt again\n", e.c_str());
                        m.reset();
                        held.clear();
                        r.ids = alike_first(r.ids, al, INT64_MAX, n_alike);
                    }
                }
                if (n_alike >= 256 && n_alike + 1 < r.ids.size()) {
                    r.pin = (long long) n_alike;
                    std::fprintf(stderr, "ds4 serve: alike tokens first: %zu, new after them: %zu\n", n_alike,
                                 r.ids.size() - n_alike);
                }
            }
        }
        if (const char* dump = std::getenv("DS4_DUMP_IDS")) {
            // (debug: the token order the model reads, as int32)
            std::ofstream f(dump, std::ios::binary | std::ios::trunc);
            f.write((const char*) r.ids.data(), (std::streamsize) (r.ids.size() * sizeof(int)));
        }
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        static const bool blocks_on = std::getenv("DS4_SPLICE_BLOCKS") && std::atoi(std::getenv("DS4_SPLICE_BLOCKS")) != 0;
        if (blocks_on && pics.empty() && r.ids.size() > c + 256) {   // (dry run: what the library would give)
            const double tb = now_ms();
            std::vector<int32_t> live(st.begin(), st.begin() + (long) c);
            std::vector<const std::vector<int32_t>*> srcs{&live};
            for (const Slot& sl : slots) srcs.push_back(&sl.tokens);
            int64_t wp = 0;
            const auto hits = find_blocks(r.ids, c, srcs, m.splice_align(), 256, wp);
            int64_t cov = 0, big = 0;
            std::vector<char> from(srcs.size(), 0);
            for (const auto& h : hits) {
                cov += (int64_t) h.n;
                big = std::max<int64_t>(big, (int64_t) h.n);
                from[(size_t) h.src] = 1;
            }
            const size_t left = r.ids.size() - c;
            std::fprintf(stderr, "ds4 blocks (dry run): %zu to read; %lld of them (%.0f%%) in %zu blocks of 256+ from %d "
                                 "sources (largest %lld; %lld more at the wrong parity); %.0f ms\n",
                         left, (long long) cov, 100.0 * (double) cov / (double) left, hits.size(),
                         (int) std::count(from.begin(), from.end(), 1), (long long) big, (long long) wp, now_ms() - tb);
        }
        // the same tokens with another picture (or none) do not share the state from that picture's span on
        auto same = [](const std::vector<Pic>& v, const Pic& q) {
            for (const Pic& x : v)
                if (x.start == q.start && x.hash == q.hash) return true;
            return false;
        };
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        bool took = false, continued = false;   // (continued: the conversation goes on - its anchor stays)
        // the conversation continues, reused in place - a conversation has to be in the model first: a fresh engine
        // (the start of a run, so every restart) holds 0 tokens, and every prompt "continues" that to token 0; taken
        // as a continuation, the take below would never run for the folder's conversations, the case they are for
        if ((n_state > 0 || slot_max <= 0) && c == n_state && n_state < r.ids.size()) {   // (slots off: as before)
            from = n_state;
            continued = true;
        }
        else {
            // CONVERSATION SLOTS: this prompt does not continue the conversation in the model - that one is set aside
            // first (another one takes the model, and its state is lost otherwise), then the kept conversation this
            // prompt does continue is taken back - if it goes further back than the checkpoints the state itself holds
            if (held.empty()) slot_put("");
            const int64_t limit = (int64_t) std::min(c, r.ids.size() - 1);
            const int64_t rs = m.restorable(limit);
            bool whole = false;
            const size_t back = pics.empty() ? slot_take(r.ids, rs > 0 ? (size_t) rs : 0, whole) : 0;
            if (back > 0) {
                from = back;
                took = true;
                continued = whole;
            } else if (rs >= 0 && m.restore(limit))
                from = (size_t) m.pos();
            else m.reset();
        }
        if (took || from == 0) held.clear();   // (a slot is text only: the state that comes back holds no pictures)
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context()) {
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            continue;
        }
        std::string err;
        bool ok = true;
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                ++encoded;
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span (" + std::to_string(q.len) + " tokens at " + std::to_string(q.start) +
                          ") is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (ok) {
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        // the prompt in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64 tokens), a checkpoint before its last
        // token and one after it: the next turn re-renders this one, and shares its start up to one of them
        const size_t total = r.ids.size();
        size_t done = from, next_pp = from + 64;
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        // (pin=N: the read stops at N for a checkpoint of the shared start; only when this read crosses N, and N leaves
        //  the prompt's last token to the usual end)
        const size_t pin_at = (r.pin >= 256 && (size_t) r.pin + 1 < total) ? (size_t) r.pin : 0;
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds) and goes back to the state before this prompt - kept here first, in checkpoint 1, which the prompt's
        // own end overwrites as before - so the next request still shares the conversation up to it
        if (from > 0 && from < total) m.checkpoint(1);
        m.set_cancel([&in] { return in.stop.load(); });
        bool stopped = false;
        // (a long read keeps a checkpoint before a turn marker
        // every DS4_CK_EVERY tokens or so: a later turn goes back there)
        static const size_t ck_every = [] {
            const char* v = std::getenv("DS4_CK_EVERY");
            return v ? (size_t) std::max(0, std::atoi(v)) : (size_t) 2048;
        }();
        size_t last_mark = done, mark_at = 0;
        const auto next_mark = [&] {
            mark_at = 0;
            if (ck_every == 0) return;
            for (size_t q = std::max(done + 1, last_mark + ck_every); q + 1 < total; ++q)
                if (is_mark(r.ids[q])) {
                    mark_at = q;
                    return;
                }
        };
        next_mark();
        while (ok && done < total) {
            if (in.stop.load()) {
                stopped = true;
                break;
            }
            size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            if (pin_at > done && pin_at < done + n) n = pin_at - done;
            if (mark_at > done && mark_at < done + n) n = mark_at - done;
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            if (ok && pin_at != 0 && done == pin_at) m.checkpoint(3);
            if (ok && mark_at != 0 && done == mark_at) {
                m.mark();
                last_mark = done;
                next_mark();
            }
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP %zu %zu %.1f %.1f %zu\n", done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
        }
        m.set_cancel(nullptr);
        m.set_images({});
        if (stopped || (!ok && m.cancelled())) {
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), now_ms() - t0, back ? from : (size_t) 0);
            std::fflush(stdout);
            continue;
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            m.reset();
            held.clear();
            continue;
        }
        held = std::move(pics);
        m.checkpoint(0);
        m.mark();   // (each turn's end stays a place to go back to)
        if (!continued && slot_max > 0) m.pin();   // (a new run of prompts: where it may be rendered again from)
        const double t1 = now_ms();
        long long produced = 0;
        const char* finish = "length";
        const long long limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        long long d_off = 0, d_acc = 0;
        bool cool = false;
        int tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, counter++);
        std::vector<int> got;
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
            const strata::ds4::Stats& s = m.stats();
            const double ms = now_ms() - t1;
            const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
            const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts);
            std::string out = "STAT";
            char b[160];
            std::snprintf(b, sizeof b, " tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f ram=%.3f promo=%.3f engram_ms=%.3f",
                          ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                          ex > 0 ? 1.0 - cx / ex : 1.0, cx / fed, (double) (s.ram_experts - s_dec.ram_experts) / fed,
                          (double) (s.swaps - s_dec.swaps) / fed, (s.engram_ms - s_dec.engram_ms) / fed);
            out += b;
            const long long io = io_read_bytes();
            if (io >= 0 && io_dec >= 0) {
                std::snprintf(b, sizeof b, " disk_mb=%.3f", (double) (io - io_dec) / 1e6 / fed);
                out += b;
            }
            if (d_off > 0) {
                std::snprintf(b, sizeof b, " drafts=%.3f", (double) d_acc / (double) d_off);
                out += b;
            }
            if (s.skipped > s_dec.skipped) {   // (STRATA_DS4_DISK_SKIP_W: experts left out a token)
                std::snprintf(b, sizeof b, " skipped=%.3f", (double) (s.skipped - s_dec.skipped) / fed);
                out += b;
            }
            std::printf("%s\n", out.c_str());
        };
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            std::printf("T %d\n", tok);
            ++produced;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= limit || m.pos() >= m.max_context()) break;
            if (got.empty()) {   // the next round: the drafts (if any) and the token after them
                const int room = (int) std::min<long long>(limit - produced, m.max_context() - m.pos());
                int off = 0, acc = 0;
                // a round whose drafts were all below the confidence floor: the next one does without the drafter
                // (a request's spec / dspark_conf keys over the engine's: the speed calibration measures both ways)
                const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !cool;
                if (!round(m, tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                           r.min_p, r.seed, counter, got,
                           off, acc, err, r.cum)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    finish = "cancel";
                    break;
                }
                d_off += off;
                d_acc += acc;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        // (a stop within accepted drafts leaves the state past it: the next prompt shares less with it, and goes back to
        //  the checkpoint at this prompt's end)
        stat_line();
        const double t2 = now_ms();
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    d_acc, d_off, from);
        std::fflush(stdout);
        const auto& s = m.stats();
        std::fprintf(stderr,
                     "maya ds4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0,
                     s.experts ? 100.0 * (double) s.cpu_experts / (double) s.experts : 0.0);
        if (d_off > 0) std::fprintf(stderr, "maya ds4 serve: drafts %lld of %lld kept; %s\n", d_acc, d_off,
                                    per_position(s_dec, s).c_str());
        if (s.kx_experts > 0 || s.kx_fails > 0)   // (DS4_EXPERTD; a line of its own)
            std::fprintf(stderr, "maya ds4 serve: expert daemon %.1f%% of the experts (%lld moved to the lane), %lld failed requests, "
                         "%.1f ms waited, %lld swapped in (%lld refused)\n",
                         s.experts ? 100.0 * (double) s.kx_experts / (double) s.experts : 0.0, (long long) s.kx_moved,
                         (long long) s.kx_fails, s.kx_wait_ms, (long long) s.kx_swaps, (long long) s.kx_swap_fails);
        {   // DS4_PROF: this turn's decode, by stage (ms a token)
            static long long pt0 = 0;
            static double pm0[6] = {};
            if (s.prof_tokens > pt0) {
                const double pt = (double) (s.prof_tokens - pt0);
                double d[6], sum = 0;
                for (int i = 0; i < 6; ++i) {
                    d[i] = (s.prof_ms[i] - pm0[i]) / pt;
                    sum += d[i];
                    pm0[i] = s.prof_ms[i];
                }
                pt0 = s.prof_tokens;
                std::fprintf(stderr, "maya ds4 serve: decode GPU per token %.2f ms: between layers %.2f, attention %.2f, "
                             "routing + shared %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f\n",
                             sum, d[0], d[1], d[2], d[3], d[4], d[5]);
            }
        }
        m.save_profile();
    }
    // the engine ends (QUIT or the end of its input): the conversation in the model is set aside, and every one in
    // RAM is written to the folder, so the next run reads on from them
    if (slot_max > 0) {
        if (held.empty()) slot_put(" as the engine ended");
        size_t written = 0;
        while (true) {   // (a write can take the folder's least recently used conversation away: the list is asked again)
            size_t i = SIZE_MAX;
            for (size_t k = 0; k < slots.size(); ++k)
                if (slots[k].data) {
                    i = k;
                    break;
                }
            if (i == SIZE_MAX) break;
            if (slot_write_file(slots[i])) ++written;
            else slot_drop(i);   // (its state could not be written: there is nothing more to do at the engine's end)
        }
        if (written > 0) {
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            char msg[256];
            std::snprintf(msg, sizeof msg, "wrote %zu conversation(s) to %s; %d there in all", written,
                          slot_dir.string().c_str(), disk);
            note(msg);
        }
    }
    settle_stop();   // (the idle thread must be off the cache before the profile is saved and the engine ends)
    m.save_profile();
    return 0;
}


// ---- Strata addition: teacher-forced scoring (the quality gate) -------------------------------------------------
// --score IDS.json --score-out RUN.tsv: every segment of a tools/quant_quality.py ids file (name + ids) is read through
// the model from a fresh state; per position the log-probability of the text's next token and the 64 most likely
// tokens go out in the STRATA_LOGPOS row format that `quant_quality.py compare` reads (natural logs, full-vocabulary
// log-softmax in double).  RUN.tsv.seg.json lists each segment's first line and size.
struct ScoreSeg {
    std::string name;
    std::vector<int> ids;
};

bool read_score_ids(const std::string& path, std::vector<ScoreSeg>& segs, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = path + ": cannot open";
        return false;
    }
    std::string s;
    char buf[1 << 16];
    size_t k;
    while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k);
    std::fclose(f);
    size_t p = 0;
    for (;;) {
        const size_t n0 = s.find("\"name\"", p);
        if (n0 == std::string::npos) break;
        const size_t q0 = s.find('"', s.find(':', n0) + 1);
        const size_t q1 = s.find('"', q0 + 1);
        const size_t i0 = s.find("\"ids\"", q1);
        if (q0 == std::string::npos || q1 == std::string::npos || i0 == std::string::npos) break;
        const size_t b0 = s.find('[', i0), b1 = s.find(']', b0);
        if (b0 == std::string::npos || b1 == std::string::npos) break;
        ScoreSeg g;
        g.name = s.substr(q0 + 1, q1 - q0 - 1);
        g.ids = parse_ids(s.substr(b0 + 1, b1 - b0 - 1));
        segs.push_back(std::move(g));
        p = b1;
    }
    if (segs.empty()) {
        err = path + ": no segments (expected quant_quality.py's ids file)";
        return false;
    }
    return true;
}

static bool st_meta_tokens(const std::string& path, std::vector<int>& out, std::string& err);

int score(strata::ds4::Model& m, const std::string& ids_path, const std::string& out_path, int window, int eos) {
    std::vector<ScoreSeg> segs;
    std::string err;
    // a slot's .meta: its tokens [a, b) (DS4_SCORE_RANGE=a:b) as one segment
    const auto ends = [&](const char* x) {
        const size_t k = std::strlen(x);
        return ids_path.size() > k && ids_path.compare(ids_path.size() - k, k, x) == 0;
    };
    if (ends(".meta") || ends(".ids")) {
        ScoreSeg sg;
        sg.name = "slot";
        if (!st_meta_tokens(ids_path, sg.ids, err)) {
            std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
            return 1;
        }
        if (const char* rg = std::getenv("DS4_SCORE_RANGE")) {
            const long long a = std::atoll(rg), b = std::strchr(rg, ':') ? std::atoll(std::strchr(rg, ':') + 1) : (long long) sg.ids.size();
            sg.ids = std::vector<int>(sg.ids.begin() + std::min<long long>(a, (long long) sg.ids.size()),
                                      sg.ids.begin() + std::min<long long>(b, (long long) sg.ids.size()));
        }
        segs.push_back(std::move(sg));
    } else if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    // DS4_SCORE_PREFIX=N (a test of the long-prompt layer skip): the segments are joined into one stream, its first N
    // tokens are read as one prompt (logits of the last row only: the path a long prompt takes, where the skip applies)
    // and only the positions after them are scored, in windows
    const long long prefix_n = std::getenv("DS4_SCORE_PREFIX") ? std::atoll(std::getenv("DS4_SCORE_PREFIX")) : 0;
    if (prefix_n > 0 && segs.size() > 1) {
        ScoreSeg all;
        all.name = "concat";
        for (const ScoreSeg& sg : segs) all.ids.insert(all.ids.end(), sg.ids.begin(), sg.ids.end());
        segs.assign(1, all);
    }
    std::FILE* out = std::fopen(out_path.c_str(), "wb");
    std::FILE* sj = std::fopen((out_path + ".seg.json").c_str(), "wb");
    if (out == nullptr || sj == nullptr) {
        std::fprintf(stderr, "strata-dsv41: cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(sj, "{\"segments\": [");
    const int V = m.vocab();
    long long line = 0;
    std::vector<float> lg;
    std::vector<int> order;
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        const long long first = line;
        double sum_nll = 0.0;
        long long top1 = 0, cnt = 0;
        size_t a0 = 0;
        if (prefix_n > 0 && (size_t) prefix_n + 1 < ids.size()) {
            a0 = (size_t) prefix_n;
            // DS4_SCORE_UNITS=1: the prefix in feed_unit() pieces, as serve reads a prompt
            const bool units = std::getenv("DS4_SCORE_UNITS") != nullptr;
            const size_t u = units ? (size_t) std::max(1, m.feed_unit()) : a0;
            for (size_t q = 0; q < a0; q += u)
                if (!m.feed(ids.data() + q, (int) std::min(u, a0 - q), 1, err)) {
                    std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                    return 1;
                }
            if (units) std::fprintf(stderr, "score: the prefix in pieces of %zu\n", u);
        }
        for (size_t a = a0; a + 1 < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - 1 - a);   // (positions a .. a+T-1 are scored)
            if (!m.feed(ids.data() + a, T, T, err) || !m.logits(lg) || lg.size() < (size_t) T * (size_t) V) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.empty() ? "no logits" : err.c_str());
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                const float* r = lg.data() + (size_t) t * (size_t) V;
                float mx = r[0];
                for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
                double z = 0.0;
                for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
                const double lz = std::log(z) + (double) mx;
                order.resize((size_t) V);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int x, int y) {
                    return r[x] > r[y] || (r[x] == r[y] && x < y);
                });
                const int tgt = ids[a + (size_t) t + 1];
                const double lt = (double) r[tgt] - lz;
                const double la = (double) r[order[0]] - lz;
                const double le = eos >= 0 && eos < V ? (double) r[eos] - lz : -1e30;
                // logp(target) with the end token left out of the distribution, as the engine's row has it
                const double lt_ne = tgt == eos ? lt : std::log(std::max(1e-300, std::exp(lt) / std::max(1e-300, 1.0 - std::exp(le))));
                std::fprintf(out, "%zu\t%d\t%.6f\t%d\t%.6f\t%d\t%.6f\t%.6f", a + (size_t) t, tgt, lt, order[0], la,
                             order[0] == tgt ? 1 : 0, le, lt_ne);
                for (int j = 0; j < 64; ++j) std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) r[order[(size_t) j]] - lz);
                std::fprintf(out, "\n");
                ++line;
                ++cnt;
                top1 += order[0] == tgt;
                sum_nll -= lt;
            }
            a += (size_t) T;
        }
        std::fprintf(sj, "%s{\"name\": \"%s\", \"first_line\": %lld, \"end_line\": %lld, \"n\": %lld, \"ppl\": %.6f, \"top1\": %.6f}", si ? ", " : "",
                     segs[si].name.c_str(), first, first + cnt, cnt, std::exp(sum_nll / (double) std::max(1LL, cnt)),
                     (double) top1 / (double) std::max(1LL, cnt));
        std::fprintf(stderr, "score %s: %lld positions, ppl %.4f, top-1 %.4f (%.0f s)\n", segs[si].name.c_str(), cnt,
                     std::exp(sum_nll / (double) std::max(1LL, cnt)), (double) top1 / (double) std::max(1LL, cnt),
                     (now_ms() - t0) / 1000.0);
        std::fflush(out);
    }
    std::fprintf(sj, "]}\n");
    std::fclose(sj);
    std::fclose(out);
    return 0;
}
// ---- block-cache research: splice quality ---------------------------------------------------------------------
// --splice-test MAIN.meta --splice-junk JUNK.meta: a block B = MAIN[a1, E) whose state was made after a foreign
// span (JUNK's tokens, cut out again with splice_out), the last S tokens of B read fresh (the seam), against the
// same tokens read whole; MAIN[E, E + nc) scored. Env: DS4_ST_A1, DS4_ST_B (B's length), DS4_ST_NC (64),
// DS4_ST_JUNK ("512,4096": junk lengths), DS4_ST_SEAMS ("0,128,256"), DS4_ST_JUNK_OFF (0).
static bool st_meta_tokens(const std::string& path, std::vector<int>& out, std::string& err) {
    // (a .ids file: a raw int32 dump, DS4_DUMP_IDS's)
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".ids") == 0) {
        std::ifstream g(path, std::ios::binary | std::ios::ate);
        const std::streamsize sz = g.tellg();
        if (!g || sz <= 0 || sz % 4 != 0) {
            err = path + ": not an ids dump";
            return false;
        }
        std::vector<int32_t> t((size_t) sz / 4);
        g.seekg(0);
        g.read((char*) t.data(), sz);
        out.assign(t.begin(), t.end());
        return true;
    }
    std::ifstream f(path, std::ios::binary);
    uint64_t m0 = 0, il = 0, bytes = 0, nt = 0;
    if (!f.read((char*) &m0, 8) || m0 != strata::ds4::slotmeta::kMagic || !f.read((char*) &il, 8)) {
        err = path + ": not a slot .meta";
        return false;
    }
    f.seekg((std::streamoff) il, std::ios::cur);
    if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (1ull << 24)) {
        err = path + ": damaged";
        return false;
    }
    std::vector<int32_t> t(nt);
    if (!f.read((char*) t.data(), (std::streamsize) (nt * 4))) {
        err = path + ": cut short";
        return false;
    }
    out.assign(t.begin(), t.end());
    return true;
}

static std::vector<int> st_list(const char* env, const char* def) {
    const char* v = std::getenv(env);
    return parse_ids(v ? v : def);
}

// teacher-forced: feeds c[0, nc) (the state ends just before c[0]) and keeps each row's log-softmax (row t: c[t + 1])
static bool st_score(strata::ds4::Model& m, const std::vector<int>& c, int nc, std::vector<float>& lp, std::string& err) {
    const int V = m.vocab();
    lp.assign((size_t) nc * (size_t) V, 0.0f);
    std::vector<float> lg;
    for (int a = 0; a < nc;) {
        const int T = std::min(16, nc - a);
        if (!m.feed(c.data() + a, T, T, err) || !m.logits(lg) || lg.size() < (size_t) T * (size_t) V) {
            if (err.empty()) err = "no logits";
            return false;
        }
        for (int t = 0; t < T; ++t) {
            const float* r = lg.data() + (size_t) t * (size_t) V;
            float mx = r[0];
            for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
            double z = 0.0;
            for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
            const float lz = (float) (std::log(z) + (double) mx);
            float* o = lp.data() + (size_t) (a + t) * (size_t) V;
            for (int v = 0; v < V; ++v) o[v] = r[v] - lz;
        }
        a += T;
    }
    return true;
}

static void st_report(const char* name, const std::vector<float>& ref, const std::vector<float>& var, const std::vector<int>& c,
                      int nc, int V) {
    double kl = 0.0, klmax = 0.0, nr = 0.0, nv = 0.0;
    int agree = 0;
    for (int t = 0; t + 1 < nc; ++t) {
        const float* p = ref.data() + (size_t) t * (size_t) V;
        const float* q = var.data() + (size_t) t * (size_t) V;
        double k = 0.0;
        int ap = 0, aq = 0;
        for (int v = 0; v < V; ++v) {
            const double pv = std::exp((double) p[v]);
            if (pv > 1e-12) k += pv * ((double) p[v] - (double) q[v]);
            if (p[v] > p[ap]) ap = v;
            if (q[v] > q[aq]) aq = v;
        }
        kl += k;
        klmax = std::max(klmax, k);
        agree += ap == aq;
        nr -= p[c[(size_t) t + 1]];
        nv -= q[c[(size_t) t + 1]];
    }
    const int n = std::max(1, nc - 1);
    std::fprintf(stderr, "splice-test %-26s KL mean %.5f max %.4f | top-1 agrees %5.1f%% | ppl ref %.4f this %.4f\n", name,
                 kl / n, klmax, 100.0 * agree / n, std::exp(nr / n), std::exp(nv / n));
}

// DS4_ST_EDIT_A=a: the edit test instead - MAIN[a, a + D) replaced by
// N junk tokens (DS4_ST_EDIT_D, _N), the state to E (DS4_ST_EDIT_E);
// splice_edit on MAIN's state against the edited prompt read whole
static int splice_edit_test(strata::ds4::Model& m, const std::vector<int>& ids, const std::vector<int>& junk) {
    std::string err;
    const auto num = [](const char* k, long long d) { return std::getenv(k) ? std::atoll(std::getenv(k)) : d; };
    const int64_t al = m.splice_align();
    const int64_t a = num("DS4_ST_EDIT_A", 8192) / al * al, D = num("DS4_ST_EDIT_D", 512) / al * al;
    const int64_t N = std::min<int64_t>(num("DS4_ST_EDIT_N", D) / al * al, D), E = num("DS4_ST_EDIT_E", 16384);
    const int64_t joff = num("DS4_ST_JUNK_OFF", 0), mk = num("DS4_ST_MARK", a / 2048 * 2048);
    const int nc = (int) num("DS4_ST_NC", 64);
    if (E + nc + 1 > (int64_t) ids.size() || a + D + 128 > E || joff + N > (int64_t) junk.size()) {
        std::fprintf(stderr, "strata-dsv41: the edit test does not fit the tokens given\n");
        return 1;
    }
    const std::vector<int32_t> rep(junk.begin() + joff, junk.begin() + joff + N);
    const std::vector<int> cont(ids.begin() + E, ids.begin() + E + nc + 1);
    std::vector<int> edited(ids.begin(), ids.begin() + a);
    edited.insert(edited.end(), rep.begin(), rep.end());
    edited.insert(edited.end(), ids.begin() + a + D, ids.begin() + E);
    const int V = m.vocab();
    const double t0 = now_ms();
    std::vector<float> ref, var;
    m.reset();
    if (!m.feed(edited.data(), (int) edited.size(), 1, err) || !st_score(m, cont, nc, ref, err)) {
        std::fprintf(stderr, "strata-dsv41: the reference: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "splice-edit: [%lld, %lld) -> %lld junk tokens, state to %lld, mark at %lld; reference (%.0f s)\n",
                 (long long) a, (long long) (a + D), (long long) N, (long long) E, (long long) mk, (now_ms() - t0) / 1000.0);
    // the unedited state, a mark before the edit (serve keeps them);
    // an edit moves rows after a, so each run reads [mk, E) again
    m.reset();
    if (!m.feed(ids.data(), (int) mk, 1, err)) return 1;
    m.mark();
    const auto unedited = [&] {
        if (!m.restore(mk) || m.pos() != mk || !m.feed(ids.data() + mk, (int) (E - mk), 1, err)) {
            std::fprintf(stderr, "strata-dsv41: the unedited state: %s\n", err.c_str());
            return false;
        }
        return true;
    };
    // the stale state as is (no edit), for scale
    std::vector<float> stale;
    if (!unedited() || !st_score(m, cont, nc, stale, err)) return 1;
    st_report("unedited state", ref, stale, cont, nc, V);
    for (int x : st_list("DS4_ST_SEAMS", "0,32,128,512")) {
        if (!unedited()) return 1;
        const double t1 = now_ms();
        if (!m.splice_edit(a, D, rep, x, err)) {
            std::fprintf(stderr, "strata-dsv41: splice_edit: %s\n", err.c_str());
            return 1;
        }
        const double ms = now_ms() - t1;
        if (!st_score(m, cont, nc, var, err)) return 1;
        char nm[64];
        std::snprintf(nm, sizeof nm, "edited, seam %d (%.1f s)", x, ms / 1000.0);
        st_report(nm, ref, var, cont, nc, V);
    }
    std::fprintf(stderr, "splice-edit: done (%.0f s)\n", (now_ms() - t0) / 1000.0);
    return 0;
}

// DS4_ST_DRIFT=1 (the edit test's settings): the followers' compressed
// rows by distance from the edit, against the edited prompt read whole
static int splice_drift_test(strata::ds4::Model& m, const std::vector<int>& ids, const std::vector<int>& junk) {
    std::string err;
    const auto num = [](const char* k, long long d) { return std::getenv(k) ? std::atoll(std::getenv(k)) : d; };
    const int64_t al = m.splice_align();
    const int64_t a = num("DS4_ST_EDIT_A", 8192) / al * al, D = num("DS4_ST_EDIT_D", 512) / al * al;
    const int64_t N = std::min<int64_t>(num("DS4_ST_EDIT_N", D) / al * al, D), E = num("DS4_ST_EDIT_E", 16384);
    const int64_t joff = num("DS4_ST_JUNK_OFF", 0), mk = num("DS4_ST_MARK", a / 2048 * 2048);
    if (a + D + 128 > E || E > (int64_t) ids.size() || joff + N > (int64_t) junk.size()) {
        std::fprintf(stderr, "strata-dsv41: the drift test does not fit the tokens given\n");
        return 1;
    }
    // DS4_ST_EDITS=k edits, DS4_ST_EDIT_GAP apart (the first at a), each
    // D tokens -> N junk tokens (junk taken further along each time)
    const int64_t K = std::max<long long>(1, num("DS4_ST_EDITS", 1)), gap = num("DS4_ST_EDIT_GAP", 2048) / al * al;
    std::vector<strata::ds4::Model::Edit> eds;
    for (int64_t i = 0; i < K; ++i) {
        strata::ds4::Model::Edit ed;
        ed.a = a + i * gap;
        ed.d = D;
        const int64_t jo = joff + i * N;
        if (ed.a + D + 128 > E || jo + N > (int64_t) junk.size() || (i > 0 && ed.a < eds.back().a + D)) {
            std::fprintf(stderr, "strata-dsv41: edit %lld does not fit\n", (long long) i);
            return 1;
        }
        ed.rep.assign(junk.begin() + jo, junk.begin() + jo + N);
        eds.push_back(std::move(ed));
    }
    const std::vector<int32_t> rep = eds[0].rep;
    std::vector<int> edited;
    int64_t at = 0;
    for (const auto& ed : eds) {
        edited.insert(edited.end(), ids.begin() + at, ids.begin() + ed.a);
        edited.insert(edited.end(), ed.rep.begin(), ed.rep.end());
        at = ed.a + ed.d;
    }
    edited.insert(edited.end(), ids.begin() + at, ids.begin() + E);
    // (the followers of the last edit start at f0, in the edited numbering)
    const int64_t E2 = (int64_t) edited.size(), f0 = eds.back().a + ((int64_t) eds.size() - 1) * (N - D) + N;
    const int ns = m.kv_sources();
    const std::vector<int64_t> edges = {0, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 1 << 30};
    // every source's follower rows of the state as it is now
    const auto grab = [&](std::vector<std::vector<float>>& out) {
        out.assign((size_t) ns, {});
        for (int s = 0; s < ns; ++s) {
            const int64_t R = m.kv_ratio(s), r0 = (f0 + R - 1) / R, r1 = E2 / R;
            if (r1 > r0 && !m.kv_rows(s, r0, r1 - r0, out[(size_t) s], err)) return false;
        }
        return true;
    };
    std::vector<std::vector<float>> ref, var;
    const auto show = [&](const char* name) {
        for (int s = 0; s < ns; ++s) {
            const int64_t R = m.kv_ratio(s), r0 = (f0 + R - 1) / R, nrow = E2 / R - r0;
            if (nrow <= 0 || ref[(size_t) s].empty()) continue;
            const int hd = (int) (ref[(size_t) s].size() / (size_t) nrow);
            std::vector<double> sum(edges.size(), 0.0);
            std::vector<long> cnt(edges.size(), 0);
            const size_t rows = ref[(size_t) s].size() / (size_t) hd;
            for (size_t r = 0; r < rows; ++r) {
                const float* x = ref[(size_t) s].data() + r * (size_t) hd;
                const float* y = var[(size_t) s].data() + r * (size_t) hd;
                double dd = 0, xx = 0;
                for (int e = 0; e < hd; ++e) {
                    dd += ((double) x[e] - y[e]) * ((double) x[e] - y[e]);
                    xx += (double) x[e] * x[e];
                }
                const int64_t dist = (r0 + (int64_t) r) * R - f0;
                size_t b = 0;
                while (b + 1 < edges.size() && dist >= edges[b + 1]) ++b;
                sum[b] += std::sqrt(dd / std::max(1e-30, xx));
                ++cnt[b];
            }
            std::string line;
            for (size_t b = 0; b + 1 < edges.size(); ++b)
                if (cnt[b] > 0) {
                    char c[32];
                    std::snprintf(c, sizeof c, " %7.4f", sum[b] / (double) cnt[b]);
                    line += c;
                }
            std::fprintf(stderr, "drift %-22s layer %2d:%s\n", name, m.kv_layer(s), line.c_str());
        }
    };
    std::fprintf(stderr, "drift: edit [%lld, %lld) -> %lld junk tokens, followers from %lld to %lld, mark %lld\n", (long long) a,
                 (long long) (a + D), (long long) N, (long long) f0, (long long) E2, (long long) mk);
    std::fprintf(stderr, "drift: relative row error by distance from the edit, buckets from 0 16 64 128 256 512 1K 2K 4K 8K\n");
    m.reset();
    if (!m.feed(edited.data(), (int) E2, 1, err) || !grab(ref)) {
        std::fprintf(stderr, "strata-dsv41: the reference: %s\n", err.c_str());
        return 1;
    }
    // the floor: the same edited tokens, read in the pieces a splice uses
    m.reset();
    if (!m.feed(edited.data(), (int) mk, 1, err)) return 1;
    m.mark();
    if (!m.feed(edited.data() + mk, (int) (E2 - mk), 1, err) || !grab(var)) return 1;
    show("noise (same tokens)");
    if (N == D) {   // the edit ignored (the followers at the same rows)
        m.reset();
        if (!m.feed(ids.data(), (int) E, 1, err) || !grab(var)) return 1;
        show("edit ignored");
    }
    for (int x : st_list("DS4_ST_SEAMS", "0,128")) {
        // (the unedited state with a mark every 2048, as serve keeps them)
        m.reset();
        for (int64_t q = 0; q < E; q += 2048) {
            if (!m.feed(ids.data() + q, (int) std::min<int64_t>(2048, E - q), 1, err)) return 1;
            m.mark();
        }
        const double t1 = now_ms();
        if (!m.splice_edits(eds, x, err) || !grab(var)) {
            std::fprintf(stderr, "strata-dsv41: the splice: %s\n", err.c_str());
            return 1;
        }
        const std::vector<int32_t>& st = m.tokens();
        const bool same = st.size() == edited.size() && std::equal(st.begin(), st.end(), edited.begin());
        char nm[48];
        std::snprintf(nm, sizeof nm, "spliced, seam %d (%.1f s%s)", x, (now_ms() - t1) / 1000.0, same ? "" : ", TOKENS DIFFER");
        show(nm);
    }
    return 0;
}

// DS4_ST_STRESS=trials: random multi-edits (every kind) on one base state
// of MAIN[0, E); each checked by its tokens and by copy probes of the
// edited spans and of unedited ones (the control)
static int splice_stress_test(strata::ds4::Model& m, const std::vector<int>& ids, const std::vector<int>& junk) {
    std::string err;
    const auto num = [](const char* k, long long d) { return std::getenv(k) ? std::atoll(std::getenv(k)) : d; };
    const int64_t al = m.splice_align(), W = m.splice_window();
    const int64_t E = num("DS4_ST_EDIT_E", 16384), trials = num("DS4_ST_STRESS", 12), seam = num("DS4_SPLICE_SEAM", 2048);
    const int64_t maxk = num("DS4_ST_MAXK", 5), joff = num("DS4_ST_JUNK_OFF", 100000);
    if (E + 1 > (int64_t) ids.size() || joff + 4096 > (int64_t) junk.size()) {
        std::fprintf(stderr, "strata-dsv41: the tokens are too short for the stress test\n");
        return 1;
    }
    const double t0 = now_ms();
    // (the base: marks every DS4_ST_BASE_STEP tokens, default one read)
    const int64_t bstep = num("DS4_ST_BASE_STEP", E);
    for (int64_t q = 0; q < E; q += bstep) {
        if (!m.feed(ids.data() + q, (int) std::min<int64_t>(bstep, E - q), 1, err)) {
            std::fprintf(stderr, "strata-dsv41: the base: %s\n", err.c_str());
            return 1;
        }
        m.mark();
        std::fprintf(stderr, "stress: base %lld of %lld tokens (%.0f s)\n", (long long) std::min(q + bstep, E), (long long) E,
                     (now_ms() - t0) / 1000.0);
    }
    std::vector<uint8_t> base;
    if (m.slot_save(base, err) == 0) {
        std::fprintf(stderr, "strata-dsv41: saving the base: %s\n", err.c_str());
        return 1;
    }
    std::mt19937_64 rng((uint64_t) num("DS4_ST_SEED", 1));
    const auto rnd = [&](int64_t lo, int64_t hi) { return lo + (int64_t) (rng() % (uint64_t) (hi - lo + 1)); };
    // copy probe at s: the 8 tokens there fed, the next 24 scored
    const auto count8 = [](const std::vector<int>& v, size_t s) {
        int c = 0;
        for (size_t i = 0; i + 8 <= v.size(); ++i)
            if (std::equal(v.begin() + (long) s, v.begin() + (long) s + 8, v.begin() + (long) i)) ++c;
        return c;
    };
    const int V = m.vocab();
    const auto probe = [&](const std::vector<int>& v, size_t s, int64_t E2, int& hit) {
        std::vector<int> c(v.begin() + (long) s, v.begin() + (long) s + 32);
        std::vector<float> lp;
        if (!st_score(m, c, 32, lp, err)) return false;
        hit = 0;
        for (int t = 7; t < 31; ++t) {
            const float* r = lp.data() + (size_t) t * (size_t) V;
            hit += (int) (std::max_element(r, r + V) - r) == c[(size_t) t + 1];
        }
        return m.restore(E2) && m.pos() == E2;
    };
    int64_t bad_tok = 0, n_ed = 0, n_ct = 0, hit_ed = 0, hit_ct = 0, fails = 0;
    double t_sum = 0.0;
    for (int64_t tr = 0; tr < trials; ++tr) {
        if (!m.slot_load(base.data(), base.size(), E, err)) {
            std::fprintf(stderr, "strata-dsv41: loading the base: %s\n", err.c_str());
            return 1;
        }
        // edits: sorted, apart, every kind; a and d on the grid
        const int64_t K = rnd(1, maxk);
        std::vector<int64_t> at;
        for (int64_t i = 0; i < K; ++i) at.push_back(rnd(2, E - 1024) / al * al);
        std::sort(at.begin(), at.end());
        std::vector<strata::ds4::Model::Edit> eds;
        std::string kinds;
        int64_t jp = joff + rnd(0, 20000);
        for (int64_t i = 0; i < K; ++i) {
            const int kind = (int) rnd(0, 5);
            int64_t d = rnd(1, 128) * al, n = d;
            const char* kn = "same";
            if (kind == 1) n = d + 1, kn = "grow+1";
            if (kind == 2) n = d + rnd(16, 300), kn = "grow";
            if (kind == 3) n = std::max<int64_t>(1, d - rnd(1, d)), kn = "shrink";
            if (kind == 4) n = 0, kn = "delete";
            if (kind == 5) d = 0, n = rnd(16, 300), kn = "insert";
            const int64_t a = at[(size_t) i];
            if (!eds.empty() && a < eds.back().a + eds.back().d + 2 * al) continue;   // (overlaps the last)
            if (a + d + W + 256 > E) continue;
            strata::ds4::Model::Edit ed;
            ed.a = a;
            ed.d = d;
            ed.rep.assign(junk.begin() + jp, junk.begin() + jp + n);
            jp += n + 7;
            eds.push_back(std::move(ed));
            char b[48];
            std::snprintf(b, sizeof b, "%s%s@%lld(%lld->%lld)", kinds.empty() ? "" : " ", kn, (long long) a, (long long) d,
                          (long long) n);
            kinds += b;
        }
        if (eds.empty()) continue;
        std::vector<int> edited;   // (the prompt, and where each edit landed)
        std::vector<std::pair<size_t, size_t>> spans;
        int64_t cur = 0;
        for (const auto& ed : eds) {
            edited.insert(edited.end(), ids.begin() + cur, ids.begin() + ed.a);
            spans.push_back({edited.size(), ed.rep.size()});
            edited.insert(edited.end(), ed.rep.begin(), ed.rep.end());
            cur = ed.a + ed.d;
        }
        edited.insert(edited.end(), ids.begin() + cur, ids.begin() + E);
        const int64_t E2 = (int64_t) edited.size();
        const double t1 = now_ms();
        if (!m.splice_edits(eds, seam, err)) {
            ++fails;
            std::fprintf(stderr, "stress %lld: [%s] SPLICE FAILED: %s\n", (long long) tr, kinds.c_str(), err.c_str());
            continue;
        }
        const double ts = (now_ms() - t1) / 1000.0;
        t_sum += ts;
        const std::vector<int32_t>& st = m.tokens();
        const bool same = (int64_t) st.size() == E2 && m.pos() == E2 && std::equal(st.begin(), st.end(), edited.begin());
        bad_tok += !same;
        m.mark();
        std::string pe, pc;
        for (const auto& sp : spans) {   // (edited spans of 32+ tokens)
            if (sp.second < 32 || count8(edited, sp.first) != 1) continue;
            int h = 0;
            if (!probe(edited, sp.first, E2, h)) {
                std::fprintf(stderr, "strata-dsv41: a probe: %s\n", err.c_str());
                return 1;
            }
            ++n_ed;
            hit_ed += h;
            pe += " " + std::to_string(h);
        }
        for (int c = 0; c < 3; ++c) {   // (unedited, after the first edit)
            const int64_t lo = (int64_t) spans[0].first + (int64_t) spans[0].second + 64;
            if (lo + 64 >= E2) break;
            const size_t s = (size_t) rnd(lo, E2 - 64);
            bool clear = count8(edited, s) == 1;
            for (const auto& sp : spans) clear = clear && (s + 32 <= sp.first || s >= sp.first + sp.second);
            int h = 0;
            if (!clear) continue;
            if (!probe(edited, s, E2, h)) {
                std::fprintf(stderr, "strata-dsv41: a probe: %s\n", err.c_str());
                return 1;
            }
            ++n_ct;
            hit_ct += h;
            pc += " " + std::to_string(h);
        }
        std::fprintf(stderr, "stress %lld: [%s] %.1f s, tokens %s | edited copies (of 24):%s | controls:%s\n", (long long) tr,
                     kinds.c_str(), ts, same ? "ok" : "DIFFER", pe.empty() ? " -" : pe.c_str(), pc.empty() ? " -" : pc.c_str());
    }
    std::fprintf(stderr, "stress: %lld trials, %lld failed, %lld with wrong tokens; copy accuracy edited %.1f%% (%lld probes), "
                         "controls %.1f%% (%lld); %.1f s a splice\n",
                 (long long) trials, (long long) fails, (long long) bad_tok, n_ed ? 100.0 * hit_ed / (24.0 * n_ed) : 0.0,
                 (long long) n_ed, n_ct ? 100.0 * hit_ct / (24.0 * n_ct) : 0.0, (long long) n_ct,
                 t_sum / std::max<int64_t>(1, trials - fails));
    return fails || bad_tok ? 2 : 0;
}

// DS4_ST_PPL=n: the meter check - MAIN[0, E) read, then MAIN[E, E + n)
// scored by feed windows of 16 and by step() one token at a time, on
// the same state; their perplexities, KL and top-1 agreement
static int splice_ppl_check(strata::ds4::Model& m, const std::vector<int>& ids) {
    std::string err;
    const auto num = [](const char* k, long long d) { return std::getenv(k) ? std::atoll(std::getenv(k)) : d; };
    const int64_t E = num("DS4_ST_EDIT_E", 12288), n = num("DS4_ST_PPL", 256);
    if (E + n + 1 > (int64_t) ids.size()) {
        std::fprintf(stderr, "strata-dsv41: the tokens are too short for the meter check\n");
        return 1;
    }
    const double t0 = now_ms();
    // (DS4_ST_PPL_FROM=s: the history is BOS then MAIN[s, E) only)
    const int64_t from = std::max<long long>(0, num("DS4_ST_PPL_FROM", 0));
    std::vector<int> hist;
    if (from > 0) hist.push_back(ids[0]);
    hist.insert(hist.end(), ids.begin() + from, ids.begin() + E);
    if (!m.feed(hist.data(), (int) hist.size(), 1, err)) {
        std::fprintf(stderr, "strata-dsv41: the read: %s\n", err.c_str());
        return 1;
    }
    const int64_t Eh = (int64_t) hist.size();
    m.mark();
    std::fprintf(stderr, "ppl: read %lld tokens of history (%.0f s)\n", (long long) Eh, (now_ms() - t0) / 1000.0);
    const int V = m.vocab();
    // (row t of each: log-softmax predicting ids[E + t + 1])
    std::vector<int> c(ids.begin() + E, ids.begin() + E + n + 1);
    std::vector<float> win;
    if (!st_score(m, c, (int) n + 1, win, err)) {
        std::fprintf(stderr, "strata-dsv41: window scoring: %s\n", err.c_str());
        return 1;
    }
    if (!m.restore(Eh) || m.pos() != Eh) {
        std::fprintf(stderr, "strata-dsv41: no way back to %lld\n", (long long) Eh);
        return 1;
    }
    std::vector<float> stp((size_t) (n + 1) * (size_t) V), lg;
    int64_t same_pick = 0;   // (greedy sample() = the host row's argmax)
    std::vector<int> picks;
    for (int64_t t = 0; t <= n; ++t) {
        if (!m.step(c[(size_t) t], true, err) || !m.logits(lg) || lg.size() < (size_t) V) {
            std::fprintf(stderr, "strata-dsv41: step %lld: %s\n", (long long) t, err.c_str());
            return 1;
        }
        const float* r = lg.data() + (lg.size() - (size_t) V);   // (the last row)
        const int g = m.sample(0.0f, 1.0f, 0, 0.0f, 1, (uint64_t) t);
        picks.push_back(g);
        same_pick += g == (int) (std::max_element(r, r + V) - r);
        if (t == 0) std::fprintf(stderr, "ppl: logits() gives %zu floats a call (vocab %d)\n", lg.size(), V);
        float mx = r[0];
        for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
        double z = 0.0;
        for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
        const float lz = (float) (std::log(z) + (double) mx);
        for (int v = 0; v < V; ++v) stp[(size_t) t * (size_t) V + (size_t) v] = r[v] - lz;
    }
    // per quarter: both perplexities, KL(step || windows), agreement
    for (int qtr = 0; qtr < 4; ++qtr) {
        const int64_t a = n * qtr / 4, b = n * (qtr + 1) / 4;
        double nw = 0, ns = 0, kl = 0;
        int agree = 0;
        for (int64_t t = a; t < b; ++t) {
            const float* p = stp.data() + (size_t) t * (size_t) V;
            const float* q = win.data() + (size_t) t * (size_t) V;
            int ap = 0, aq = 0;
            for (int v = 0; v < V; ++v) {
                const double pv = std::exp((double) p[v]);
                if (pv > 1e-12) kl += pv * ((double) p[v] - (double) q[v]);
                if (p[v] > p[ap]) ap = v;
                if (q[v] > q[aq]) aq = v;
            }
            agree += ap == aq;
            ns -= p[c[(size_t) t + 1]];
            nw -= q[c[(size_t) t + 1]];
        }
        const double k = (double) std::max<int64_t>(1, b - a);
        std::fprintf(stderr, "ppl tokens %4lld-%4lld: step() %.3f | windows of 16 %.3f | KL %.4f | top-1 agrees %.1f%%\n",
                     (long long) a, (long long) b, std::exp(ns / k), std::exp(nw / k), kl / k, 100.0 * agree / k);
    }
    // a row off by one: each step() row against the token before and
    // after the one it should predict
    double nb = 0, na = 0;
    int64_t cnt = 0, pick_hits = 0;
    for (int64_t t = 1; t + 2 <= n; ++t, ++cnt) {
        const float* p = stp.data() + (size_t) t * (size_t) V;
        nb -= p[c[(size_t) t]];
        na -= p[c[(size_t) t + 2]];
    }
    for (int64_t t = 0; t < n; ++t) pick_hits += picks[(size_t) t] == c[(size_t) t + 1];
    std::fprintf(stderr, "ppl step() rows against the token behind %.3f, ahead %.3f; greedy sample() = logits() argmax "
                         "%lld of %lld; greedy sample() = the text's next token %.1f%%\n",
                 std::exp(nb / (double) std::max<int64_t>(1, cnt)), std::exp(na / (double) std::max<int64_t>(1, cnt)),
                 (long long) same_pick, (long long) (n + 1), 100.0 * (double) pick_hits / (double) n);
    return 0;
}

int splice_test(strata::ds4::Model& m, const std::string& main_meta, const std::string& junk_meta) {
    std::vector<int> ids, junk;
    std::string err;
    if (!st_meta_tokens(main_meta, ids, err) || !st_meta_tokens(junk_meta, junk, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    if (std::getenv("DS4_ST_PPL")) return splice_ppl_check(m, ids);
    if (std::getenv("DS4_ST_STRESS")) return splice_stress_test(m, ids, junk);
    if (std::getenv("DS4_ST_DRIFT")) return splice_drift_test(m, ids, junk);
    if (std::getenv("DS4_ST_EDIT_A")) return splice_edit_test(m, ids, junk);
    const int64_t al = m.splice_align(), W = m.splice_window();
    const int64_t a1 = std::getenv("DS4_ST_A1") ? std::atoll(std::getenv("DS4_ST_A1")) / al * al : 8192;
    const int64_t B = std::getenv("DS4_ST_B") ? std::atoll(std::getenv("DS4_ST_B")) : 4096;
    const int nc = std::getenv("DS4_ST_NC") ? std::atoi(std::getenv("DS4_ST_NC")) : 64;
    const int64_t joff = std::getenv("DS4_ST_JUNK_OFF") ? std::atoll(std::getenv("DS4_ST_JUNK_OFF")) : 0;
    const int64_t E = a1 + B;
    if (E + nc + 1 > (int64_t) ids.size()) {
        std::fprintf(stderr, "strata-dsv41: %s has %zu tokens, the test needs %lld\n", main_meta.c_str(), ids.size(), (long long) (E + nc + 1));
        return 1;
    }
    const std::vector<int> cont(ids.begin() + E, ids.begin() + E + nc + 1);
    const int V = m.vocab();
    std::fprintf(stderr, "splice-test: X = [0, %lld), B = [%lld, %lld), %d scored after it; grid %lld, window %lld\n",
                 (long long) a1, (long long) a1, (long long) E, nc, (long long) al, (long long) W);
    auto feed = [&](const int* p, int64_t n) {
        if (n <= 0) return true;
        if (!m.feed(p, (int) n, 1, err)) {
            std::fprintf(stderr, "strata-dsv41: feed: %s\n", err.c_str());
            return false;
        }
        return true;
    };
    std::vector<float> ref, var;
    const double t0 = now_ms();
    const std::vector<int64_t> seams = [&] {
        std::vector<int64_t> v;
        for (int x : st_list("DS4_ST_SEAMS", "0,128,256")) v.push_back(std::min<int64_t>(x, B - W));
        return v;
    }();
    std::vector<int64_t> junks;
    for (int x : st_list("DS4_ST_JUNK", "512,4096")) junks.push_back(std::min<int64_t>(x / al * al, (int64_t) junk.size() - joff));
    // the foreign states first: B (less its seam) read after junk
    // alone, no X before it - the block a library would hold
    std::vector<std::vector<uint8_t>> lib;
    for (int64_t L : junks)
        for (int64_t S : seams) {
            m.reset();
            lib.emplace_back();
            if (!feed(junk.data() + joff, L) || !feed(ids.data() + a1, B - S) || m.slot_save(lib.back(), err) == 0) {
                std::fprintf(stderr, "strata-dsv41: the foreign state: %s\n", err.c_str());
                return 1;
            }
        }
    std::fprintf(stderr, "splice-test: %zu foreign states (%.0f s)\n", lib.size(), (now_ms() - t0) / 1000.0);
    // X once; every run below goes back to its mark
    m.reset();
    if (!feed(ids.data(), a1)) return 1;
    m.mark();
    const auto back = [&] {
        if (!m.restore(a1) || m.pos() != a1) {
            std::fprintf(stderr, "strata-dsv41: X's mark is gone\n");
            return false;
        }
        return true;
    };
    if (!feed(ids.data() + a1, B) || !st_score(m, cont, nc, ref, err)) return 1;
    std::fprintf(stderr, "splice-test: reference read (%.0f s)\n", (now_ms() - t0) / 1000.0);
    char nm[64];
    size_t li = 0;
    for (size_t ji = 0; ji < junks.size(); ++ji)
        for (int64_t S : seams) {
            const int64_t L = junks[ji];
            if (ji == 0) {   // the control: the same pieces, no junk
                if (!back() || !feed(ids.data() + a1, B - S) || !feed(ids.data() + E - S, S) || !st_score(m, cont, nc, var, err))
                    return 1;
                std::snprintf(nm, sizeof nm, "control, seam %lld", (long long) S);
                st_report(nm, ref, var, cont, nc, V);
            }
            // stale: B read after X and junk, the junk cut out again
            if (!back() || !feed(junk.data() + joff, L) || !feed(ids.data() + a1, B - S)) return 1;
            if (!m.splice_out(a1, L, err)) {
                std::fprintf(stderr, "strata-dsv41: splice_out: %s\n", err.c_str());
                return 1;
            }
            if (!feed(ids.data() + E - S, S) || !st_score(m, cont, nc, var, err)) return 1;
            std::snprintf(nm, sizeof nm, "cut junk %lld, seam %lld", (long long) L, (long long) S);
            st_report(nm, ref, var, cont, nc, V);
            // foreign: B's rows from the state where junk alone came first
            const std::vector<uint8_t>& sl = lib[li++];
            if (!back() || !m.splice_in(sl.data(), sl.size(), L, B - S, err)) {
                std::fprintf(stderr, "strata-dsv41: splice_in: %s\n", err.c_str());
                return 1;
            }
            if (!feed(ids.data() + E - S, S) || !st_score(m, cont, nc, var, err)) return 1;
            std::snprintf(nm, sizeof nm, "foreign %lld, seam %lld", (long long) L, (long long) S);
            st_report(nm, ref, var, cont, nc, V);
        }
    std::fprintf(stderr, "splice-test: done (%.0f s)\n", (now_ms() - t0) / 1000.0);
    return 0;
}

// ---- Strata H100 addition: profile run -------------------------------------------------------------------------
// --profile-run IDS.json: every segment is fed through the model from a fresh state (no logits); the engine's expert
// profile (--expert-profile FILE, counts a layer and expert) and DS4_TRACE collect the routing.  The profile is saved
// every 10 segments.  --window sets the chunk (a chunked prompt path above 16).
int profile_run(strata::ds4::Model& m, const std::string& ids_path, int window) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    const double t0 = now_ms();
    long long tot = 0;
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            if (!m.feed(ids.data() + a, T, 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        tot += (long long) ids.size();
        std::fprintf(stderr, "profile-run %zu/%zu %s: %zu tokens, %lld in all, %.0f s, %.1f tok/s\n", si + 1, segs.size(),
                     segs[si].name.c_str(), ids.size(), tot, (now_ms() - t0) / 1000.0, 1000.0 * (double) tot / (now_ms() - t0));
        if ((si + 1) % 10 == 0) m.save_profile();
    }
    m.save_profile();
    return 0;
}
// ---- Strata H100 addition: DSpark acceptance log ---------------------------------------------------------------
// --spec-eval IDS.json --n N: each segment is a prompt: fed from reset, then N greedy tokens decoded with the drafter, every
// draft verified (--dspark-conf 0); DS4_ROUND_LOG=file gets a line a round (segment name, drafts, accepted, confidences).
int spec_eval(strata::ds4::Model& m, const std::string& ids_path, int n, int window, float conf_min, bool spec) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    if (const char* f = std::getenv("DS4_ROUND_LOG")) g_round_log = std::fopen(f, "wb");
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        g_round_tag = segs[si].name;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            const bool last = a + (size_t) T == ids.size();
            if (!m.feed(ids.data() + a, T, last ? 1 : 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        unsigned long long counter = 0;
        int tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, counter++);
        std::vector<int> got;
        int emitted = 1;
        long long rounds = 0, acc_all = 0;
        bool stop = false;
        while (emitted < n && !stop) {
            int off = 0, acc = 0;
            if (!round(m, tok, n - emitted, spec, conf_min, 0.0f, 1.0f, 1, 0.0f, 1, counter, got, off, acc, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            acc_all += acc;
            for (int t : got) {
                ++emitted;
                if (t == 1) stop = true;
            }
            tok = got.back();
        }
        std::fprintf(stderr, "spec-eval %zu/%zu %s: prompt %zu, %d tokens, %lld rounds, %.2f tokens a round, accepted %lld (%.0f s)\n", si + 1,
                     segs.size(), segs[si].name.c_str(), ids.size(), emitted, rounds, (double) (emitted - 1) / (double) std::max(1LL, rounds),
                     acc_all, (now_ms() - t0) / 1000.0);
        if (g_round_log != nullptr) std::fflush(g_round_log);
    }
    return 0;
}
}  // namespace

#ifndef DS4_BUILD_ID
#define DS4_BUILD_ID "unknown"
#endif

#if defined(STRATA_METAL)
#include "sampler_stat.inc"
#endif
int main(int argc, char** argv) {
    strata::ds4::os::fine_timer();
#if defined(STRATA_METAL)
    if (argc > 1 && std::string(argv[1]) == "--sampler-test") return sampler_stat::run(argc - 1, argv + 1);   // (the sampler's statistics harness)
#endif
#if defined(__APPLE__)
    if (std::getenv("DS4_BT")) {   // a crash prints where it was (no debugger on the Mac box)
        signal(SIGSEGV, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(139); });
        signal(SIGBUS, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(138); });
        signal(SIGABRT, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(134); });
    }
#endif
    strata::ds4::Options o;
    std::string ids_s, dump, eos = "1";
    int n = 0, window = 0;   // (0: the model's feed unit - a whole prompt on the chunked path; 16 for the scoring runs)
    bool spec = true;
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false;
    bool pack_info_mode = false;
    std::string images, enc_rgb, enc_out;
    std::string score_ids, score_out, prof_ids, spec_ids, st_main, st_junk;
    int enc_w = 0, enc_h = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val("--pack");
        else if (a == "--gpus") {
            for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
        } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
        else if (a == "--vram-experts") { o.vram_experts = std::atof(val("--vram-experts").c_str()); o.vram_experts_auto = false; }
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--expert-profile") o.profile = val("--expert-profile");
        else if (a == "--profile-readonly") o.profile_readonly = true;   // (read the placement, never write the counts back)
        else if (a == "--ssd-cache") o.ssd_cache_gb = std::atof(val("--ssd-cache").c_str());
        else if (a == "--ssd-policy") {
            const std::string v = val("--ssd-policy");
            o.ssd_policy = v == "lfu" ? 1 : v == "lfulru" ? 2 : 0;
        }
        else if (a == "--eos") eos = val("--eos");
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--dspark") {
            const std::string v = val("--dspark");
            spec = v != "off";
            o.dspark = v != "none";   // none: not even loaded
            o.dspark_auto = v == "auto";
        } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
        else if (a == "--experts") {
            const std::string v = val("--experts");
            o.experts = v == "auto" ? -1 : std::max(1, std::atoi(v.c_str()));
        }
        else if (a == "--dspark-block") o.dspark_block = std::max(0, std::atoi(val("--dspark-block").c_str()));
        else if (a == "--dspark-depth") g_spec_depth = std::max(1, std::atoi(val("--dspark-depth").c_str()));
        else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
        else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
        else if (a == "--threads") o.threads = val("--threads") != "off";
        else if (a == "--vision") o.vision = val("--vision") != "off";
        else if (a == "--mmap-experts") {
            const std::string v = val("--mmap-experts");
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        } else if (a == "--images") images = val("--images");
        else if (a == "--encode-image") enc_rgb = val("--encode-image");
        else if (a == "--dump-image") enc_out = val("--dump-image");
        else if (a == "--size") {
            const std::string v = val("--size");
            enc_w = std::atoi(v.c_str());
            enc_h = v.find('x') != std::string::npos ? std::atoi(v.c_str() + v.find('x') + 1) : 0;
        }
        else if (a == "--temperature") temp = (float) std::atof(val("--temperature").c_str());
        else if (a == "--seed") cli_seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
        else if (a == "--verbose" || a == "-v") {
            o.verbose = true;
            std::fprintf(stderr, "ds4: build %s (%s)\n", DS4_BUILD_ID, STRATA_VERSION);
        } else if (a == "--version") {
            std::printf("strata-dsv41 %s build %s\n", STRATA_VERSION, DS4_BUILD_ID);
            return 0;
        }
        else if (a == "--ids") ids_s = val("--ids");
        else if (a == "--n") n = std::atoi(val("--n").c_str());
        else if (a == "--window") window = std::max(1, std::atoi(val("--window").c_str()));   // (> 16: the chunked prompt path)
        else if (a == "--dump-logits") dump = val("--dump-logits");
        else if (a == "--score") score_ids = val("--score");
        else if (a == "--score-out") score_out = val("--score-out");
        else if (a == "--profile-run") prof_ids = val("--profile-run");
        else if (a == "--spec-eval") spec_ids = val("--spec-eval");
        else if (a == "--splice-test") st_main = val("--splice-test");
        else if (a == "--splice-junk") st_junk = val("--splice-junk");
        else if (a == "--serve") serve_mode = true;
        else if (a == "--pack-info") pack_info_mode = true;
        else if (a == "--layer-split") (void) val("--layer-split");   // the server's (this engine balances its own split)
        else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (pack_info_mode) {   // the pack's description only: no device, no weights, no lock
        if (o.pack.empty()) {
            usage();
            return 2;
        }
        return strata::ds4::pack_info(o.pack, stdout);
    }
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty() && score_ids.empty() && prof_ids.empty() && spec_ids.empty() &&
                           st_main.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    if (const char* idle = std::getenv("DS4_TEST_IDLE_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(idle)));   // (tests: a person typing)
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack);
    const int window_given = window;
    if (window == 0) window = 16;   // (the scoring / profiling runs; the plain run below takes the model's feed unit)
    if (!prof_ids.empty()) return profile_run(m, prof_ids, window);
    if (!spec_ids.empty()) return spec_eval(m, spec_ids, n, window, o.dspark_conf, spec);
    if (!score_ids.empty()) return score(m, score_ids, score_out.empty() ? score_ids + ".tsv" : score_out, window, std::atoi(eos.c_str()));
    if (!st_main.empty()) return splice_test(m, st_main, st_junk.empty() ? st_main : st_junk);

    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's span rows (BF16 [rows][dim]) into --dump-image
        std::vector<uint8_t> px((size_t) std::max(0, enc_w) * std::max(0, enc_h) * 3);
        std::FILE* f = std::fopen(enc_rgb.c_str(), "rb");
        const bool got = f != nullptr && !px.empty() && std::fread(px.data(), 1, px.size(), f) == px.size();
        if (f != nullptr) std::fclose(f);
        if (!got) {
            std::fprintf(stderr, "maya-ds4: %s: expected %d x %d RGB bytes (--size WxH)\n", enc_rgb.c_str(), enc_w, enc_h);
            return 1;
        }
        std::vector<uint16_t> span;
        for (int k = 0; k < 2; ++k) {   // (the second: warm)
            const double te = now_ms();
            if (!m.encode_image(px.data(), enc_w, enc_h, span, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "vision: %d x %d pixels -> a span of %zu rows in %.0f ms\n", enc_w, enc_h,
                         span.size() / (size_t) m.dim(), now_ms() - te);
        }
        std::FILE* o2 = enc_out.empty() ? nullptr : std::fopen(enc_out.c_str(), "wb");
        if (o2 != nullptr) {
            std::fwrite(span.data(), 2, span.size(), o2);
            std::fclose(o2);
        }
        return 0;
    }
    if (!ids_s.empty() && ids_s[0] == '@') {   // --ids @file: the ids in a file (a command line is 32 KB on Windows)
        std::ifstream idf(ids_s.substr(1));
        std::stringstream ss;
        ss << idf.rdbuf();
        ids_s = ss.str();
    }
    const std::vector<int> ids = parse_ids(ids_s);
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        if (!read_pics(images, pics, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::vector<strata::ds4::Image> imgs;
        for (const Pic& q : pics) {
            auto rows = std::make_shared<std::vector<uint16_t>>();
            const double te = now_ms();
            if (!m.encode_image(q.rgb.data(), q.width, q.height, *rows, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "picture at %lld: %d x %d pixels, %zu tokens, %.0f ms\n", (long long) q.start, q.width,
                         q.height, rows->size() / (size_t) m.dim(), now_ms() - te);
            imgs.push_back({q.start, rows});
        }
        m.set_images(std::move(imgs));
    }
    std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    std::vector<float> lg;
    if (window_given == 0) window = df != nullptr ? 16 : std::max(16, std::min(m.feed_unit(), 1 << 30));
    const double t0 = now_ms();
    for (size_t a = 0; a < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
        const bool last = a + (size_t) T == ids.size();
        const double tf0 = now_ms();
        if (!m.feed(ids.data() + a, T, df != nullptr ? T : (last ? 1 : 0), err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        if (std::getenv("DS4_FEEDLOG") != nullptr)   // each feed's time: the incremental turn (--window W: a long prefix, then a short one)
            std::fprintf(stderr, "feed at %zu: %d tokens in %.0f ms (%.1f tok/s)\n", a, T, now_ms() - tf0, 1000.0 * T / (now_ms() - tf0));
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        a += (size_t) T;
    }
    m.set_images({});
    const double t1 = now_ms();
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)\n", ids.size(), t1 - t0,
                 1000.0 * (double) ids.size() / (t1 - t0), window);
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SEQ_LOG")) std::fprintf(stderr, "dispatch seq after the prompt: %llu\n", ds4_dispatch_seq());
#endif
    const strata::ds4::Stats s0 = m.stats();
    if (s0.eg_rows > 0)
        std::fprintf(stderr, "prompt's Engram rows: %lld, %.1f%% without a read (repeats and the row cache), %lld read\n",
                     (long long) s0.eg_rows, 100.0 * (double) s0.eg_hits / (double) s0.eg_rows, (long long) s0.eg_reads);
    const long long io0 = io_read_bytes();
    std::vector<int> out;
    std::vector<double> tok_ms;   // (DS4_TOKLOG: each token's time since the one before - the latency ledger)
    double t_prev = t1;
    unsigned long long counter = 0;
    long long d_off = 0, d_acc = 0, rounds = 0;
    bool cool = false;
    if (n > 0) {
        int tok = m.sample(temp, 1.0f, temp > 0 ? 64 : 1, 0.0f, cli_seed, counter++);
        std::vector<int> got;
        for (;;) {
            out.push_back(tok);
            {
                const double tn = now_ms();
                tok_ms.push_back(tn - t_prev);
                t_prev = tn;
            }
            std::printf("%d,", tok);
            std::fflush(stdout);
            if (ds4_dbg("DS4_PROGRESS") && out.size() % 200 == 0) {   // (sustained-run time series)
                static double tp = now_ms();
                static size_t np = 0;
                const double tn = now_ms();
                std::fprintf(stderr, "progress %zu tokens: %.2f tok/s over the last %zu\n", out.size(), 1000.0 * (double) (out.size() - np) / (tn - tp), out.size() - np);
                tp = tn;
                np = out.size();
            }
            if ((tok == 1 && std::getenv("DS4_NO_EOS") == nullptr) || (int) out.size() >= n) break;   // (DS4_NO_EOS: decode n tokens whatever)
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool;
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err)) {
                    std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                    return 1;
                }
                d_off += off;
                d_acc += acc;
                ++rounds;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        std::printf("\n");
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
    }
    if (spec && rounds > 0) {
        const auto& s1 = m.stats();
        const double nv = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
        std::fprintf(stderr, "dspark: %lld rounds, %lld drafts offered, %lld accepted (%.2f tokens a round); drafting "
                             "%.2f ms a round, verifying %.1f ms in all (%lld windows)\n",
                     rounds, d_off, d_acc, (double) out.size() / (double) rounds, (s1.draft_ms - s0.draft_ms) / nv,
                     s1.verify_ms - s0.verify_ms, (long long) (s1.drafts_offered > 0 ? 0 : 0));
        std::fprintf(stderr, "dspark %s\n", per_position(s0, s1).c_str());
    }
    if (std::getenv("DS4_TOKLOG") != nullptr && tok_ms.size() > 8) {   // the decode's tokens after the first (which carries the prompt's tail)
        std::vector<double> v(tok_ms.begin() + 1, tok_ms.end());
        std::sort(v.begin(), v.end());
        std::fprintf(stderr, "token latency ms: p50 %.0f p95 %.0f p99 %.0f max %.0f mean %.0f (%zu tokens)\n", v[v.size() / 2],
                     v[(size_t) ((double) v.size() * 0.95)], v[(size_t) ((double) v.size() * 0.99)], v.back(),
                     std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size(), v.size());
    }
    const double t2 = now_ms();
    const auto& s = m.stats();
    const double nt = std::max<double>(1.0, (double) (s.tokens - s0.tokens));
    std::fprintf(stderr, "decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts; the cache moved %lld "
                         "(%lld in all)\n", out.size(), t2 - t1,
                 out.size() > 1 ? 1000.0 * (double) (out.size() - 1) / (t2 - t1) : 0.0,
                 (long long) (s.cpu_experts - s0.cpu_experts), (long long) (s.experts - s0.experts),
                 (long long) (s.swaps - s0.swaps), (long long) s.swaps);
    std::fprintf(stderr, "decode per token: layers %.2f ms (of it: routing waits %.2f, CPU lane %.2f, the rest host launches), "
                         "head wait %.2f, Engram reads %.2f\n",
                 (s.layer_ms - s0.layer_ms) / nt, (s.gpu_wait_ms - s0.gpu_wait_ms) / nt, (s.cpu_ms - s0.cpu_ms) / nt,
                 (s.head_ms - s0.head_ms) / nt, (s.engram_ms - s0.engram_ms) / nt);
    if (s.eg_rows > s0.eg_rows) {   // the Engram rows: how many the row cache served, the reads' time behind the layers
        const double rows = (double) (s.eg_rows - s0.eg_rows);
        std::fprintf(stderr, "Engram per token: %.0f rows, %.1f%% without a read (the row cache holds %lld of %lld), "
                             "%.1f read, the reads %.2f ms from asking to the last one (left to wait: %.2f)\n",
                     rows / nt, 100.0 * (double) (s.eg_hits - s0.eg_hits) / rows, (long long) s.eg_cached,
                     (long long) s.eg_cache_cap, (double) (s.eg_reads - s0.eg_reads) / nt, (s.eg_io_ms - s0.eg_io_ms) / nt,
                     (s.engram_ms - s0.engram_ms) / nt);
    }
    const long long io1 = io_read_bytes();
    if (io0 >= 0 && io1 >= 0)
        std::fprintf(stderr, "decode reads from the disk: %.1f MB a token\n", (double) (io1 - io0) / 1e6 / nt);
    if (s.skipped > s0.skipped)   // STRATA_DS4_DISK_SKIP_W
        std::fprintf(stderr, "decode left out %.1f experts a token not in RAM (%.1f%% of the routed)\n",
                     (double) (s.skipped - s0.skipped) / nt,
                     100.0 * (double) (s.skipped - s0.skipped) / std::max<double>(1.0, (double) (s.experts - s0.experts)));
    if (s.store_slots_max > 0) {   // --ssd-cache
        const double h = (double) (s.store_hits - s0.store_hits), m = (double) (s.store_misses - s0.store_misses);
        std::fprintf(stderr,
                     "store: %.1f%% of the CPU lane's experts were in RAM (%.1f hits, %.1f misses a token); read %.1f MB a token "
                     "(demand and guesses), the CPU lane waited %.1f ms a token for them; %lld prefetched, %lld used, %lld given up "
                     "unused; slots %lld of %lld, %lld evictions, %lld failed reads; the GPU computed %.1f experts a token from the store "
                     "(%lld slots visible)\n",
                     100.0 * h / std::max(1.0, h + m), h / nt, m / nt, (double) (s.store_bytes - s0.store_bytes) / 1e6 / nt,
                     (s.store_wait_ms - s0.store_wait_ms) / nt, (long long) (s.store_prefetched - s0.store_prefetched),
                     (long long) (s.store_pf_used - s0.store_pf_used), (long long) (s.store_pf_wasted - s0.store_pf_wasted),
                     (long long) s.store_slots, (long long) s.store_slots_max, (long long) (s.store_evictions - s0.store_evictions),
                     (long long) (s.store_fails - s0.store_fails), (double) (s.store_zc - s0.store_zc) / nt,
                     (long long) s.store_zc_visible);
    }
    if (s.ahead_picks > s0.ahead_picks) {   // lookahead
        const double p = (double) (s.ahead_picks - s0.ahead_picks), c = (double) (s.ahead_cold - s0.ahead_cold);
        std::fprintf(stderr, "lookahead: the guesses a layer before held %.1f%% of the picks (their top k %.1f%%, two layers "
                             "before %.1f%%); file-read picks asked for early %.1f%% (%lld of %lld); %.1f experts asked "
                             "for early a token\n",
                     100.0 * (double) (s.ahead_hit - s0.ahead_hit) / p, 100.0 * (double) (s.ahead_hit_k - s0.ahead_hit_k) / p,
                     100.0 * (double) (s.ahead_hit2 - s0.ahead_hit2) / p,
                     c > 0 ? 100.0 * (double) (s.ahead_cold_hit - s0.ahead_cold_hit) / c : 0.0,
                     (long long) (s.ahead_cold_hit - s0.ahead_cold_hit), (long long) (s.ahead_cold - s0.ahead_cold),
                     (double) (s.ahead_pf - s0.ahead_pf) / nt);
        std::string rk = "lookahead: picked by the guess's rank (1 ..):";
        for (int i = 0; i < 16; ++i) {
            const long long n = (long long) (s.ahead_rank_n[i] - s0.ahead_rank_n[i]);
            if (n <= 0) break;
            char b[24];
            std::snprintf(b, sizeof b, " %.0f%%", 100.0 * (double) (s.ahead_rank_hit[i] - s0.ahead_rank_hit[i]) / (double) n);
            rk += b;
        }
        std::fprintf(stderr, "%s\n", rk.c_str());
    }
    if (s.prof_tokens > s0.prof_tokens) {   // DS4_PROF
        const double pt = (double) (s.prof_tokens - s0.prof_tokens);
        double sum = 0;
        for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s0.prof_ms[i]) / pt;
        std::fprintf(stderr, "GPU per token (DS4_PROF): %.2f ms - between layers %.2f, attention %.2f, routing + shared "
                             "expert %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f\n", sum,
                     (s.prof_ms[0] - s0.prof_ms[0]) / pt, (s.prof_ms[1] - s0.prof_ms[1]) / pt,
                     (s.prof_ms[2] - s0.prof_ms[2]) / pt, (s.prof_ms[3] - s0.prof_ms[3]) / pt,
                     (s.prof_ms[4] - s0.prof_ms[4]) / pt, (s.prof_ms[5] - s0.prof_ms[5]) / pt);
    }
    if (df != nullptr) std::fclose(df);
    m.save_profile();
    return 0;
}
