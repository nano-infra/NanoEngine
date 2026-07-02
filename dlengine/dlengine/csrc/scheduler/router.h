#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "dlengine/csrc/sequence/sequence.h"

#include "group_manager.h"

namespace dlengine {

// Decision context for one routing call. Holds the live per-rank worker state
// the routers rank candidates from, so strategies stay decoupled from the rest
// of the scheduler internals.
struct RouteContext {
    const std::vector<std::shared_ptr<GroupManager>>& group_manager;
    int                                               attention_dp;
};

// Base class for DP-rank routing strategies.
//
// A router returns an ORDERED list of candidate DP ranks for a sequence; the
// scheduler attempts ``try_allocate`` on each in order until one is admitted.
// Returning a single-element list expresses a hard preference ("hold this
// sequence for that rank"): if it cannot allocate this step, the scheduler
// leaves the sequence waiting and retries next step.
//
// Stateful strategies (round-robin cursor, session affinity table, wait
// budget) keep their own members; the scheduler owns one long-lived Router
// instance per strategy.
class Router {
public:
    virtual ~Router() = default;

    // Ordered DP-rank candidates for ``seq`` (earlier = more preferred).
    virtual std::vector<int> rank_candidates(const Sequence& seq, const RouteContext& ctx) = 0;

    // Notify the router that ``seq`` was admitted on ``dp_idx`` (lets stateful
    // routers record affinity and clear any wait bookkeeping).
    virtual void on_placed(const Sequence& /*seq*/, int /*dp_idx*/) {}

    // Notify the router that ``seq_id`` was aborted so it can clear bookkeeping.
    virtual void on_aborted(uint64_t /*seq_id*/) {}

    virtual const char* name() const = 0;

protected:
    // DP ranks ordered by ascending running-token load (least-cache order):
    // the load-balancing fallback shared by every strategy.
    static std::vector<int> least_cache_order(const RouteContext& ctx)
    {
        std::vector<int> order(ctx.attention_dp);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return ctx.group_manager[a]->num_running_tokens() < ctx.group_manager[b]->num_running_tokens();
        });
        return order;
    }

    // ``preferred`` first, then the remaining ranks in load order (spill path).
    static std::vector<int> preferred_first(int preferred, const std::vector<int>& base)
    {
        std::vector<int> out;
        out.reserve(base.size());
        out.push_back(preferred);
        for (int dp : base) {
            if (dp != preferred) {
                out.push_back(dp);
            }
        }
        return out;
    }
};

// Classic round-robin: each sequence starts the rotation one rank further on.
class RoundRobinRouter: public Router {
public:
    std::vector<int> rank_candidates(const Sequence& /*seq*/, const RouteContext& ctx) override
    {
        std::vector<int> out(ctx.attention_dp);
        for (int i = 0; i < ctx.attention_dp; ++i) {
            out[i] = (cursor_ + i) % ctx.attention_dp;
        }
        cursor_ = (cursor_ + 1) % ctx.attention_dp;
        return out;
    }
    const char* name() const override
    {
        return "RoundRobin";
    }

private:
    int cursor_ = 0;
};

// Least-loaded routing, by running sequence count (LeastBatch) or running
// token count (LeastCache).
class LeastLoadedRouter: public Router {
public:
    explicit LeastLoadedRouter(bool by_tokens): by_tokens_(by_tokens) {}

    std::vector<int> rank_candidates(const Sequence& /*seq*/, const RouteContext& ctx) override
    {
        std::vector<int> out(ctx.attention_dp);
        std::iota(out.begin(), out.end(), 0);
        auto load = [&](int i) {
            return by_tokens_ ? ctx.group_manager[i]->num_running_tokens() : ctx.group_manager[i]->num_running_seqs();
        };
        std::stable_sort(out.begin(), out.end(), [&](int a, int b) { return load(a) < load(b); });
        return out;
    }
    const char* name() const override
    {
        return by_tokens_ ? "LeastCache" : "LeastBatch";
    }

private:
    bool by_tokens_;
};

// Session / prefix-aware routing.
//
// Two affinity signals, in priority order:
//   1. Explicit client session key (Sequence::affinity_key()): authoritative.
//      The session is pinned to the rank that first served it, and the
//      sequence is HELD for that rank up to ``max_wait_`` scheduling steps
//      before spilling to the least-loaded rank (bounded head-of-line wait).
//   2. Content-derived longest cached prefix: a soft hint. We try the rank
//      with the longest warm prefix first, then immediately spill in load
//      order. This avoids hot-spotting when many distinct sessions merely
//      share a common system-prompt prefix.
//
// With no signal at all it degrades to least-cache load balancing.
class SessionPrefixRouter: public Router {
public:
    explicit SessionPrefixRouter(int max_wait = 3): max_wait_(max_wait) {}

    std::vector<int> rank_candidates(const Sequence& seq, const RouteContext& ctx) override
    {
        std::vector<int> base = least_cache_order(ctx);

        const uint64_t key = seq.affinity_key();
        if (key != 0) {
            auto it = affinity_.find(key);
            if (it != affinity_.end()) {
                const int preferred = it->second;
                // Hold for the session's rank until the wait budget is spent.
                int& waited = wait_[seq.seq_id()];
                if (waited < max_wait_) {
                    ++waited;
                    return {preferred};
                }
                return preferred_first(preferred, base);
            }
            // First sighting of this key: place by load; on_placed records it.
            return base;
        }

        // No explicit key: soft cache-aware preference.
        int preferred = -1;
        int best      = 0;
        for (int i = 0; i < ctx.attention_dp; ++i) {
            const int m = ctx.group_manager[i]->matched_prefix_blocks(const_cast<Sequence&>(seq));
            if (m > best) {
                best      = m;
                preferred = i;
            }
        }
        if (preferred < 0) {
            return base;
        }
        return preferred_first(preferred, base);
    }

    void on_placed(const Sequence& seq, int dp_idx) override
    {
        const uint64_t key = seq.affinity_key();
        if (key != 0) {
            affinity_[key] = dp_idx;
        }
        wait_.erase(seq.seq_id());
    }

    void on_aborted(uint64_t seq_id) override
    {
        wait_.erase(seq_id);
    }

    const char* name() const override
    {
        return "SessionPrefix";
    }

private:
    int                               max_wait_;
    std::unordered_map<uint64_t, int> affinity_;  // session key -> dp_idx
    std::unordered_map<uint64_t, int> wait_;      // seq_id -> steps waited
};

// Factory: build the Router for a strategy enum.
inline std::unique_ptr<Router> make_router(RoutingStrategy strategy)
{
    switch (strategy) {
        case RoutingStrategy::RoundRobin:
            return std::make_unique<RoundRobinRouter>();
        case RoutingStrategy::LeastBatch:
            return std::make_unique<LeastLoadedRouter>(/*by_tokens=*/false);
        case RoutingStrategy::LeastCache:
            return std::make_unique<LeastLoadedRouter>(/*by_tokens=*/true);
        case RoutingStrategy::SessionPrefix:
            return std::make_unique<SessionPrefixRouter>();
    }
    return std::make_unique<RoundRobinRouter>();
}

}  // namespace dlengine
