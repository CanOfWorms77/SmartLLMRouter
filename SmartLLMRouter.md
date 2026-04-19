### RFC : Turn-Scoped, Subagent-Aware LLM Routing for Hermes

**Author:** Dave ([CanOfWorms777](https://github.com/CanOfWorms777))  
**Status:** Discussion draft — looking for feedback, not asking for a merge  
**Date:** 2026-04-19  
**Target:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

---

## TL;DR

Hermes already has smart model-routing logic for deciding when a turn is simple enough to use a cheaper model. I'd like to open a discussion about extending that into a more general router that is:

- **turn-scoped** — re-evaluates on every turn, not just the first
- **mid-session fault-tolerant** — swaps provider/model on the fly if a rate limit, token cap, or outage is hit mid-conversation, without dropping the session or losing context
- **subagent-aware** — delegated child agents can route independently based on their own goal
- **tier-based** — `free / budget / standard / deep` tiers, each with an ordered fallback chain
- **friendly to custom endpoints** — any OpenAI-compatible endpoint can participate

The goal is simple: keep easy work cheap, send harder work to stronger models, let delegated subagents use models that fit *their* task, and degrade gracefully when a provider becomes unavailable — including in the middle of an active session.

I built something similar for a personal agent system I use for my own research workflows, and it noticeably reduced cost and improved resilience without obvious quality loss. I'm sharing the design here as an idea, not a patch or implementation request, to ask whether this direction feels useful for Hermes.

This is not the same as Hermes' existing provider-routing support for OpenRouter sub-providers. That feature controls which upstream providers OpenRouter prefers for a request; this RFC is about turn-level route resolution across models/providers, mid-session failover when a route becomes unavailable, and applying the same routing logic to delegated subagents. [web:53]

---

## Why this might be worth discussing

Hermes sessions are heterogeneous. In a single conversation you might see:

- a quick acknowledgment
- a short summary
- a debugging request
- a delegated extraction task
- a delegated research task

Those are not the same kind of work, and they do not all need the same model path. Beyond task variety, provider reliability is also a real concern: rate limits, exhausted subscription tokens, temporary outages, and API instability mean sessions can fail mid-flow for reasons that have nothing to do with the task itself.

That creates five practical issues:

1. **Cost mismatch**  
   Simple turns can still consume premium tokens when a cheaper route would give identical results.

2. **Mid-session provider failure**  
   If a provider is rate-limited, a token cap is hit, or a subscription is exhausted mid-conversation, the session stalls rather than continuing on a different model. The user has to manually intervene, break context, and restart — which is a poor experience.

3. **Reliability friction**  
   Even before a full outage, slow responses and intermittent errors from an unhealthy provider degrade the session. An automated fallback would mean the user often never notices.

4. **Delegation mismatch**  
   Subagents doing very different work can inherit a single routing decision that does not match their actual task.

5. **Model-landscape drift**  
   New models arrive constantly. Users have to notice them, evaluate them, and manually update config before they benefit.

Hermes already seems to be moving in this direction with smarter routing and better fallback/error handling. This proposal is really about pushing that idea one step further into a small reusable routing layer — one that operates at the turn level, not just at session start.

---

## Background: where this idea came from

I built a similar routing system for a personal multi-provider agent I use for my own research workflows. In that system:

- Queries are classified into tiers on each turn
- Each tier has an ordered list of provider/model candidates
- If a provider returns a rate-limit error, token-exhausted error, or upstream failure at *any point during a session*, the router catches it, swaps transparently to the next candidate, and retries the same turn
- A daily background job probes available free-model options and evaluates candidates so the free tier stays current

The key insight from running it was that **mid-session swapping is the most operationally valuable part** — not the fancy discovery logic. Being able to hit a rate limit in the middle of a long coding session and have the agent quietly continue on a different provider without losing conversational context was genuinely useful. The discovery layer was nice, but much less immediately important.

That's why this RFC leads with mid-session fault tolerance, not discovery.

---

## The core idea

Treat routing as a lightweight resolver that runs **per turn** and, where relevant, **per agent instance** — and make it capable of executing a model swap mid-session without losing context.

### Proposed properties

**1. Tiered, not binary**  
Replace a simple "cheap vs primary" decision with `free / budget / standard / deep` tiers, each backed by an ordered list of provider/model candidates.

**2. Turn-scoped**  
Re-evaluate routing on each turn rather than implicitly locking a whole session to one model path. A casual follow-up after a heavy task should be allowed to drop back down to a cheaper route.

**3. Mid-session model swapping**  
When a provider error is caught mid-conversation — rate limit, token quota exhausted, subscription cap hit, HTTP 429, 503, or similar — the router should:

- catch the error before it surfaces to the user
- move to the next candidate within the current tier's fallback chain
- retry the same turn with the same message and context intact
- log the swap transparently: e.g. `[routing] openrouter → groq (rate limited)`
- apply exponential backoff on the failed provider so it is not retried immediately
- optionally notify the user if all preferred candidates for a tier have failed

This is different from simply handling startup config. The router needs to remain active throughout the session, not just at the first turn, and it needs to handle failures that happen on turn 50, not just turn 1.

**4. Subagent-aware**  
When Hermes spawns a delegated child, that child can:

- **inherit** the parent's already-resolved route
- **follow delegation-specific config** if set explicitly
- **resolve its own route** from the delegated goal and context

A lightweight extraction subagent and a heavy research subagent should not necessarily be forced onto the same model.

Because delegated child agents are isolated workstreams with their own goal and context, they are a natural place for independent route resolution rather than inheriting a one-size-fits-all parent decision. [web:29][web:30]

Critically, subagents should also benefit from mid-session swapping. A child agent that hits a token cap on its first tool call should fall back to the next candidate rather than failing the whole delegation.

**5. Provider-fault-tolerant by default**  
Provider health state should be tracked per session so the router is aware of which providers have been failing and at what rate. Unhealthy providers should be skipped or deprioritised automatically rather than retried immediately.

**6. Custom endpoint friendly**  
Any OpenAI-compatible endpoint should be usable as a candidate in a tier's fallback chain — local Ollama, LM Studio, private deployments, enterprise gateways, and so on.

---

## Failure modes the router should handle

This is the part I feel is most underspecified in the current smart routing design. These are real failure cases I encountered running my own router over several months:

| Failure mode | Current behaviour | Proposed behaviour |
|---|---|---|
| Provider rate-limited (HTTP 429) | Session stalls or errors | Swap to next candidate in the tier, retry same turn |
| Token quota exhausted mid-session | Session stalls or errors | Swap to next candidate, continue from same turn |
| Subscription cap hit | Session stalls or errors | Swap to next candidate, log the event |
| Provider temporarily unavailable (503) | Session errors | Retry with backoff, then swap candidate |
| All candidates in a tier fail | Session errors | Optionally escalate to next stronger tier |
| Subagent hits a rate limit | Delegation fails | Subagent swaps candidate independently |
| Primary model slow/degraded | Session continues slowly | Configurable: swap after timeout threshold |

---

## Query classification

The router needs a lightweight classifier that does not require an extra LLM call. A heuristic version is sufficient for the initial implementation:

```python
class QueryTier(Enum):
    FREE = "free"
    BUDGET = "budget"
    STANDARD = "standard"
    DEEP = "deep"


def classify_query(message: str, conversation_history: list[str]) -> QueryTier:
    text = (message or "").strip().lower()
    words = text.split()

    if not text:
        return QueryTier.BUDGET

    # FREE tier: very short, no code, no URLs, greeting-like
    if (
        len(words) <= 6
        and "```" not in text
        and "`" not in text
        and "http://" not in text
        and "https://" not in text
    ):
        greetings = {"hi", "hello", "hey", "thanks", "ok", "yes", "no"}
        if words and words[0] in greetings:
            return QueryTier.FREE

    # DEEP tier: multiple strong signals
    deep_signals = [
        "design a system" in text,
        "architect" in text,
        "write a spec" in text,
        "implement from scratch" in text,
        "research and compare" in text,
        len(words) > 200,
        text.count("\n") > 10,
        "```" in text,
    ]
    if sum(deep_signals) >= 2:
        return QueryTier.DEEP

    # STANDARD tier: any moderate complexity signal
    standard_signals = [
        "debug" in text,
        "implement" in text,
        "refactor" in text,
        "analyze" in text,
        "write code" in text,
        "build a" in text,
        "http://" in text or "https://" in text,
        "`" in text,
    ]
    if any(standard_signals):
        return QueryTier.STANDARD

    return QueryTier.BUDGET
```
