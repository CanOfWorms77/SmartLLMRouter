# core/llm_router.py
"""
LLMRouter — unified model routing for Elise Intelligence Engine.

Tiers:
    free     → NVIDIA NIM developer credits (no token cost)
    budget   → RouteLLM cheap models  (~$0.05–0.50/M input)
    standard → RouteLLM mid models    (~$0.50–3.00/M input)
    deep     → RouteLLM premium       (>$3.00/M input)

Usage:
    router = LLMRouter(nvidia_key="...", routellm_key="...")
    await router.refresh_models()
    text = await router.route("free", messages)

Telegram commands (wire in your bot handler):
    /models       → router.get_menu_text()
    /model <n>    → router.handle_model_select(n)
    /llmstatus    → router.status_text()
"""

import asyncio
import os
import aiohttp
import structlog
from dataclasses import dataclass, field
from typing import Optional
from openai import AsyncOpenAI

log = structlog.get_logger(__name__)

# ── Provider base URLs ─────────────────────────────────────────────────────────
NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"
ROUTELLM_BASE_URL = "https://routellm.abacus.ai/v1"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# ── Hardcoded defaults ─────────────────────────────────────────────────────────
# These are safe starting points based on verified pricing as of 2026-03-08.
# Updated automatically when refresh_models() classifies a better option,
# or manually overridden by the user via /model <n> Telegram command.
TIER_DEFAULTS: dict[str, str] = {
    "free":     "deepseek-ai/deepseek-v3.2",                # NVIDIA NIM, free
    "budget":   "gpt-5-nano",                               # RouteLLM ~$0.05/M
    "standard": "gpt-5.2",                                  # RouteLLM ~$1.75/M
    "deep":     "claude-sonnet-4-6",                        # RouteLLM ~$3.00/M
    "search":   "sonar",          
}

TIER_FALLBACKS: dict[str, str] = {
    "free":     "gpt-4.1-mini",                             # NVIDIA NIM
    "budget":   "gpt-4.1-mini",                             # RouteLLM ~$0.30/M
    "standard": "gemini-2.5-pro",                           # RouteLLM ~$1.25/M
    "deep":     "route-llm",                                # Abacus auto-route
    "search":   "sonar-pro",          
}

# ── RouteLLM: models to skip entirely ─────────────────────────────────────────
# These are either free on NVIDIA NIM already, legacy/superseded,
# or duplicates that would bloat the menu without adding value.
SKIP_ROUTELLM: set[str] = {
    # Free on NVIDIA — no reason to pay RouteLLM for these
    "deepseek-ai/deepseek-v3.2",
    "deepseek/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.1-terminus",
    "meta-llama/meta-llama-3.1-8b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "meta-llama/meta-llama-3.1-405b-instruct-turbo",
    "openai/gpt-oss-120b",
    # Legacy — superseded by newer models at same or lower price
    "gpt-4o-2024-11-20",
    "claude-3-7-sonnet-20250219",
    # Coding/specialised — not useful for alert generation
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-xhigh",
    "grok-code-fast-1",
    "qwen-2.5-coder-32b",
    "qwen/qwen3-coder-480b-a35b-instruct",
}

# ── RouteLLM: keyword-based tier classification ────────────────────────────────
# Matched against lowercased model ID. First match wins: deep → standard → budget.
# Anything unmatched falls to standard (safe default).
_DEEP_KEYWORDS: list[str] = [
    "claude-sonnet-4",
    "claude-opus-4",
    "claude-haiku-4",
    "gpt-5.3",
    "gpt-5.4",
    "o3-pro",
    "grok-4-0709",
    "route-llm",
    "o3 ",          # trailing space avoids matching o3-mini
    "/o3",
]

_STANDARD_KEYWORDS: list[str] = [
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5-mini",   # $0.25/M but better quality than nano
    "gpt-5 ",
    "/gpt-5\n",     # edge case — gpt-5 exact
    "gpt-4.1",
    "gemini-2.5",
    "gemini-3-pro",
    "gemini-3.1-pro",
    "deepseek-r1",
    "deepseek-v3",
    "kimi-k2",
    "grok-4-fast",
    "grok-4-1-fast",
    "qwen3",
    "qwq-32b",
    "glm-4",
    "glm-5",
    "o3-mini",
    "o4-mini",
]

_BUDGET_KEYWORDS: list[str] = [
    "gpt-5-nano",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "flash-lite",
    "gemini-3-flash",
    "gemini-3.1-flash",
    "llama-3.3-70b-versatile",
]

# ── NVIDIA NIM: model types to skip ───────────────────────────────────────────
# The NIM catalogue is huge — filter to text-generation chat models only.
_NVIDIA_SKIP_KEYWORDS: set[str] = {
    "embed", "clip", "reward", "guard", "vision", "vlm",
    "safety", "retriev", "parse", "translate", "pii",
    "coder", "code", "starcoder", "bge", "neva", "vila",
    "deplot", "paligemma", "kosmos", "whisper", "riva",
    "chatqa", "usdcode", "streampetr", "nvclip",
    "rerank", "nv-embed", "embed-qa",
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    id:       str
    provider: str    # "nvidia" | "routellm"
    tier:     str    # "free" | "budget" | "standard" | "deep"
    context:  int = 0


@dataclass
class ModelRegistry:
    free:     list[ModelEntry] = field(default_factory=list)
    budget:   list[ModelEntry] = field(default_factory=list)
    standard: list[ModelEntry] = field(default_factory=list)
    deep:     list[ModelEntry] = field(default_factory=list)
    search:   list[ModelEntry] = field(default_factory=list)

    # User-selected active model per tier (overrides TIER_DEFAULTS)
    _active: dict[str, str] = field(default_factory=dict)

    def for_tier(self, tier: str) -> list[ModelEntry]:
        return getattr(self, tier, [])

    def get_active_id(self, tier: str) -> str:
        return self._active.get(tier, TIER_DEFAULTS[tier])

    def set_active(self, tier: str, model_id: str) -> bool:
        all_ids = [m.id for m in self.for_tier(tier)]
        if model_id in all_ids:
            self._active[tier] = model_id
            log.info("llm_router.model_activated", tier=tier, model=model_id)
            return True
        log.warning("llm_router.model_not_found", tier=tier, model=model_id)
        return False

    def total(self) -> int:
        return sum(len(self.for_tier(t)) for t in ("free", "budget", "standard", "deep"))

@dataclass
class UsageRecord:
    model_id: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate. Update rates as pricing changes."""
        rates = {
            "free":     (0.0,    0.0),
            "budget":   (0.05,   0.20),   # per 1M tokens in/out
            "standard": (1.75,   7.00),
            "deep":     (3.00,  15.00),
            "search":   (0.006,  0.006),  # per request flat
        }
        in_rate, out_rate = rates.get(self.tier, (1.75, 7.00))
        return (self.prompt_tokens * in_rate + self.completion_tokens * out_rate) / 1_000_000



# ── Main class ─────────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Routes LLM calls across NVIDIA NIM (free) and RouteLLM/Abacus (paid tiers).
    Discovers available models from live /v1/models endpoints at startup and on
    daily refresh. Falls back to hardcoded TIER_FALLBACKS if the primary model
    errors. User can swap active models via Telegram /model command.
    """

    def __init__(self, nvidia_key: str, routellm_key: str, perplexity_key: str = ""):
        self.registry = ModelRegistry()
        self.registry._active = dict(TIER_DEFAULTS)  # start with known-good defaults

        self._nvidia_client = AsyncOpenAI(
            api_key=nvidia_key,
            base_url=NVIDIA_BASE_URL,
        )
        self._routellm_client = AsyncOpenAI(
            api_key=routellm_key,
            base_url=ROUTELLM_BASE_URL,
        )
        self._perplexity_client = AsyncOpenAI(
            api_key=perplexity_key or os.getenv("PERPLEXITY_GMAIL_KEY", ""),
            base_url=PERPLEXITY_BASE_URL,
        )

        # Seed search tier with Sonar models (static — no live discovery endpoint needed)
        self.registry.search = [
            ModelEntry(id="sonar",     provider="perplexity", tier="search", context=127000),
            ModelEntry(id="sonar-pro", provider="perplexity", tier="search", context=127000),
        ]

        # Built by get_menu_text() — maps displayed number → (tier, model_id)
        self._menu_index: dict[int, tuple[str, str]] = {}

        self._session_cost: float = 0.0


    # ── Model discovery ────────────────────────────────────────────────────────

    async def refresh_models(self) -> None:
        """Fetch live model lists from both providers and rebuild registry."""
        log.info("llm_router.refresh_start")

        async with aiohttp.ClientSession() as session:
            nvidia_raw, routellm_raw = await asyncio.gather(
                self._fetch_models(
                    session,
                    f"{NVIDIA_BASE_URL}/models",
                    self._nvidia_client.api_key,
                    "nvidia",
                ),
                self._fetch_models(
                    session,
                    f"{ROUTELLM_BASE_URL}/models",
                    self._routellm_client.api_key,
                    "routellm",
                ),
            )

        self.registry.free   = self._classify_nvidia(nvidia_raw)
        budget, std, deep    = self._classify_routellm(routellm_raw)
        self.registry.budget   = budget
        self.registry.standard = std
        self.registry.deep     = deep

        log.info(
            "llm_router.refresh_complete",
            free=len(self.registry.free),
            budget=len(self.registry.budget),
            standard=len(self.registry.standard),
            deep=len(self.registry.deep),
            total=self.registry.total(),
        )

    @staticmethod
    async def _fetch_models(
        session: aiohttp.ClientSession,
        url: str,
        api_key: str,
        provider: str,
    ) -> list[dict]:
        try:
            async with session.get(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                models = data.get("data", [])
                log.info("llm_router.fetch_ok", provider=provider, count=len(models))
                return models
        except Exception as e:
            log.warning("llm_router.fetch_failed", provider=provider,
                        error=type(e).__name__, detail=str(e))
            return []

    @staticmethod
    def _classify_nvidia(raw: list[dict]) -> list[ModelEntry]:
        """Keep text-generation chat models only. All NVIDIA NIM = free tier."""
        entries = []
        for m in raw:
            mid = m.get("id", "")
            if any(k in mid.lower() for k in _NVIDIA_SKIP_KEYWORDS):
                continue
            entries.append(ModelEntry(
                id=mid,
                provider="nvidia",
                tier="free",
                context=m.get("context_length") or m.get("context_window") or 0,
            ))
        return sorted(entries, key=lambda x: x.id)

    @staticmethod
    def _classify_routellm(raw: list[dict]) -> tuple[
        list[ModelEntry], list[ModelEntry], list[ModelEntry]
    ]:
        """
        Classify RouteLLM models into budget / standard / deep.
        Skips models in SKIP_ROUTELLM. Uses keyword matching on lowercased ID.
        Unmatched models default to standard (conservative).
        """
        budget, standard, deep = [], [], []

        for m in raw:
            mid = m.get("id", "")
            if mid.lower() in {s.lower() for s in SKIP_ROUTELLM}:
                continue

            mid_lower = mid.lower()
            ctx = m.get("context_length") or m.get("context_window") or 0

            entry = ModelEntry(id=mid, provider="routellm", tier="", context=ctx)

            if any(k in mid_lower for k in _DEEP_KEYWORDS):
                entry.tier = "deep"
                deep.append(entry)
            elif any(k in mid_lower for k in _STANDARD_KEYWORDS):
                entry.tier = "standard"
                standard.append(entry)
            elif any(k in mid_lower for k in _BUDGET_KEYWORDS):
                entry.tier = "budget"
                budget.append(entry)
            else:
                # Unknown model — default to standard rather than risking
                # an expensive deep call or an underpowered budget call
                entry.tier = "standard"
                standard.append(entry)

        return (
            sorted(budget,   key=lambda x: x.id),
            sorted(standard, key=lambda x: x.id),
            sorted(deep,     key=lambda x: x.id),
        )

    # ── Routing ────────────────────────────────────────────────────────────────

    async def route(
        self,
        tier: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> str:
        """
        Route a chat completion to the active model for the given tier.
        Falls back to TIER_FALLBACKS on any exception.

        Args:
            tier:        "free" | "budget" | "standard" | "deep"
            messages:    OpenAI-format message list
            max_tokens:  Upper limit on response length
            temperature: Sampling temperature

        Returns:
            Response content string.

        Raises:
            RuntimeError if both primary and fallback fail.
        """
        if tier not in TIER_DEFAULTS:
            raise ValueError(f"Unknown tier '{tier}'. Choose: free, budget, standard, deep")

        model_id = self.registry.get_active_id(tier)
        client = self._client_for_tier(tier)

        log.info("llm_router.request", tier=tier, model=model_id, messages=len(messages))

        try:
            content, usage = await self._call(client, model_id, messages, max_tokens, temperature)
        except Exception as primary_err:
            log.warning("llm_router.primary_failed", tier=tier, model=model_id, error=str(primary_err))
            content, usage = await self._run_fallback(tier, messages, max_tokens, temperature)

        if usage:
            record = UsageRecord(model_id=model_id, tier=tier, **usage)
            self._session_cost += record.cost_estimate_usd
            log.info(
                "llm_router.usage",
                tier=tier,
                model=model_id,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                total_tokens=record.total_tokens,
                cost_usd=round(record.cost_estimate_usd, 6),
                session_cost_usd=round(self._session_cost, 4),
            )

        return content

    async def _run_fallback(
        self,
        tier: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict]:
        fallback_id = TIER_FALLBACKS.get(tier, TIER_DEFAULTS[tier])
        client = self._client_for_tier(tier)

        log.info("llm_router.fallback_attempt", tier=tier, model=fallback_id)
        try:
            return await self._call(client, fallback_id, messages, max_tokens, temperature)
        except Exception as fallback_err:
            log.error("llm_router.fallback_failed", tier=tier, model=fallback_id, error=str(fallback_err))
            raise RuntimeError(
                f"LLMRouter: both primary and fallback failed for tier '{tier}'"
            ) from fallback_err

    @staticmethod
    async def _call(
        client: AsyncOpenAI,
        model_id: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict]:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        timeout=25.0,
        content = resp.choices[0].message.content or ""
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }
        return content, usage


    def _client_for_tier(self, tier: str) -> AsyncOpenAI:
            if tier == "free":
                return self._nvidia_client
            if tier == "search":
                return self._perplexity_client
            return self._routellm_client

    # ── Telegram UI helpers ────────────────────────────────────────────────────

    def get_menu_text(self) -> str:
        """
        Build a numbered model menu for Telegram.
        Rebuilds self._menu_index so handle_model_select() works correctly.
        Call this before handle_model_select().
        """
        tier_labels = {
            "free":     "🟢 Free     [NVIDIA NIM — no cost]",
            "budget":   "🔵 Budget   [RouteLLM ~$0.05–0.50/M]",
            "standard": "🟡 Standard [RouteLLM ~$0.50–3.00/M]",
            "deep":     "🔴 Deep     [RouteLLM >$3.00/M]",
            "search":   "🔍 Search [Perplexity Sonar — ~$0.006/req]",
        }

        self._menu_index = {}
        lines = ["🤖 *Elise — LLM Model Menu*\n"]
        idx = 1

        for tier in ("free", "budget", "standard", "deep", "search"):
            models = self.registry.for_tier(tier)
            if not models:
                continue

            active_id = self.registry.get_active_id(tier)
            lines.append(f"*{tier_labels[tier]}*")

            for m in models:
                marker = "✅" if m.id == active_id else "  "
                # Trim org prefix (e.g. meta/ mistralai/) for readability
                short_name = m.id.split("/")[-1]
                ctx_str    = f"  {m.context // 1000}K" if m.context else ""
                lines.append(f"{marker} `{idx}` {short_name}{ctx_str}")
                self._menu_index[idx] = (tier, m.id)
                idx += 1

            lines.append("")

        lines.append("_Reply_ `/model <number>` _to activate a model for its tier._")
        return "\n".join(lines)

    def handle_model_select(self, number: int) -> str:
        """
        Handle a /model <n> command from Telegram.
        Must call get_menu_text() first to build the index.
        """
        if not self._menu_index:
            return "⚠️ Run /models first to load the menu, then pick a number."

        entry = self._menu_index.get(number)
        if not entry:
            return f"❌ Invalid number `{number}`. Run /models to see valid options."

        tier, model_id = entry
        success = self.registry.set_active(tier, model_id)
        if success:
            short = model_id.split("/")[-1]
            return f"✅ *{tier.upper()}* tier → `{short}`"
        return f"❌ Could not activate `{model_id}` — model not found in registry."

    def status_text(self) -> str:
        """Return a compact /llmstatus summary for Telegram."""
        lines = ["🔧 *Elise — LLM Router Status*\n"]
        for tier in ("free", "budget", "standard", "deep", "search"):
            active   = self.registry.get_active_id(tier)
            fallback = TIER_FALLBACKS[tier]
            count    = len(self.registry.for_tier(tier))
            lines.append(
                f"*{tier.upper()}* ({count} models)\n"
                f"  active   → `{active.split('/')[-1]}`\n"
                f"  fallback → `{fallback.split('/')[-1]}`"
            )
            lines.append("")
        lines.append(f"_Total models loaded: {self.registry.total()}_")
        lines.append(f"_Session cost estimate: ${self._session_cost:.4f} USD_")
        return "\n".join(lines)

    def reset_tier(self, tier: str) -> str:
        """Reset a tier back to its hardcoded default (useful if a custom pick misbehaves)."""
        if tier not in TIER_DEFAULTS:
            return f"❌ Unknown tier '{tier}'"
        self.registry._active[tier] = TIER_DEFAULTS[tier]
        return f"🔄 *{tier.upper()}* reset → `{TIER_DEFAULTS[tier].split('/')[-1]}`"
