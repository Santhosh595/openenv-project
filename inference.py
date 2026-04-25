#!/usr/bin/env python3
"""
inference.py — Supply Chain Disruption Management OpenEnv
==========================================================
Hackathon-compliant inference script.

Required environment variables:
  API_BASE_URL  — LLM endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — API key (Hugging Face or OpenAI)

Emits EXACT structured stdout format required by evaluator:
  [START] task=<task> env=<env> model=<model>
  [STEP] step=<n> action=<str> reward=<float> done=<bool> error=<str|None>
  [END] success=<bool> steps=<n> score=<float> rewards=<list>
"""

from __future__ import annotations
import os
import sys

# ── Safe path: ensures local modules (environment, models, etc.) are always
#    importable regardless of the working directory the evaluator uses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Required environment variables (hackathon spec)
# ─────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")

ENV_BENCHMARK: str = "supply-chain-disruption-management"

# ─────────────────────────────────────────────────────────────
# Per-task caps — total inference < 20 minutes on 2vCPU/8GB
# ─────────────────────────────────────────────────────────────
TASK_CONFIGS: Dict[str, Dict] = {
    "task_easy":   {"max_steps": 15, "max_total_reward": 15.0, "success_threshold": 0.45},
    "task_medium": {"max_steps": 25, "max_total_reward": 25.0, "success_threshold": 0.35},
    "task_hard":   {"max_steps": 35, "max_total_reward": 35.0, "success_threshold": 0.25},
}
TASKS_TO_RUN: List[str] = ["task_easy", "task_medium", "task_hard"]


# ─────────────────────────────────────────────────────────────
# Mandatory structured log helpers (exact hackathon format)
# ─────────────────────────────────────────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    print(
        f"[STEP] step={step} action={action!r} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# OpenAI client (uses API_BASE_URL + HF_TOKEN as key)
# ─────────────────────────────────────────────────────────────

def make_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


SYSTEM_PROMPT = """You are a supply chain operations manager routing logistics orders.
You see the current network state and must choose a routing decision for ONE pending order.

Respond with ONLY valid JSON (no markdown, no extra text):
{"order_id": "<id>", "routing_decision": "<decision>", "alternate_supplier": null, "reasoning": "<1 sentence>"}

routing_decision must be exactly one of:
standard_route, express_route, spot_market, split_shipment, defer_24h, defer_48h, source_alternative, partial_fulfill

Decision rules:
- CRITICAL SLA + slack <= 2d + disruption active  → spot_market
- CRITICAL SLA + slack <= 2d                       → express_route
- STANDARD  SLA + slack <= 3d + disruption active  → split_shipment
- Any SLA + slack >= 5d                            → standard_route
- FLEXIBLE  SLA + slack <= 2d                      → defer_24h
- Budget < $5000                                   → partial_fulfill
- Primary supplier disrupted                       → source_alternative
- Default                                          → standard_route"""


def get_llm_action(client: OpenAI, obs: dict) -> Optional[dict]:
    """
    Call the LLM and parse action dict. Returns None on any failure.
    Gracefully handles models that don't support json_object response format.
    """
    orders = obs.get("pending_orders", [])
    if not orders:
        return None

    day = obs["episode_day"]
    budget = obs["budget_remaining"]
    disruptions = obs.get("active_disruptions", [])

    lines = [
        f"Day {day}/{obs['max_days']} | Budget: ${budget:,.0f} | Spot premium: {obs['spot_market_premium']:.2f}x",
        f"On-time rate: {obs['on_time_delivery_rate']:.1%} | Critical SLA: {obs['service_level']:.1%}",
    ]
    if disruptions:
        dis_summary = [f"{d['event_type']}(sev={d['severity']:.0%})" for d in disruptions]
        lines.append(f"ACTIVE DISRUPTIONS: {', '.join(dis_summary)}")

    lines.append("PENDING ORDERS (pick the most urgent one):")
    for o in orders[:5]:
        slack = o["deadline_day"] - day
        lines.append(
            f"  {o['order_id']} [{o['sla_tier'].upper():8}] → {o['demand_node']} "
            f"| {o['units_required']} units | {slack}d slack | ${o['value_usd']:,.0f}"
        )
    lines.append("\nRespond with JSON only.")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            temperature=0.1,
            max_tokens=250,
        )
        raw = response.choices[0].message.content or ""
        # Extract JSON even if model wraps it in markdown
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
    return None


# ─────────────────────────────────────────────────────────────
# Heuristic fallback (used when LLM unavailable/fails)
# ─────────────────────────────────────────────────────────────

def heuristic_action(obs: dict) -> dict:
    """Cost-priority heuristic — mirrors /baseline endpoint logic."""
    orders = obs.get("pending_orders", [])
    day    = obs["episode_day"]
    budget = obs["budget_remaining"]
    disruptions = obs.get("active_disruptions", [])

    sla_priority = {"critical": 0, "standard": 1, "flexible": 2}
    orders_sorted = sorted(
        orders,
        key=lambda o: (sla_priority[o["sla_tier"]], o["deadline_day"] - day),
    )
    order = orders_sorted[0]
    slack = order["deadline_day"] - day
    sla   = order["sla_tier"]

    if budget < 5000:
        decision = "partial_fulfill"
        reason   = "Budget critically low — preserve funds."
    elif disruptions and sla == "critical" and slack <= 2:
        decision = "spot_market"
        reason   = f"Critical + {slack}d slack + active disruption."
    elif sla == "critical" and slack <= 2:
        decision = "express_route"
        reason   = f"Critical SLA, only {slack}d slack."
    elif disruptions and sla == "standard" and slack <= 3:
        decision = "split_shipment"
        reason   = "Disruption active + tight standard deadline."
    elif slack >= 5:
        decision = "standard_route"
        reason   = f"{slack}d slack — standard is cost-efficient."
    elif sla == "flexible" and slack <= 2:
        decision = "defer_24h"
        reason   = "Flexible SLA — defer to let disruption clear."
    else:
        decision = "standard_route"
        reason   = "Default: standard route."

    return {
        "order_id":         order["order_id"],
        "routing_decision": decision,
        "alternate_supplier": None,
        "reasoning":        reason,
    }


# ─────────────────────────────────────────────────────────────
# Task runner — one full episode with mandatory log output
# ─────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> Dict[str, Any]:
    """
    Run one complete episode and emit [START] / [STEP] / [END] logs.
    Returns summary dict with score and success flag.
    """
    # Local imports so inference.py can run from repo root
    from environment import SupplyChainEnv  # type: ignore
    from models import Action               # type: ignore
    from grader import grade                # type: ignore
    from tasks import TASK_MAP              # type: ignore

    cfg       = TASK_CONFIGS[task_id]
    task_spec = TASK_MAP[task_id]
    max_steps        = cfg["max_steps"]
    max_total_reward = cfg["max_total_reward"]
    success_threshold = cfg["success_threshold"]

    log_start(task=task_id, env=ENV_BENCHMARK, model=MODEL_NAME)

    env = SupplyChainEnv(task_id=task_id)
    obs = env.reset()

    rewards:     List[float] = []
    steps_taken: int   = 0
    score:       float = 0.0
    success:     bool  = False

    try:
        for step in range(1, max_steps + 1):
            if env._is_done():
                break

            obs_dict = obs.model_dump()
            orders   = obs_dict.get("pending_orders", [])

            # No orders on this day — advance and continue
            if not orders:
                env._advance_day()
                obs = env._build_observation()
                continue

            # ── Get action ────────────────────────────────────
            action_data = get_llm_action(client, obs_dict)
            if action_data is None:
                # LLM unavailable or failed — use heuristic
                action_data = heuristic_action(obs_dict)

            # ── Execute action ────────────────────────────────
            error_msg: Optional[str] = None
            reward_val: float = 0.0
            done: bool = False

            try:
                action = Action(
                    order_id=action_data["order_id"],
                    routing_decision=action_data["routing_decision"],
                    alternate_supplier=action_data.get("alternate_supplier"),
                    reasoning=action_data.get("reasoning", ""),
                )
                result    = env.step(action)
                reward_val = float(result.reward.total)
                done       = bool(result.done)
                obs        = result.observation
            except Exception as exc:
                error_msg  = str(exc)[:80]
                reward_val = -0.5
                done       = False

            rewards.append(reward_val)
            steps_taken = step

            # Action summary for log (concise but informative)
            action_str = (
                f"{action_data.get('routing_decision','unknown')}"
                f":{action_data.get('order_id','?')[:16]}"
            )
            log_step(
                step=step,
                action=action_str,
                reward=reward_val,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # ── Compute final score ───────────────────────────────
        raw_score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score     = float(min(max(raw_score, 0.0), 1.0))
        success   = score >= success_threshold

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} fatal error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "steps":   steps_taken,
        "success": success,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(
        f"[DEBUG] HF_TOKEN={'<set>' if HF_TOKEN else '<NOT SET — heuristic fallback active>'}",
        flush=True,
    )
    print(f"[DEBUG] Tasks to run: {TASKS_TO_RUN}", flush=True)

    client = make_client()

    all_results: List[Dict[str, Any]] = []
    for task_id in TASKS_TO_RUN:
        result = run_task(task_id, client)
        all_results.append(result)
        time.sleep(0.5)   # brief pause between tasks

    # Final summary
    print("\n[SUMMARY]", flush=True)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task_id']:<15} score={r['score']:.4f} "
            f"steps={r['steps']} {status}",
            flush=True,
        )

    avg_score = (
        sum(r["score"] for r in all_results) / len(all_results)
        if all_results else 0.0
    )
    print(f"  average_score={avg_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
