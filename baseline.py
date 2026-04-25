"""
Baseline inference script — LLM-powered agent using OpenAI API.
Reads OPENAI_API_KEY from environment. Reproducible via fixed task seeds.
Run: python baseline.py [--task task_easy|task_medium|task_hard|all]
"""

from __future__ import annotations
import os
import json
import argparse
import time
from typing import List

from openai import OpenAI

from environment import SupplyChainEnv
from grader import grade
from models import Action, BaselineResult
from tasks import TASK_MAP

# Respect the same env vars as inference.py
_MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
_API_BASE:   str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
_HF_TOKEN:   str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))


SYSTEM_PROMPT = """You are an expert supply chain operations manager.
You will receive the current state of a logistics network and must decide how to route a specific order.

Your decisions must:
1. Prioritize CRITICAL SLA orders — deliver on time even at higher cost
2. Use express/spot routes only when deadline math demands it
3. Prefer standard routes for flexible orders
4. Consider active disruptions — disrupted lanes increase transit time
5. Monitor budget — running out of budget means future critical orders go unmet

Respond with ONLY a valid JSON object in this exact format:
{
  "order_id": "<the order_id from pending_orders>",
  "routing_decision": "<one of: standard_route, express_route, spot_market, split_shipment, defer_24h, defer_48h, source_alternative, partial_fulfill>",
  "alternate_supplier": null,
  "reasoning": "<1-2 sentence explanation of your decision>"
}

Routing decision guide:
- standard_route: cheapest, longest transit — use for flexible orders with slack
- express_route: 2.8x cost, 35% transit time — use when deadline is tight
- spot_market: 4.5x cost, fastest — use ONLY for critical orders at imminent deadline risk
- split_shipment: 1.6x cost, 80% transit — hedges risk across two paths
- defer_24h / defer_48h: free now, but adds delay — only if deadline window allows
- source_alternative: 1.3x cost, 120% transit — use when primary supplier is disrupted
- partial_fulfill: 0.5x cost — last resort when budget is critically low
"""


def build_prompt(obs_dict: dict) -> str:
    pending = obs_dict["pending_orders"]
    disruptions = obs_dict["active_disruptions"]
    budget = obs_dict["budget_remaining"]
    day = obs_dict["episode_day"]
    max_days = obs_dict["max_days"]

    lines = [
        f"Day {day}/{max_days} | Budget remaining: ${budget:,.0f}",
        f"On-time rate so far: {obs_dict['on_time_delivery_rate']:.1%}",
        f"Spot market premium: {obs_dict['spot_market_premium']:.2f}x",
        "",
    ]

    if disruptions:
        lines.append("⚠ ACTIVE DISRUPTIONS:")
        for d in disruptions:
            lines.append(f"  [{d['event_type'].upper()}] affects nodes {d['affected_nodes']} "
                         f"and lanes {d['affected_lanes']} — severity {d['severity']:.0%} "
                         f"— ~{d['estimated_duration_days']}d duration (uncertainty {d['uncertainty']:.0%})")
        lines.append("")

    lines.append("PENDING ORDERS (act on ONE per step):")
    for o in pending[:5]:   # show top 5 to fit context
        slack = o["deadline_day"] - day
        lines.append(
            f"  [{o['sla_tier'].upper():8}] {o['order_id']} | {o['units_required']} units → {o['demand_node']} "
            f"| deadline day {o['deadline_day']} ({slack}d slack) | value ${o['value_usd']:,.0f}"
        )

    lines.append("")
    lines.append("Choose the MOST URGENT order above and decide its routing.")
    return "\n".join(lines)


def run_baseline(task_id: str, client: OpenAI, verbose: bool = True) -> BaselineResult:
    env = SupplyChainEnv(task_id=task_id)
    task = TASK_MAP[task_id]
    obs = env.reset()

    steps = 0
    max_steps = task.max_steps
    reasoning_samples: List[str] = []
    cumulative_reward = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task.name} [{task.difficulty.upper()}]")
        print(f"{'='*60}")

    while not env._is_done() and steps < max_steps:
        obs_dict = obs.model_dump()

        if not obs_dict["pending_orders"]:
            # No orders yet — advance
            break

        prompt = build_prompt(obs_dict)

        try:
            response = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content or "{}"
            action_data = json.loads(raw)
            action = Action(**action_data)
        except Exception as e:
            # Fallback: take first order with standard route
            if obs_dict["pending_orders"]:
                order = obs_dict["pending_orders"][0]
                action = Action(
                    order_id=order["order_id"],
                    routing_decision="standard_route",
                    alternate_supplier=None,
                    reasoning=f"Fallback due to API error: {e}",
                )
            else:
                break

        result = env.step(action)
        cumulative_reward += result.reward.total

        if verbose and steps < 5:
            print(f"\nStep {steps+1}: {action.routing_decision} for {action.order_id}")
            print(f"  Reward: {result.reward.total:+.3f} | {result.reward.reasoning[:120]}")

        if steps < 3:
            reasoning_samples.append(
                f"[{action.routing_decision}] {action.reasoning or 'no reasoning provided'}"
            )

        obs = result.observation
        steps += 1
        time.sleep(0.05)   # rate limit buffer

    # Grade
    final_state = env.state()
    action_history = final_state.pop("action_history", [])
    grade_result = grade(task_id, final_state, action_history, task.budget_usd)

    if verbose:
        print(f"\n{grade_result.explanation}")

    return BaselineResult(
        task_id=task_id,
        agent="gpt-4o-mini",
        score=grade_result.score,
        steps_taken=steps,
        total_cost=final_state.get("cumulative_cost", 0.0),
        on_time_rate=obs.on_time_delivery_rate,
        sla_met=obs.service_level,
        reasoning_samples=reasoning_samples,
    )


def main():
    parser = argparse.ArgumentParser(description="Supply Chain OpenEnv Baseline Agent")
    parser.add_argument("--task", default="all",
                        choices=["task_easy", "task_medium", "task_hard", "all"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--heuristic", action="store_true",
                        help="Run heuristic agent only — no OPENAI_API_KEY required (reproducible)")
    args = parser.parse_args()

    tasks = ["task_easy", "task_medium", "task_hard"] if args.task == "all" else [args.task]
    results: List[BaselineResult] = []

    api_key = _HF_TOKEN
    use_heuristic = args.heuristic or not api_key

    if use_heuristic:
        print("Running heuristic baseline (no LLM — fully reproducible)...")
        # Import heuristic logic from main.py's baseline endpoint
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(__file__))
        from app.main import run_baseline_endpoint, BaselineRunRequest
        for tid in tasks:
            req = BaselineRunRequest(task_id=tid)
            r = run_baseline_endpoint(req)
            results.append(r)
            if not args.quiet:
                print(f"  {tid}: score={r.score:.3f}  steps={r.steps_taken}  "
                      f"on_time={r.on_time_rate:.1%}  sla={r.sla_met:.1%}")
    else:
        client = OpenAI(base_url=_API_BASE, api_key=api_key)
        for tid in tasks:
            r = run_baseline(tid, client, verbose=not args.quiet)
            results.append(r)

    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r.task_id:<15} score={r.score:.3f}  steps={r.steps_taken}  "
              f"on_time={r.on_time_rate:.1%}  sla={r.sla_met:.1%}")

    # Save results
    out = [r.model_dump() for r in results]
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
