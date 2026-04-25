#!/usr/bin/env python3
"""
demo.py — Supply Chain Multi-Agent Demo Runner
===============================================
Shows the full multi-agent decision pipeline with:
  - Environment state at each step
  - All three agent decisions with reasoning
  - Reward breakdown (cost_score, delay_score, sla_score, final_reward)
  - Adaptive strategy updates
  - Episode summary

USAGE:
  python demo.py                          # run task_medium demo (default)
  python demo.py --task task_hard         # harder scenario
  python demo.py --steps 10              # limit steps shown
  python demo.py --json                  # emit JSON for API consumers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo(
    task_id: str = "task_medium",
    max_steps: int = 15,
    emit_json: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a full multi-agent demo episode.

    Returns a structured dict with:
      - task_id, steps_executed
      - per_step_trace: list of step records
      - episode_summary: final KPIs
      - reward_series: list of floats (for plot)
    """
    from environment import SupplyChainEnv
    from agents import MultiAgentCoordinator, AdaptiveStrategy
    from models import Action
    from grader import grade
    from tasks import TASK_MAP

    env         = SupplyChainEnv(task_id=task_id)
    coordinator = MultiAgentCoordinator()
    strategy    = AdaptiveStrategy(window=5)
    obs         = env.reset()

    steps_executed = 0
    reward_series: List[float] = []
    per_step_trace: List[Dict] = []

    SEPARATOR = "═" * 68

    if verbose and not emit_json:
        print(f"\n{SEPARATOR}")
        print(f"  SUPPLY CHAIN MULTI-AGENT DEMO  |  Task: {task_id.upper()}")
        task = TASK_MAP[task_id]
        print(f"  {task.name}")
        print(f"  Budget: ${task.budget_usd:,.0f}  |  Max Days: {task.max_steps}")
        print(SEPARATOR)

    for step in range(1, max_steps + 1):
        if env._is_done():
            break

        obs_dict = obs.model_dump()
        orders   = obs_dict.get("pending_orders", [])

        if not orders:
            env._advance_day()
            obs = env._build_observation()
            continue

        # ── Select most urgent order ─────────────────────────
        day = obs_dict["episode_day"]
        sla_pri = {"critical": 0, "standard": 1, "flexible": 2}
        order   = min(orders, key=lambda o: (sla_pri[o["sla_tier"]], o["deadline_day"] - day))

        # ── Multi-agent decision ─────────────────────────────
        ma_step = coordinator.coordinate(obs_dict, order)
        decision = ma_step.logistics

        # ── Execute action ───────────────────────────────────
        try:
            action = Action(
                order_id=decision.order_id,
                routing_decision=decision.routing_decision,  # type: ignore
                alternate_supplier=decision.alternate_supplier,
                reasoning=decision.reasoning,
            )
            result = env.step(action)
            reward = result.reward
            obs    = result.observation
        except Exception as exc:
            if verbose and not emit_json:
                print(f"  [ERROR] Step {step}: {exc}")
            continue

        # ── Update adaptive strategy ─────────────────────────
        last_action = env._state.get("_last_action_detail", {})
        on_time = last_action.get("on_time", True)
        strategy.record(on_time, order["sla_tier"])

        # ── Structured reward breakdown ──────────────────────
        reward_breakdown = {
            "cost_score":    round(reward.cost_efficiency, 4),
            "delay_score":   round(reward.delivery_reward, 4),
            "sla_score":     round(reward.sla_compliance, 4),
            "disruption":    round(reward.disruption_penalty, 4),
            "final_reward":  round(reward.total, 4),
        }

        reward_series.append(reward.total)
        steps_executed += 1

        # ── Step record ──────────────────────────────────────
        step_record = {
            "step":           step,
            "day":            day,
            "order_id":       order["order_id"],
            "sla_tier":       order["sla_tier"],
            "demand_node":    order["demand_node"],
            "slack_days":     order["deadline_day"] - day,
            "active_disruptions": len(obs_dict.get("active_disruptions", [])),
            "agent_decisions": {
                "producer": {
                    "recommended_supplier": ma_step.producer.recommended_supplier,
                    "buffer_signal":        round(ma_step.producer.buffer_signal, 2),
                    "reason":               ma_step.producer.reason,
                    "confidence":           round(ma_step.producer.confidence, 2),
                },
                "warehouse": {
                    "target_warehouse":  ma_step.warehouse.target_warehouse,
                    "pre_position":      ma_step.warehouse.pre_position,
                    "safety_stock_days": ma_step.warehouse.safety_stock_days,
                    "reason":            ma_step.warehouse.reason,
                    "confidence":        round(ma_step.warehouse.confidence, 2),
                },
                "logistics": {
                    "routing_decision":      decision.routing_decision,
                    "influenced_by_producer": decision.influenced_by_producer,
                    "influenced_by_warehouse": decision.influenced_by_warehouse,
                    "reasoning":             decision.reasoning,
                },
            },
            "agent_interactions":  ma_step.agent_interactions,
            "reward_breakdown":    reward_breakdown,
            "on_time":             on_time,
            "adaptive_strategy":   strategy.describe(),
            "budget_remaining":    round(obs.budget_remaining, 2),
        }
        per_step_trace.append(step_record)

        # ── Verbose CLI output ───────────────────────────────
        if verbose and not emit_json:
            disrupt_str = (
                f" ⚠ {len(obs_dict['active_disruptions'])} disruption(s)"
                if obs_dict["active_disruptions"] else ""
            )
            slack = order["deadline_day"] - day
            print(f"\n{'─'*68}")
            print(
                f"  STEP {step:>2} │ Day {day:>2} │ [{order['sla_tier'].upper():8}] "
                f"{order['order_id']} → {order['demand_node']}  slack={slack}d{disrupt_str}"
            )
            print(f"{'─'*68}")
            print(f"  ProducerAgent  → {ma_step.producer.recommended_supplier}  "
                  f"buffer={ma_step.producer.buffer_signal:.0%}  "
                  f"conf={ma_step.producer.confidence:.0%}")
            print(f"    └ {ma_step.producer.reason}")
            print(f"  WarehouseAgent → {ma_step.warehouse.target_warehouse}  "
                  f"pre_position={ma_step.warehouse.pre_position}  "
                  f"safety={ma_step.warehouse.safety_stock_days}d")
            print(f"    └ {ma_step.warehouse.reason}")
            print(f"  LogisticsAgent → ROUTING: {decision.routing_decision.upper()}")
            print(f"    └ {decision.reasoning[:100]}")
            if ma_step.agent_interactions:
                for ia in ma_step.agent_interactions:
                    print(f"    ↳ {ia}")
            print(
                f"  Reward │ cost={reward_breakdown['cost_score']:+.3f} "
                f"delay={reward_breakdown['delay_score']:+.3f} "
                f"sla={reward_breakdown['sla_score']:+.3f} "
                f"final={reward_breakdown['final_reward']:+.3f}"
            )
            outcome = "✓ ON TIME" if on_time else "✗ LATE"
            print(f"  Outcome: {outcome} │ Budget remaining: ${obs.budget_remaining:,.0f}")
            print(f"  Strategy: {strategy.describe()}")

    # ── Episode summary ──────────────────────────────────────
    final_state = env.state()
    action_history = final_state.pop("action_history", [])

    from tasks import TASK_MAP
    task = TASK_MAP[task_id]
    grade_result = grade(task_id, final_state, action_history, task.budget_usd)

    fulfilled   = final_state.get("fulfilled", 0)
    late        = final_state.get("late_deliveries", 0)
    on_time_cnt = fulfilled - late
    cost        = final_state.get("cumulative_cost", 0.0)
    crit_met    = final_state.get("critical_sla_met", 0)
    crit_total  = final_state.get("critical_sla_total", 1)

    episode_summary = {
        "task_id":         task_id,
        "score":           grade_result.score,
        "passed":          grade_result.passed,
        "steps_executed":  steps_executed,
        "fulfilled":       fulfilled,
        "on_time":         on_time_cnt,
        "late":            late,
        "on_time_rate":    round(on_time_cnt / max(fulfilled, 1), 4),
        "critical_sla":    f"{crit_met}/{crit_total}",
        "total_cost_usd":  round(cost, 2),
        "budget_used_pct": round(cost / task.budget_usd * 100, 1),
        "adaptive_adjustments": strategy.adjustments_made,
        "mean_reward":     round(sum(reward_series) / max(len(reward_series), 1), 4),
        "subscores":       grade_result.subscores,
    }

    if verbose and not emit_json:
        print(f"\n{SEPARATOR}")
        print(f"  EPISODE SUMMARY")
        print(SEPARATOR)
        print(f"  Score:          {grade_result.score:.3f} │ {'PASSED ✓' if grade_result.passed else 'FAILED ✗'}")
        print(f"  Orders:         {fulfilled} fulfilled  │  {on_time_cnt} on-time  │  {late} late")
        print(f"  Critical SLA:   {crit_met}/{crit_total}")
        print(f"  Cost:           ${cost:,.0f} ({episode_summary['budget_used_pct']:.1f}% of budget)")
        print(f"  Mean reward:    {episode_summary['mean_reward']:+.3f}")
        print(f"  Adaptive adjustments: {strategy.adjustments_made}")
        print(f"  Subscores:")
        for k, v in grade_result.subscores.items():
            print(f"    {k:<28} {v:.3f}")
        print(SEPARATOR)

    result = {
        "task_id":         task_id,
        "steps_executed":  steps_executed,
        "per_step_trace":  per_step_trace,
        "episode_summary": episode_summary,
        "reward_series":   reward_series,
    }

    if emit_json:
        print(json.dumps(result, indent=2))

    return result


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Multi-Agent Demo")
    parser.add_argument("--task",   default="task_medium",
                        choices=["task_easy", "task_medium", "task_hard"])
    parser.add_argument("--steps",  type=int, default=15,
                        help="Max steps to show in demo (default: 15)")
    parser.add_argument("--json",   action="store_true",
                        help="Emit structured JSON output")
    args = parser.parse_args()

    result = run_demo(
        task_id=args.task,
        max_steps=args.steps,
        emit_json=args.json,
        verbose=not args.json,
    )

    if not args.json:
        print(f"\n[DEMO] Complete — {result['steps_executed']} steps executed")
        print(f"[DEMO] Score: {result['episode_summary']['score']:.3f}")


if __name__ == "__main__":
    main()
