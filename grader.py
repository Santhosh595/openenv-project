"""
Deterministic Grader System — scores agent performance 0.0–1.0
Each task has its own rubric with weighted sub-scores.
"""

from __future__ import annotations
from typing import Dict, Any
from models import GraderResult


# ─────────────────────────────────────────────────────────────
# Grading rubrics per task
# ─────────────────────────────────────────────────────────────

RUBRICS: Dict[str, Dict[str, float]] = {
    "task_easy": {
        "on_time_delivery_rate": 0.40,
        "budget_utilization":    0.20,   # staying within budget
        "mean_reward":           0.30,
        "sla_compliance":        0.10,
    },
    "task_medium": {
        "on_time_delivery_rate": 0.30,
        "budget_utilization":    0.20,
        "mean_reward":           0.20,
        "sla_compliance":        0.20,
        "disruption_adaptation": 0.10,   # quality of decisions during disruptions
    },
    "task_hard": {
        "on_time_delivery_rate": 0.25,
        "budget_utilization":    0.15,
        "mean_reward":           0.15,
        "sla_compliance":        0.25,
        "disruption_adaptation": 0.15,
        "cost_per_unit":         0.05,   # efficiency under tight budget
    },
}

PASSING_THRESHOLDS: Dict[str, float] = {
    "task_easy":   0.70,
    "task_medium": 0.60,
    "task_hard":   0.50,
}


def grade(task_id: str, final_state: Dict[str, Any],
          action_history: list, budget_usd: float) -> GraderResult:
    """
    Deterministic grader — all inputs come from the environment's state().
    Returns a GraderResult with sub-scores and explanation.
    """
    if task_id not in RUBRICS:
        return GraderResult(
            task_id=task_id,
            score=0.0,
            subscores={},
            passed=False,
            explanation=f"Unknown task_id: {task_id}",
        )

    rubric = RUBRICS[task_id]
    fulfilled  = final_state.get("fulfilled", 0)
    late       = final_state.get("late_deliveries", 0)
    cost       = final_state.get("cumulative_cost", 0.0)
    crit_met   = final_state.get("critical_sla_met", 0)
    crit_total = max(final_state.get("critical_sla_total", 1), 1)
    disruptions_during = final_state.get("active_disruptions", 0)

    # ── Sub-score calculations ─────────────────────────────

    subscores: Dict[str, float] = {}

    # 1. On-time delivery rate
    on_time_rate = (fulfilled - late) / max(fulfilled, 1)
    subscores["on_time_delivery_rate"] = round(max(0.0, min(1.0, on_time_rate)), 4)

    # 2. Budget utilization — reward staying within budget efficiently
    #    Score 1.0 if 50–95% used; penalize overspend or extreme underspend
    utilization_ratio = cost / max(budget_usd, 1.0)
    if utilization_ratio > 1.0:
        budget_score = max(0.0, 1.0 - 2.0 * (utilization_ratio - 1.0))
    elif utilization_ratio < 0.10:
        budget_score = 0.30   # barely spent → likely skipped decisions
    else:
        # Peak at ~0.75 utilization
        budget_score = 1.0 - abs(utilization_ratio - 0.75) * 1.4
        budget_score = max(0.0, min(1.0, budget_score))
    subscores["budget_utilization"] = round(budget_score, 4)

    # 3. Mean reward across actions
    rewards = [a["reward"] for a in action_history if "reward" in a]
    mean_reward = sum(rewards) / max(len(rewards), 1)
    # Normalize from [-1,1] → [0,1]
    reward_score = (mean_reward + 1.0) / 2.0
    subscores["mean_reward"] = round(max(0.0, min(1.0, reward_score)), 4)

    # 4. SLA compliance (critical orders only)
    sla_score = crit_met / crit_total
    subscores["sla_compliance"] = round(max(0.0, min(1.0, sla_score)), 4)

    # 5. Disruption adaptation (medium/hard only)
    #
    # IMPROVED METRIC: measures three dimensions, all deterministic:
    #   A) Triage quality  — did the agent use premium routes on orders that
    #      actually hit a disrupted lane (disruption_hit=True)?  Good agents
    #      escalate selectively, not reflexively.
    #   B) Outcome quality — what was the on-time rate specifically for orders
    #      that experienced a disruption?  Did premium routing actually save them?
    #   C) Efficiency      — did the agent avoid burning premium routes when the
    #      lane was clear?  Penalises over-spending outside disruption events.
    #
    if "disruption_adaptation" in rubric:
        # Split history into disruption-affected and clean actions
        disrupted_actions  = [a for a in action_history if a.get("disruption_hit")]
        clean_actions      = [a for a in action_history if not a.get("disruption_hit")]
        premium_decisions  = {"spot_market", "express_route",
                               "source_alternative", "split_shipment"}

        # ── A) Triage quality during disruptions ─────────────────
        # Ideal: when hit by a disruption, use a premium route
        if disrupted_actions:
            correct_escalations = sum(
                1 for a in disrupted_actions
                if a.get("decision") in premium_decisions
            )
            triage_score = correct_escalations / len(disrupted_actions)
        else:
            # No disruptions encountered — environment was stable; give neutral
            triage_score = 0.65

        # ── B) Outcome quality on disrupted orders ────────────────
        # On-time rate for orders that experienced a disrupted lane
        if disrupted_actions:
            on_time_disrupted = sum(
                1 for a in disrupted_actions if a.get("on_time")
            )
            outcome_score = on_time_disrupted / len(disrupted_actions)
        else:
            # Infer from overall on-time rate as proxy
            outcome_score = (fulfilled - late) / max(fulfilled, 1)

        # ── C) Efficiency outside disruptions ────────────────────
        # Penalise premium routes on clean (non-disrupted) lanes — unnecessary spend
        if clean_actions:
            unnecessary_premium = sum(
                1 for a in clean_actions
                if a.get("decision") in premium_decisions
            )
            efficiency_score = 1.0 - (unnecessary_premium / len(clean_actions))
            efficiency_score = max(0.0, efficiency_score)
        else:
            efficiency_score = 0.70   # no clean actions recorded

        # ── Weighted composite ────────────────────────────────────
        # Triage (45%) + Outcome (35%) + Efficiency (20%)
        adapt_score = (
            0.45 * triage_score
            + 0.35 * outcome_score
            + 0.20 * efficiency_score
        )
        adapt_score = round(max(0.0, min(1.0, adapt_score)), 4)

        subscores["disruption_adaptation"] = adapt_score

    # 6. Cost per fulfilled unit (hard only)
    if "cost_per_unit" in rubric:
        avg_units = 500  # approximate expected units per order
        target_cost_per_unit = 15.0  # reasonable benchmark
        actual_cpu = cost / max(fulfilled * avg_units, 1)
        cpu_score = max(0.0, 1.0 - (actual_cpu / target_cost_per_unit - 1.0) * 0.5)
        subscores["cost_per_unit"] = round(min(1.0, cpu_score), 4)

    # ── Weighted composite ────────────────────────────────

    weighted_score = sum(
        subscores[criterion] * weight
        for criterion, weight in rubric.items()
        if criterion in subscores
    )
    final_score = round(max(0.0, min(1.0, weighted_score)), 4)
    passing = final_score >= PASSING_THRESHOLDS[task_id]

    # ── Human-readable explanation ────────────────────────

    lines = [f"Task: {task_id} | Final Score: {final_score:.3f} | {'PASSED' if passing else 'FAILED'}"]
    lines.append(f"Threshold to pass: {PASSING_THRESHOLDS[task_id]:.2f}")
    lines.append("Sub-scores:")
    for criterion, weight in rubric.items():
        if criterion in subscores:
            s = subscores[criterion]
            contrib = s * weight
            lines.append(f"  {criterion:<28} {s:.3f} x {weight:.2f} = {contrib:.3f}")
    lines.append(f"Orders fulfilled: {fulfilled} | Late: {late} | On-time: {on_time_rate:.1%}")
    lines.append(f"Critical SLA met: {crit_met}/{crit_total} ({sla_score:.1%})")
    lines.append(f"Total spend: ${cost:,.0f} / ${budget_usd:,.0f} ({utilization_ratio:.1%})")
    lines.append(f"Actions taken: {len(action_history)}")
    # Show disruption breakdown when applicable
    if "disruption_adaptation" in rubric and action_history:
        dis_acts  = [a for a in action_history if a.get("disruption_hit")]
        cln_acts  = [a for a in action_history if not a.get("disruption_hit")]
        prem_set  = {"spot_market","express_route","source_alternative","split_shipment"}
        esc       = sum(1 for a in dis_acts if a.get("decision") in prem_set)
        lines.append(
            f"Disruption adaptation detail: "
            f"{len(dis_acts)} disrupted orders | "
            f"{esc} escalated to premium routes | "
            f"{len(cln_acts)} clean-lane orders"
        )

    return GraderResult(
        task_id=task_id,
        score=final_score,
        subscores=subscores,
        passed=passing,
        explanation="\n".join(lines),
    )
