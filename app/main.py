"""
FastAPI application — OpenEnv Supply Chain Disruption Management
All endpoints comply with the OpenEnv specification.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import SupplyChainEnv, SCENARIOS
from grader import grade, PASSING_THRESHOLDS
from models import Action, BaselineResult, GraderResult, Observation, Reward, StepResult, TaskSpec
from tasks import TASK_MAP, TASKS

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain Disruption Management — OpenEnv",
    description=(
        "A production-grade reinforcement learning environment for AI-driven "
        "logistics decision-making under uncertainty. Three tasks across easy→hard "
        "difficulty test an agent's ability to route orders, respond to disruptions, "
        "and balance cost vs SLA compliance."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment registry (one env per task, lazily initialized)
_envs: Dict[str, SupplyChainEnv] = {}


def get_env(task_id: str) -> SupplyChainEnv:
    if task_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    if task_id not in _envs:
        _envs[task_id] = SupplyChainEnv(task_id=task_id)
    return _envs[task_id]


# ─────────────────────────────────────────────────────────────
# Request/Response schemas
# ─────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    task_id: str = "task_easy"
    action: Action


class GraderRequest(BaseModel):
    task_id: str
    final_state: Optional[Dict[str, Any]] = None   # if None, use live env state


class BaselineRunRequest(BaseModel):
    task_id: str = "task_easy"
    max_steps: Optional[int] = None


# ─────────────────────────────────────────────────────────────
# OpenEnv Required Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "version": "1.0.0"}


@app.get("/tasks", response_model=List[TaskSpec])
def list_tasks():
    """List all available tasks with descriptions, objectives, and grading criteria."""
    return TASKS


@app.get("/tasks/{task_id}", response_model=TaskSpec)
def get_task(task_id: str):
    if task_id not in TASK_MAP:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TASK_MAP[task_id]


@app.post("/reset", response_model=Observation)
def reset(task_id: str = Query("task_easy")):
    """
    Reset the environment to its initial state for the given task.
    Returns the initial observation.
    """
    env = get_env(task_id)
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    """
    Apply one action to the environment.
    Returns (observation, reward, done, info).
    The reward is decomposed into sub-components for interpretability.
    """
    env = get_env(request.task_id)
    result = env.step(request.action)
    return result


@app.get("/state")
def get_state(task_id: str = Query("task_easy")):
    """Return current internal state (for debugging and grading)."""
    env = get_env(task_id)
    return env.state()


@app.get("/observation", response_model=Observation)
def get_observation(task_id: str = Query("task_easy")):
    """Return current observation without taking a step."""
    env = get_env(task_id)
    return env._build_observation()


# ─────────────────────────────────────────────────────────────
# Grader Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/grader", response_model=GraderResult)
def run_grader(request: GraderRequest):
    """
    Score the agent's performance deterministically (0.0–1.0).
    Uses the current environment state if final_state is not provided.
    """
    env = get_env(request.task_id)
    task = TASK_MAP[request.task_id]

    if request.final_state:
        state = request.final_state
        action_history = state.pop("action_history", [])
    else:
        state = env.state()
        action_history = state.pop("action_history", [])

    return grade(request.task_id, state, action_history, task.budget_usd)


@app.get("/grader/{task_id}", response_model=GraderResult)
def grade_current(task_id: str):
    """Grade the current state of a running task."""
    env = get_env(task_id)
    task = TASK_MAP[task_id]
    state = env.state()
    action_history = state.pop("action_history", [])
    return grade(task_id, state, action_history, task.budget_usd)


# ─────────────────────────────────────────────────────────────
# Baseline Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResult)
def run_baseline_endpoint(request: BaselineRunRequest):
    """
    Run the heuristic baseline agent (no LLM required).
    Returns deterministic scores for reproducibility.
    Uses a cost-priority rule: always choose cheapest route that meets the deadline.
    """
    task_id = request.task_id
    task = TASK_MAP.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    env = SupplyChainEnv(task_id=task_id)
    obs = env.reset()
    max_steps = request.max_steps or task.max_steps

    steps = 0
    reasoning_samples: List[str] = []

    while not env._is_done() and steps < max_steps:
        obs_dict = obs.model_dump()
        orders = obs_dict["pending_orders"]
        if not orders:
            # All orders dispatched for this day — advance to next day
            obs = env._build_observation()
            env._advance_day()
            obs = env._build_observation()
            steps += 1
            continue

        # Heuristic: sort by urgency (sla_tier × deadline slack)
        sla_priority = {"critical": 0, "standard": 1, "flexible": 2}
        orders.sort(key=lambda o: (
            sla_priority[o["sla_tier"]],
            o["deadline_day"] - obs_dict["episode_day"]
        ))

        order = orders[0]
        day = obs_dict["episode_day"]
        slack = order["deadline_day"] - day
        sla = order["sla_tier"]
        disruptions = obs_dict["active_disruptions"]
        budget = obs_dict["budget_remaining"]

        # Decision logic
        if budget < 5000:
            decision = "partial_fulfill"
            reason = "Budget critically low — partial fulfill to preserve funds."
        elif disruptions and sla == "critical" and slack <= 2:
            decision = "spot_market"
            reason = f"Critical order with {slack}d slack during disruption — spot market required."
        elif sla == "critical" and slack <= 2:
            decision = "express_route"
            reason = f"Critical SLA, only {slack}d slack — express route."
        elif disruptions and sla == "standard" and slack <= 3:
            decision = "split_shipment"
            reason = "Disruption active + tight standard deadline — split shipment hedges risk."
        elif slack >= 5:
            decision = "standard_route"
            reason = f"Comfortable {slack}d slack — standard route is most cost-efficient."
        elif sla == "flexible" and slack <= 2:
            decision = "defer_24h"
            reason = "Flexible SLA with minimal slack — defer 24h to wait for better lane."
        else:
            decision = "standard_route"
            reason = "Default: standard route."

        action = Action(
            order_id=order["order_id"],
            routing_decision=decision,  # type: ignore
            alternate_supplier=None,
            reasoning=reason,
        )

        if steps < 3:
            reasoning_samples.append(f"[Day {day}][{decision}] {reason}")

        # Also capture first premium-route decision (shows disruption handling)
        if decision in ("spot_market", "express_route", "split_shipment") \
                and not any("spot_market" in r or "express_route" in r or "split_shipment" in r
                            for r in reasoning_samples):
            reasoning_samples.append(f"[Day {day}][PREMIUM:{decision}] {reason}")

        result = env.step(action)
        obs = result.observation
        steps += 1

    final_state = env.state()
    action_history = final_state.pop("action_history", [])
    grade_result = grade(task_id, final_state, action_history, task.budget_usd)

    return BaselineResult(
        task_id=task_id,
        agent="heuristic-baseline",
        score=grade_result.score,
        steps_taken=steps,
        total_cost=final_state.get("cumulative_cost", 0.0),
        on_time_rate=obs.on_time_delivery_rate,
        sla_met=obs.service_level,
        reasoning_samples=reasoning_samples,
    )


@app.get("/baseline/{task_id}", response_model=BaselineResult)
def baseline_get(task_id: str):
    """GET convenience wrapper for baseline on a given task."""
    return run_baseline_endpoint(BaselineRunRequest(task_id=task_id))


# ─────────────────────────────────────────────────────────────
# Info / Documentation
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Welcome endpoint — overview of the environment and quick-start pointers.
    For a live walkthrough, call GET /demo?task_id=task_easy
    """
    return {
        "name": "Supply Chain Disruption Management — OpenEnv",
        "version": "1.1.0",
        "tagline": (
            "Train AI to make real-time logistics decisions under uncertainty: "
            "route orders, respond to disruptions, balance cost vs SLA compliance."
        ),
        "quick_start": {
            "30_second_demo":    "GET  /demo?task_id=task_easy",
            "full_docs":         "GET  /docs",
            "list_tasks":        "GET  /tasks",
            "run_baseline":      "GET  /baseline/task_medium",
        },
        "tasks": {
            t.task_id: {
                "difficulty": t.difficulty,
                "days": SCENARIOS[t.task_id]["max_days"],
                "budget_usd": t.budget_usd,
                "passing_score": t.passing_score,
                "name": t.name,
            }
            for t in TASKS
        },
        "action_space": {
            "standard_route":    "1.0× cost, 1.0× transit — cheapest, use when slack ≥ 4d",
            "express_route":     "2.8× cost, 0.35× transit — fast, for tight critical deadlines",
            "spot_market":       "4.5× cost × spot_premium, 0.25× transit — bypasses disruptions",
            "split_shipment":    "1.6× cost, 0.8× transit — hedges across two paths",
            "defer_24h":         "0× cost now, +1d — gamble on disruption clearing",
            "defer_48h":         "0× cost now, +2d — only for flexible SLA",
            "source_alternative":"1.3× cost, 1.2× transit — switch to backup supplier",
            "partial_fulfill":   "0.5× cost — last resort when budget is critical",
        },
        "reward_formula": "0.35×delivery + 0.25×cost_efficiency + 0.30×sla_compliance + 0.10×disruption_penalty",
    }


@app.get("/demo")
def demo(task_id: str = Query("task_easy", description="Task to demonstrate: task_easy, task_medium, task_hard")):
    """
    30-second judge demo: runs reset → 3 representative steps → grade.
    Shows the full decision cycle with rich reward reasoning.
    No setup required — deterministic and reproducible.
    """
    if task_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    task = TASK_MAP[task_id]
    env = SupplyChainEnv(task_id=task_id)
    obs = env.reset()

    demo_steps = []
    steps_taken = 0
    max_demo_steps = 5   # run enough to show variety

    while not env._is_done() and steps_taken < max_demo_steps:
        obs_dict = obs.model_dump()
        orders = obs_dict["pending_orders"]
        if not orders:
            env._advance_day()
            obs = env._build_observation()
            steps_taken += 1
            continue

        # Heuristic decision (same as /baseline)
        sla_priority = {"critical": 0, "standard": 1, "flexible": 2}
        orders.sort(key=lambda o: (sla_priority[o["sla_tier"]], o["deadline_day"] - obs_dict["episode_day"]))
        order = orders[0]
        day = obs_dict["episode_day"]
        slack = order["deadline_day"] - day
        sla = order["sla_tier"]
        disruptions = obs_dict["active_disruptions"]
        budget = obs_dict["budget_remaining"]

        if budget < 5000:
            decision, reason = "partial_fulfill", "Budget critically low — preserving funds for future critical orders."
        elif disruptions and sla == "critical" and slack <= 2:
            decision, reason = "spot_market", f"CRITICAL order with only {slack}d slack during active disruption — spot market guarantees delivery."
        elif sla == "critical" and slack <= 2:
            decision, reason = "express_route", f"CRITICAL SLA with {slack}d slack — express route reduces transit by 65%."
        elif disruptions and sla == "standard" and slack <= 3:
            decision, reason = "split_shipment", "Disruption active + tight standard deadline — split shipment hedges risk across two paths."
        elif slack >= 5:
            decision, reason = "standard_route", f"{slack}d slack is comfortable — standard route is most budget-efficient."
        elif sla == "flexible" and slack <= 2:
            decision, reason = "defer_24h", f"Flexible SLA — defer 24h to let disruption clear, saving premium cost."
        else:
            decision, reason = "standard_route", f"Sufficient slack ({slack}d) for standard routing."

        action = Action(order_id=order["order_id"], routing_decision=decision, alternate_supplier=None, reasoning=reason)  # type: ignore
        result = env.step(action)

        demo_steps.append({
            "step": steps_taken + 1,
            "day": day,
            "order": {
                "id": order["order_id"],
                "sku": order["sku"],
                "units": order["units_required"],
                "destination": order["demand_node"],
                "sla_tier": sla,
                "deadline_day": order["deadline_day"],
                "deadline_slack_days": slack,
            },
            "disruptions_active": [
                {"type": d["event_type"], "severity": d["severity"]}
                for d in disruptions
            ],
            "decision": decision,
            "decision_rationale": reason,
            "reward": {
                "total": result.reward.total,
                "delivery": result.reward.delivery_reward,
                "cost_efficiency": result.reward.cost_efficiency,
                "sla_compliance": result.reward.sla_compliance,
                "disruption_penalty": result.reward.disruption_penalty,
                "reasoning": result.reward.reasoning,
            },
            "outcome": result.info.get("last_decision", {}),
            "budget_remaining": result.info.get("budget_remaining"),
            "on_time_rate_so_far": result.info.get("on_time_rate"),
        })

        obs = result.observation
        steps_taken += 1

    # Final grade
    final_state = env.state()
    action_history = final_state.pop("action_history", [])
    grade_result = grade(task_id, final_state, action_history, task.budget_usd)

    return {
        "demo_task": task_id,
        "task_name": task.name,
        "difficulty": task.difficulty,
        "environment_overview": {
            "problem": (
                "AI must route logistics orders across a global supply chain "
                "under disruptions (port closures, weather, carrier strikes), "
                "balancing cost vs delivery speed vs SLA compliance."
            ),
            "network": "3 suppliers -> 3 warehouses -> 3 demand nodes | 12 freight lanes",
            "action_space": "8 decisions: standard/express/spot_market/split/defer_24h/defer_48h/source_alternative/partial_fulfill",
            "reward_formula": "0.35*delivery + 0.25*cost_efficiency + 0.30*sla_compliance + 0.10*disruption_penalty",
            "reward_range": "[-1.0, 1.0]  (positive = good decision, negative = harmful)",
        },
        "scenario": {
            "days": SCENARIOS[task_id]["max_days"],
            "budget_usd": SCENARIOS[task_id]["budget_usd"],
            "disruption_probability_per_day": SCENARIOS[task_id]["disruption_probability"],
            "spot_market_baseline_premium": SCENARIOS[task_id]["spot_market_baseline"],
        },
        "reward_guide": {
            "delivery_performance": "+1.0 on-time, -0.25 per day late (35% weight)",
            "cost_efficiency":      "+1.0 standard cost, penalised for premium routes (25% weight)",
            "sla_compliance":       "+1.0 critical on-time / -1.0 critical late (30% weight)",
            "disruption_penalty":   "-0.4 if route was disrupted, -0.5 reckless deferral (10% weight)",
        },
        "steps_shown": len(demo_steps),
        "demo_steps": demo_steps,
        "partial_grade": {
            "score_so_far": grade_result.score,
            "passed": grade_result.passed,
            "threshold": task.passing_score,
            "subscores": grade_result.subscores,
        },
        "next_steps": {
            "continue_episode": "POST /step  body: {task_id, action: {order_id, routing_decision}}",
            "run_full_baseline": f"GET  /baseline/{task_id}",
            "get_full_grade":    f"GET  /grader/{task_id}",
            "api_docs":          "GET  /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


# ─────────────────────────────────────────────────────────────
# Multi-Agent Endpoints (NEW — hackathon upgrade)
# ─────────────────────────────────────────────────────────────

class MultiAgentStepRequest(BaseModel):
    task_id: str = "task_easy"
    order: Optional[Dict[str, Any]] = None   # if None, auto-select most urgent


@app.post("/multi-agent/step")
def multi_agent_step(request: MultiAgentStepRequest):
    """
    Run one multi-agent decision step.
    Coordinates ProducerAgent, WarehouseAgent, and LogisticsAgent.
    Returns structured decisions from all three agents + reward breakdown.
    """
    from agents import MultiAgentCoordinator, AdaptiveStrategy

    env = get_env(request.task_id)
    obs_dict = env._build_observation().model_dump()
    orders   = obs_dict.get("pending_orders", [])

    if not orders:
        raise HTTPException(status_code=400, detail="No pending orders to act on. Call /reset or advance the day.")

    # Select order
    if request.order:
        order = request.order
    else:
        day = obs_dict["episode_day"]
        sla_pri = {"critical": 0, "standard": 1, "flexible": 2}
        order   = min(orders, key=lambda o: (sla_pri[o["sla_tier"]], o["deadline_day"] - day))

    coordinator = MultiAgentCoordinator()
    ma_step     = coordinator.coordinate(obs_dict, order)
    decision    = ma_step.logistics

    # Execute action
    from models import Action
    action = Action(
        order_id=decision.order_id,
        routing_decision=decision.routing_decision,  # type: ignore
        alternate_supplier=decision.alternate_supplier,
        reasoning=decision.reasoning,
    )
    result = env.step(action)
    reward = result.reward

    return {
        "order": {
            "order_id":    order.get("order_id"),
            "sla_tier":    order.get("sla_tier"),
            "demand_node": order.get("demand_node"),
            "slack_days":  order.get("deadline_day", 0) - obs_dict["episode_day"],
        },
        "agent_decisions": {
            "producer_agent": {
                "recommended_supplier": ma_step.producer.recommended_supplier,
                "buffer_signal":        round(ma_step.producer.buffer_signal, 3),
                "reason":               ma_step.producer.reason,
                "confidence":           round(ma_step.producer.confidence, 3),
            },
            "warehouse_agent": {
                "target_warehouse":  ma_step.warehouse.target_warehouse,
                "pre_position":      ma_step.warehouse.pre_position,
                "safety_stock_days": ma_step.warehouse.safety_stock_days,
                "reason":            ma_step.warehouse.reason,
                "confidence":        round(ma_step.warehouse.confidence, 3),
            },
            "logistics_agent": {
                "routing_decision":        decision.routing_decision,
                "influenced_by_producer":  decision.influenced_by_producer,
                "influenced_by_warehouse": decision.influenced_by_warehouse,
                "reasoning":               decision.reasoning,
            },
        },
        "agent_interactions":  ma_step.agent_interactions,
        "reward_breakdown": {
            "cost_score":   round(reward.cost_efficiency, 4),
            "delay_score":  round(reward.delivery_reward, 4),
            "sla_score":    round(reward.sla_compliance, 4),
            "disruption":   round(reward.disruption_penalty, 4),
            "final_reward": round(reward.total, 4),
            "reasoning":    reward.reasoning,
        },
        "done":            result.done,
        "budget_remaining": round(result.observation.budget_remaining, 2),
        "on_time_rate":     round(result.observation.on_time_delivery_rate, 4),
    }


@app.get("/multi-agent/demo")
def multi_agent_demo(task_id: str = Query("task_medium"), steps: int = Query(10)):
    """
    Run a full multi-agent demo episode.
    Returns structured step-by-step trace with all agent decisions and rewards.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)) + "/..")
    from demo import run_demo
    result = run_demo(task_id=task_id, max_steps=min(steps, 20), emit_json=False, verbose=False)
    return result


@app.get("/reward-breakdown/{task_id}")
def reward_breakdown(task_id: str):
    """
    Return the current reward component breakdown for the active episode.
    Shows cost_score, delay_score, sla_score with running averages.
    """
    env   = get_env(task_id)
    state = env.state()
    history = state.get("action_history", [])

    if not history:
        return {"message": "No actions taken yet.", "task_id": task_id}

    rewards  = [a.get("reward", 0) for a in history]
    on_times = [a.get("on_time", False) for a in history]
    costs    = [a.get("cost_usd", 0) for a in history]

    return {
        "task_id":         task_id,
        "steps_taken":     len(history),
        "reward_summary": {
            "mean_final_reward": round(sum(rewards) / len(rewards), 4),
            "max_reward":        round(max(rewards), 4),
            "min_reward":        round(min(rewards), 4),
            "cumulative_reward": round(sum(rewards), 4),
        },
        "kpi_summary": {
            "on_time_rate":   round(sum(on_times) / len(on_times), 4),
            "mean_cost_usd":  round(sum(costs) / len(costs), 2),
            "total_cost_usd": round(sum(costs), 2),
        },
        "last_5_rewards": [round(r, 4) for r in rewards[-5:]],
        "formula": "final = 0.35*delay_score + 0.25*cost_score + 0.30*sla_score + 0.10*disruption",
    }
