"""Tests for the supply-chain disruption environment.

The environment is the part of this project that a trained policy is
graded against, so its behaviour has to be pinned down: the same seed
must produce the same episode, the cost and transit tables must match
what the README claims, and a malformed action must be penalised rather
than silently accepted.

The environment is seeded (`random.Random(scenario["seed"])`) and does
no network I/O, so the whole suite runs offline and deterministically.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from environment import (
    DISRUPTION_TEMPLATES,
    ROUTING_COST_MULTIPLIER,
    ROUTING_TRANSIT_MULTIPLIER,
    SCENARIOS,
    SupplyChainEnv,
)
from models import Action


# --------------------------------------------------------------------------
# Scenario and action-space definitions
# --------------------------------------------------------------------------


def test_three_scenarios_are_defined() -> None:
    assert set(SCENARIOS) == {"task_easy", "task_medium", "task_hard"}


def test_there_are_exactly_five_disruption_types() -> None:
    # The README says "5-type disruption engine". This pins that number.
    assert len(DISRUPTION_TEMPLATES) == 5
    assert {d["event_type"] for d in DISRUPTION_TEMPLATES} == {
        "port_closure",
        "weather",
        "supplier_failure",
        "carrier_strike",
        "geopolitical",
    }


def test_action_space_has_eight_decisions() -> None:
    # The README says "8-action decision space".
    assert len(ROUTING_COST_MULTIPLIER) == 8
    assert set(ROUTING_COST_MULTIPLIER) == set(ROUTING_TRANSIT_MULTIPLIER)
    assert set(ROUTING_COST_MULTIPLIER) == {
        "standard_route",
        "express_route",
        "spot_market",
        "split_shipment",
        "defer_24h",
        "defer_48h",
        "source_alternative",
        "partial_fulfill",
    }


@pytest.mark.parametrize(
    "decision, cost, transit",
    [
        ("standard_route", 1.0, 1.0),
        ("express_route", 2.8, 0.35),
        ("spot_market", 4.5, 0.25),
        ("split_shipment", 1.6, 0.80),
        ("defer_24h", 0.0, 1.0),
        ("defer_48h", 0.0, 1.0),
        ("source_alternative", 1.3, 1.2),
        ("partial_fulfill", 0.5, 1.0),
    ],
)
def test_cost_and_transit_multipliers_match_the_documented_table(
    decision: str, cost: float, transit: float
) -> None:
    assert ROUTING_COST_MULTIPLIER[decision] == cost
    assert ROUTING_TRANSIT_MULTIPLIER[decision] == transit


# --------------------------------------------------------------------------
# Construction and reset
# --------------------------------------------------------------------------


def test_unknown_task_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown task_id"):
        SupplyChainEnv("task_nonexistent")


@pytest.mark.parametrize("task_id", ["task_easy", "task_medium", "task_hard"])
def test_environment_resets_with_pending_orders(task_id: str) -> None:
    env = SupplyChainEnv(task_id)
    env.reset()
    assert env._state["pending_orders"], "a fresh episode must have orders to route"
    assert env._state["episode_day"] == 0
    assert env._state["active_disruptions"] == []


def test_reset_is_deterministic_for_the_same_seed() -> None:
    # Same seed, same initial orders. The policy is graded on this
    # reproducibility, so it must not drift.
    first = SupplyChainEnv("task_easy").reset()
    second = SupplyChainEnv("task_easy").reset()
    assert first == second


def test_reset_clears_state_from_a_prior_episode() -> None:
    env = SupplyChainEnv("task_easy")
    env.reset()
    before = len(env._state["pending_orders"])
    env._state["cumulative_cost"] = 999.0
    env.reset()
    assert env._state["cumulative_cost"] == 0.0
    assert len(env._state["pending_orders"]) == before


# --------------------------------------------------------------------------
# Stepping the environment
# --------------------------------------------------------------------------


def _first_order(env: SupplyChainEnv):
    return env._state["pending_orders"][0]


def test_stepping_removes_the_routed_order() -> None:
    env = SupplyChainEnv("task_easy")
    env.reset()
    order = _first_order(env)
    result = env.step(Action(order_id=order.order_id, routing_decision="standard_route"))
    assert result is not None
    assert order.order_id not in {o.order_id for o in env._state["pending_orders"]}


def test_action_history_records_the_decision() -> None:
    env = SupplyChainEnv("task_easy")
    env.reset()
    order = _first_order(env)
    env.step(Action(order_id=order.order_id, routing_decision="standard_route"))
    assert env._state["action_history"], "the decision must be recorded"
    decision = env._state["action_history"][-1].get("decision")
    assert decision == "standard_route"


def test_invalid_order_id_is_penalised() -> None:
    env = SupplyChainEnv("task_easy")
    env.reset()
    result = env.step(Action(order_id="does-not-exist", routing_decision="standard_route"))
    assert result.reward.total == pytest.approx(-0.5)
    assert result.info.get("error") == "invalid_order_id"


def test_the_same_actions_produce_the_same_episode() -> None:
    # The grader is deterministic given a seed and a sequence of actions,
    # so two identical rollouts must agree.

    def rollout() -> list[float]:
        env = SupplyChainEnv("task_easy")
        env.reset()
        rewards = []
        for _ in range(10):
            if not env._state["pending_orders"]:
                env.reset()
            order = _first_order(env)
            result = env.step(
                Action(order_id=order.order_id, routing_decision="standard_route")
            )
            rewards.append(result.reward.total)
        return rewards

    assert rollout() == rollout()


def test_cumulative_cost_never_goes_negative() -> None:
    env = SupplyChainEnv("task_easy")
    env.reset()
    for _ in range(5):
        if not env._state["pending_orders"]:
            env.reset()
        order = _first_order(env)
        env.step(Action(order_id=order.order_id, routing_decision="standard_route"))
    assert env._state["cumulative_cost"] >= 0.0


def test_budget_remaining_stays_within_bounds() -> None:
    env = SupplyChainEnv("task_easy")
    env.reset()
    for _ in range(5):
        if not env._state["pending_orders"]:
            env.reset()
        order = _first_order(env)
        env.step(Action(order_id=order.order_id, routing_decision="standard_route"))
    assert env._state["budget_remaining"] <= SCENARIOS["task_easy"]["budget_usd"]


# --------------------------------------------------------------------------
# Action model
# --------------------------------------------------------------------------


def test_action_rejects_an_unknown_routing_decision() -> None:
    with pytest.raises(Exception):
        Action(order_id="x", routing_decision="teleport")


def test_action_requires_an_order_id() -> None:
    with pytest.raises(Exception):
        Action(routing_decision="standard_route")
