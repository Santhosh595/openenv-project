"""
OpenEnv Supply Chain Disruption Management — Typed Models
Fully compliant with OpenEnv specification.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import time


class NodeStatus(BaseModel):
    node_id: str
    node_type: Literal["supplier", "warehouse", "demand"]
    inventory: float
    capacity: float
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    is_disrupted: bool = False
    disruption_severity: float = Field(0.0, ge=0.0, le=1.0)
    lat: float
    lon: float


class ShipmentLane(BaseModel):
    lane_id: str
    from_node: str
    to_node: str
    base_cost_per_unit: float
    transit_days: int
    capacity_units: int
    utilization: float = Field(..., ge=0.0, le=1.0)
    is_active: bool = True
    weather_risk: float = Field(0.0, ge=0.0, le=1.0)
    congestion_level: float = Field(0.0, ge=0.0, le=1.0)


class PendingOrder(BaseModel):
    order_id: str
    sku: str
    units_required: int
    demand_node: str
    deadline_day: int
    sla_tier: Literal["critical", "standard", "flexible"]
    penalty_per_day_late: float
    value_usd: float


class DisruptionEvent(BaseModel):
    event_id: str
    event_type: Literal[
        "port_closure", "weather", "supplier_failure",
        "geopolitical", "demand_spike", "carrier_strike"
    ]
    affected_nodes: List[str]
    affected_lanes: List[str]
    started_day: int
    estimated_duration_days: int
    severity: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)


class Observation(BaseModel):
    """
    Full environment state returned after every reset() and step().
    Contains everything an agent needs to make the next routing decision.
    """
    episode_day: int = Field(..., description="Current simulation day (0-indexed).")
    max_days: int = Field(..., description="Total episode length in days.")
    nodes: List[NodeStatus] = Field(..., description="All 9 network nodes: 3 suppliers, 3 warehouses, 3 demand sites.")
    lanes: List[ShipmentLane] = Field(..., description="All 12 freight lanes. Disrupted lanes have is_active=False.")
    pending_orders: List[PendingOrder] = Field(..., description="Orders awaiting a routing decision this step.")
    active_disruptions: List[DisruptionEvent] = Field(..., description="Live disruption events. Each blocks lanes and raises costs.")
    budget_remaining: float = Field(..., description="USD remaining in episode budget. Reaching 0 ends the episode.")
    cumulative_cost: float = Field(..., description="Total USD spent on routing decisions so far.")
    on_time_delivery_rate: float = Field(..., ge=0.0, le=1.0, description="Running fraction of orders delivered by deadline (key KPI).")
    service_level: float = Field(..., ge=0.0, le=1.0, description="Fraction of CRITICAL-tier SLA orders met on time.")
    weather_forecast: Dict[str, float] = Field(..., description="Per-node storm probability for the next 3 days (0.0=clear, 1.0=severe).")
    spot_market_premium: float = Field(..., description="Current emergency carrier cost multiplier (rises with active disruptions).")
    fulfilled_orders: int = Field(..., description="Total orders dispatched (includes late deliveries).")
    total_orders: int = Field(..., description="Total orders generated in the episode so far.")


class Action(BaseModel):
    """
    A single routing decision: which order to route, and how.
    The agent must submit one Action per pending order.
    """
    order_id: str = Field(..., description="The order_id from pending_orders to act on.")
    routing_decision: Literal[
        "standard_route",      # cheapest path, full transit time — use when slack ≥ 4d
        "express_route",       # 2.8× cost, 65% faster — use for tight critical deadlines
        "spot_market",         # 4.5× × spot_premium, fastest — bypasses disruptions
        "split_shipment",      # 1.6× cost, 20% faster — hedges across two paths
        "defer_24h",           # free now, +1d delay — gamble on disruption clearing
        "defer_48h",           # free now, +2d delay — only for flexible SLA
        "source_alternative",  # 1.3× cost, +20% transit — switch to backup supplier
        "partial_fulfill",     # 0.5× cost — send available stock, backorder remainder
    ]
    alternate_supplier: Optional[str] = Field(None, description="Required when routing_decision=source_alternative.")
    reasoning: Optional[str] = Field(None, description="Agent's plain-English explanation. Included in reward output.")


class Reward(BaseModel):
    """
    Dense decomposed reward returned after every step().
    Total is a weighted combination of four real-world logistics KPIs.
    Range: [-1.0, 1.0]. Positive = good decision; negative = harmful decision.
    """
    total: float = Field(..., description="Composite reward: 0.35×delivery + 0.25×cost_efficiency + 0.30×sla_compliance + 0.10×disruption_penalty.")
    delivery_reward: float = Field(..., description="+1.0 if on-time; decreases by 0.25 per day late; min -1.0.")
    cost_efficiency: float = Field(..., description="+1.0 for standard-route cost; decreases by 0.4 for each cost multiplier above baseline.")
    sla_compliance: float = Field(..., description="+1.0 critical on-time; -1.0 critical late; scaled for standard/flexible.")
    disruption_penalty: float = Field(..., description="-0.4 if route disrupted; -0.5 for reckless deferral past deadline.")
    reasoning: str = Field(..., description="Plain-English explanation of why this reward was earned: trade-offs, alternatives, budget impact.")


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GraderResult(BaseModel):
    """
    Deterministic episode score returned by /grader. Always reproducible given the same seed and actions.
    """
    task_id: str = Field(..., description="Which task was graded.")
    score: float = Field(..., ge=0.0, le=1.0, description="Composite episode score (0.0–1.0). See subscores for breakdown.")
    subscores: Dict[str, float] = Field(..., description="Per-criterion scores that make up the composite. Weights vary by task difficulty.")
    passed: bool = Field(..., description="True if score >= task's passing_score threshold.")
    explanation: str = Field(..., description="Human-readable breakdown: criterion scores, weights, and performance summary.")
    timestamp: float = Field(default_factory=time.time)


class TaskSpec(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    objective: str
    max_steps: int
    budget_usd: float
    passing_score: float
    grading_criteria: Dict[str, float] = Field(..., description="Criterion name → weight. Weights sum to 1.0.")


class BaselineResult(BaseModel):
    """
    Result from running the built-in heuristic baseline agent.
    Provides a reproducible performance floor for comparison.
    """
    task_id: str
    agent: str = Field(..., description="Agent type: 'heuristic-baseline' or 'gpt-4o-mini'.")
    score: float = Field(..., description="Final graded score (0.0–1.0).")
    steps_taken: int
    total_cost: float = Field(..., description="Total USD spent across the episode.")
    on_time_rate: float = Field(..., description="Fraction of orders delivered by deadline.")
    sla_met: float = Field(..., description="Fraction of CRITICAL-tier SLAs met.")
    reasoning_samples: List[str] = Field(..., description="Sample decisions with agent rationale — spread across episode stages.")
