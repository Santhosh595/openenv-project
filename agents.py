"""
agents.py — Multi-Agent Supply Chain Intelligence Layer
=========================================================
Three cooperative agents with interdependent decisions:

  ProducerAgent    — manages supplier allocation & production signals
  LogisticsAgent   — handles routing decisions & carrier selection
  WarehouseAgent   — controls inventory positioning & buffer strategy

Each agent emits a typed decision. The LogisticsAgent's routing choice is
passed to the environment as the primary Action; the other two agents'
decisions feed into it as soft constraints that bias the final routing
and inform the reward explanation.

This is intentionally minimal — no neural networks required for demo
execution, but the interface is structured so a TRL training loop can
swap in learned policies per agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Typed agent decisions
# ─────────────────────────────────────────────────────────────

@dataclass
class ProducerDecision:
    """ProducerAgent output: which supplier to recommend and buffer signal."""
    recommended_supplier: str          # e.g. "SUP_CN_SHG"
    buffer_signal: float               # 0.0 = no buffer needed, 1.0 = max urgency
    reason: str
    confidence: float                  # 0.0–1.0


@dataclass
class WarehouseDecision:
    """WarehouseAgent output: inventory positioning recommendation."""
    target_warehouse: str              # e.g. "WH_US_LAX"
    pre_position: bool                 # True = stage inventory proactively
    safety_stock_days: int             # recommended buffer in days
    reason: str
    confidence: float


@dataclass
class LogisticsDecision:
    """LogisticsAgent output: the final routing action to take."""
    order_id: str
    routing_decision: str              # must be a valid Action.routing_decision
    alternate_supplier: Optional[str]
    reasoning: str
    influenced_by_producer: bool
    influenced_by_warehouse: bool


@dataclass
class MultiAgentStep:
    """Complete multi-agent decision bundle for one order."""
    producer:  ProducerDecision
    warehouse: WarehouseDecision
    logistics: LogisticsDecision
    agent_interactions: List[str]      # human-readable log of cross-agent influence


# ─────────────────────────────────────────────────────────────
# Agent implementations (rule-based with structured reasoning)
# ─────────────────────────────────────────────────────────────

class ProducerAgent:
    """
    Manages supplier health and recommends sourcing strategy.
    Responsibilities:
      - Track which suppliers are disrupted
      - Signal buffer urgency when disruptions threaten supply
      - Recommend alternative sourcing before stockouts
    """

    # Proximity map: which supplier best serves which demand region
    SUPPLIER_AFFINITY: Dict[str, List[str]] = {
        "DEM_US_CHI": ["SUP_MX_MTY", "SUP_CN_SHG"],
        "DEM_DE_MUC": ["SUP_IN_MUM", "SUP_CN_SHG"],
        "DEM_JP_TYO": ["SUP_CN_SHG", "SUP_IN_MUM"],
    }
    ALL_SUPPLIERS = ["SUP_CN_SHG", "SUP_IN_MUM", "SUP_MX_MTY"]

    def decide(self, obs: dict, order: dict) -> ProducerDecision:
        disrupted_nodes = {
            n["node_id"] for n in obs.get("nodes", [])
            if n.get("is_disrupted") and n.get("node_type") == "supplier"
        }
        demand_node = order.get("demand_node", "")
        preferred = self.SUPPLIER_AFFINITY.get(demand_node, self.ALL_SUPPLIERS)

        # Find best non-disrupted supplier
        for sup in preferred:
            if sup not in disrupted_nodes:
                buffer = 0.3 if obs.get("active_disruptions") else 0.1
                return ProducerDecision(
                    recommended_supplier=sup,
                    buffer_signal=buffer,
                    reason=f"Primary supplier {sup} is healthy; {len(disrupted_nodes)} disrupted.",
                    confidence=0.9 if not disrupted_nodes else 0.7,
                )

        # All preferred suppliers disrupted — escalate
        fallback = next((s for s in self.ALL_SUPPLIERS if s not in disrupted_nodes), self.ALL_SUPPLIERS[0])
        return ProducerDecision(
            recommended_supplier=fallback,
            buffer_signal=0.9,
            reason=f"All preferred suppliers disrupted. Using {fallback} as fallback.",
            confidence=0.4,
        )


class WarehouseAgent:
    """
    Manages inventory positioning and safety stock decisions.
    Responsibilities:
      - Monitor warehouse inventory levels near demand nodes
      - Recommend pre-positioning when disruption risk is high
      - Signal safety stock requirements to LogisticsAgent
    """

    # Which warehouse serves which demand node
    WAREHOUSE_MAP: Dict[str, str] = {
        "DEM_US_CHI": "WH_US_LAX",
        "DEM_DE_MUC": "WH_EU_RTM",
        "DEM_JP_TYO": "WH_SG_SIN",
    }

    def decide(self, obs: dict, order: dict) -> WarehouseDecision:
        demand_node = order.get("demand_node", "")
        target_wh = self.WAREHOUSE_MAP.get(demand_node, "WH_US_LAX")
        disruptions = obs.get("active_disruptions", [])
        budget = obs.get("budget_remaining", 500_000)
        sla = order.get("sla_tier", "standard")

        # Check warehouse node status
        wh_disrupted = any(
            target_wh in d.get("affected_nodes", []) for d in disruptions
        )

        # Pre-position if: high disruption risk, critical SLA, or warehouse is at risk
        disruption_count = len(disruptions)
        pre_position = (
            disruption_count >= 2
            or (disruption_count >= 1 and sla == "critical")
            or wh_disrupted
        )

        # Safety stock days: more buffer when disruptions are active
        safety_stock_days = 3 + disruption_count * 2

        if wh_disrupted:
            reason = (
                f"{target_wh} is affected by disruption. "
                f"Recommend alternative routing to avoid warehouse bottleneck."
            )
            confidence = 0.85
        elif pre_position:
            reason = (
                f"{disruption_count} active disruption(s). Pre-position {safety_stock_days}d "
                f"safety stock at {target_wh} to protect {sla} SLA."
            )
            confidence = 0.75
        else:
            reason = (
                f"Network stable. {target_wh} has adequate inventory. "
                f"Standard {safety_stock_days}d buffer sufficient."
            )
            confidence = 0.9

        return WarehouseDecision(
            target_warehouse=target_wh,
            pre_position=pre_position,
            safety_stock_days=safety_stock_days,
            reason=reason,
            confidence=confidence,
        )


class LogisticsAgent:
    """
    Core routing agent. Takes inputs from ProducerAgent and WarehouseAgent
    and synthesises them into a final routing decision.

    This is the agent that directly interacts with the OpenEnv action space.
    Its decisions are biased by the other two agents' signals, creating
    genuine multi-agent interdependence.
    """

    def decide(
        self,
        obs: dict,
        order: dict,
        producer: ProducerDecision,
        warehouse: WarehouseDecision,
    ) -> LogisticsDecision:
        day = obs.get("episode_day", 0)
        budget = obs.get("budget_remaining", 500_000)
        disruptions = obs.get("active_disruptions", [])
        spot_premium = obs.get("spot_market_premium", 1.0)

        sla = order.get("sla_tier", "standard")
        slack = order.get("deadline_day", day + 5) - day
        order_id = order.get("order_id", "")

        # Track which agents influenced this decision
        influenced_by_producer = False
        influenced_by_warehouse = False
        interactions: List[str] = []

        # ── Base routing via SLA + slack heuristic ───────────
        if budget < 5_000:
            routing = "partial_fulfill"
            reason = "Budget critically low — preserve remaining funds."

        elif disruptions and sla == "critical" and slack <= 2:
            routing = "spot_market"
            reason = f"CRITICAL SLA, {slack}d slack, disruption active → spot_market bypasses blockage."

        elif sla == "critical" and slack <= 2:
            routing = "express_route"
            reason = f"CRITICAL SLA, {slack}d slack → express_route guarantees on-time."

        elif disruptions and sla == "standard" and slack <= 3:
            routing = "split_shipment"
            reason = f"STANDARD SLA, {slack}d slack, disruption active → split hedges risk."

        elif sla == "flexible" and slack >= 5 and not disruptions:
            routing = "defer_48h"
            reason = f"FLEXIBLE SLA, {slack}d slack, no disruptions → safe to defer."

        elif slack >= 5:
            routing = "standard_route"
            reason = f"{slack}d slack — standard route is cost-efficient."

        elif sla == "flexible" and slack <= 2 and not disruptions:
            routing = "defer_24h"
            reason = f"FLEXIBLE SLA, {slack}d slack → defer to preserve budget."

        else:
            routing = "standard_route"
            reason = "Default: standard route balances cost and transit time."

        # ── ProducerAgent influence ─────────────────────────
        if producer.buffer_signal > 0.7 and routing == "standard_route":
            # High buffer urgency from producer → upgrade to source_alternative
            routing = "source_alternative"
            reason += f" | ProducerAgent: high buffer signal ({producer.buffer_signal:.0%}) — switching to {producer.recommended_supplier}."
            influenced_by_producer = True
            interactions.append(
                f"ProducerAgent → LogisticsAgent: buffer_signal={producer.buffer_signal:.2f} "
                f"triggered source_alternative upgrade."
            )

        # ── WarehouseAgent influence ─────────────────────────
        wh_disrupted = any(
            warehouse.target_warehouse in d.get("affected_nodes", [])
            for d in disruptions
        )
        if wh_disrupted and routing in ("standard_route", "split_shipment"):
            routing = "source_alternative" if producer.confidence > 0.5 else "spot_market"
            reason += (
                f" | WarehouseAgent: {warehouse.target_warehouse} disrupted — "
                f"bypassing warehouse via {routing}."
            )
            influenced_by_warehouse = True
            interactions.append(
                f"WarehouseAgent → LogisticsAgent: {warehouse.target_warehouse} blocked, "
                f"escalated to {routing}."
            )

        if warehouse.pre_position and sla == "critical" and routing == "standard_route":
            routing = "split_shipment"
            reason += (
                f" | WarehouseAgent recommends pre-positioning "
                f"({warehouse.safety_stock_days}d safety stock) → split_shipment hedges delivery."
            )
            influenced_by_warehouse = True
            interactions.append(
                f"WarehouseAgent → LogisticsAgent: pre_position=True on critical order, "
                f"upgraded standard → split_shipment."
            )

        alternate_supplier = None
        if routing == "source_alternative":
            alternate_supplier = producer.recommended_supplier

        return LogisticsDecision(
            order_id=order_id,
            routing_decision=routing,
            alternate_supplier=alternate_supplier,
            reasoning=reason,
            influenced_by_producer=influenced_by_producer,
            influenced_by_warehouse=influenced_by_warehouse,
        )


# ─────────────────────────────────────────────────────────────
# Multi-agent coordinator
# ─────────────────────────────────────────────────────────────

class MultiAgentCoordinator:
    """
    Orchestrates the three-agent decision pipeline.
    Call `.coordinate(obs, order)` to get a full MultiAgentStep.
    """

    def __init__(self):
        self.producer  = ProducerAgent()
        self.warehouse = WarehouseAgent()
        self.logistics = LogisticsAgent()

    def coordinate(self, obs: dict, order: dict) -> MultiAgentStep:
        """
        Full multi-agent decision for a single order.
        Returns typed decisions from all three agents + their interaction log.
        """
        producer_dec  = self.producer.decide(obs, order)
        warehouse_dec = self.warehouse.decide(obs, order)
        logistics_dec = self.logistics.decide(obs, order, producer_dec, warehouse_dec)

        # Build cross-agent interaction narrative
        interactions = logistics_dec.influenced_by_producer or logistics_dec.influenced_by_warehouse
        agent_interactions: List[str] = []

        if logistics_dec.influenced_by_producer:
            agent_interactions.append(
                f"[P→L] ProducerAgent buffer_signal={producer_dec.buffer_signal:.2f} "
                f"influenced routing from standard → {logistics_dec.routing_decision}"
            )
        if logistics_dec.influenced_by_warehouse:
            agent_interactions.append(
                f"[W→L] WarehouseAgent pre_position={warehouse_dec.pre_position} "
                f"at {warehouse_dec.target_warehouse} shaped routing strategy"
            )
        if not agent_interactions:
            agent_interactions.append(
                "[L] LogisticsAgent decided independently (no upstream override needed)"
            )

        return MultiAgentStep(
            producer=producer_dec,
            warehouse=warehouse_dec,
            logistics=logistics_dec,
            agent_interactions=agent_interactions,
        )

    def explain(self, step: MultiAgentStep) -> str:
        """Return a structured human-readable explanation of the multi-agent decision."""
        lines = [
            "┌─ MULTI-AGENT DECISION ─────────────────────────────────",
            f"│ ProducerAgent  → supplier={step.producer.recommended_supplier} "
            f"buffer={step.producer.buffer_signal:.0%} conf={step.producer.confidence:.0%}",
            f"│   Reason: {step.producer.reason}",
            f"│ WarehouseAgent → wh={step.warehouse.target_warehouse} "
            f"pre_position={step.warehouse.pre_position} safety={step.warehouse.safety_stock_days}d",
            f"│   Reason: {step.warehouse.reason}",
            f"│ LogisticsAgent → routing={step.logistics.routing_decision} "
            f"order={step.logistics.order_id}",
            f"│   Reason: {step.logistics.reasoning}",
            "│ Agent interactions:",
        ]
        for interaction in step.agent_interactions:
            lines.append(f"│   {interaction}")
        lines.append("└────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Adaptive strategy (self-improvement)
# ─────────────────────────────────────────────────────────────

class AdaptiveStrategy:
    """
    Simple adaptive logic: tracks recent decision outcomes and adjusts
    agent thresholds when failure patterns are detected.

    After N consecutive failures (late deliveries on critical orders),
    the strategy escalates its routing aggression.
    After N consecutive successes, it relaxes to save cost.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self.recent_outcomes: List[bool] = []   # True = on-time
        self.escalation_level: int = 0           # 0 = normal, 1 = cautious, 2 = aggressive
        self.adjustments_made: int = 0

    def record(self, on_time: bool, sla_tier: str) -> None:
        weight = 3 if sla_tier == "critical" else 1
        self.recent_outcomes.extend([on_time] * weight)
        self.recent_outcomes = self.recent_outcomes[-self.window * 3:]
        self._adapt()

    def _adapt(self) -> None:
        if len(self.recent_outcomes) < self.window:
            return
        recent = self.recent_outcomes[-self.window:]
        failure_rate = recent.count(False) / len(recent)

        prev_level = self.escalation_level
        if failure_rate > 0.6:
            self.escalation_level = min(2, self.escalation_level + 1)
        elif failure_rate < 0.2:
            self.escalation_level = max(0, self.escalation_level - 1)

        if self.escalation_level != prev_level:
            self.adjustments_made += 1

    def slack_threshold_adjust(self) -> int:
        """Returns slack threshold adjustment based on escalation level."""
        return {0: 0, 1: 1, 2: 2}.get(self.escalation_level, 0)

    def cost_tolerance_adjust(self) -> float:
        """Returns cost multiplier tolerance based on escalation level."""
        return {0: 1.0, 1: 1.3, 2: 1.8}.get(self.escalation_level, 1.0)

    def describe(self) -> str:
        label = {0: "NORMAL (balanced cost/speed)",
                 1: "CAUTIOUS (prefer speed to avoid late penalties)",
                 2: "AGGRESSIVE (prioritize on-time at all costs)"}[self.escalation_level]
        recent = self.recent_outcomes[-self.window:] if self.recent_outcomes else []
        rate = f"{sum(recent)/len(recent):.0%}" if recent else "N/A"
        return (
            f"AdaptiveStrategy: level={label} | "
            f"recent_on_time={rate} | adjustments_made={self.adjustments_made}"
        )
