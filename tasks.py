"""
Task definitions — 3 tasks, easy → medium → hard.
Each includes clear objectives, grading criteria, and scenario mapping.
"""

from models import TaskSpec

TASKS = [
    TaskSpec(
        task_id="task_easy",
        name="Single-Lane Routing Under Stable Conditions",
        difficulty="easy",
        description=(
            "A regional distributor must fulfill a week's worth of orders from three demand nodes "
            "(Chicago, Munich, Tokyo) using a stable, disruption-free supply network. "
            "All lanes are active, inventory is sufficient, and the budget is comfortable. "
            "The challenge is to choose the lowest-cost routing that still meets each order's deadline."
        ),
        objective=(
            "Maximize on-time delivery rate while spending efficiently. "
            "Prefer standard routes unless a deadline is tight enough to justify express. "
            "Do not use spot market — it is never cost-justified in this scenario."
        ),
        max_steps=20,
        budget_usd=500_000,
        passing_score=0.70,
        grading_criteria={
            "on_time_delivery_rate": 0.40,
            "budget_utilization":    0.20,
            "mean_reward":           0.30,
            "sla_compliance":        0.10,
        },
    ),
    TaskSpec(
        task_id="task_medium",
        name="Disruption Response with Budget Trade-offs",
        difficulty="medium",
        description=(
            "A global logistics manager must route 14 days of orders across a network experiencing "
            "occasional disruptions — port congestion, weather events, and a supplier delay. "
            "The budget is moderate but disruptions will force expensive reroutes on some days. "
            "The agent must decide WHEN to absorb the premium cost of express/spot routes "
            "versus deferring or partially fulfilling lower-priority orders."
        ),
        objective=(
            "Balance cost efficiency against SLA compliance. Critical-tier orders must arrive on time "
            "even if it requires express routing. Standard and flexible orders should use the cheapest "
            "viable option. Deferral is a legitimate tool — but only when the deadline window permits it."
        ),
        max_steps=60,
        budget_usd=800_000,
        passing_score=0.60,
        grading_criteria={
            "on_time_delivery_rate": 0.30,
            "budget_utilization":    0.20,
            "mean_reward":           0.20,
            "sla_compliance":        0.20,
            "disruption_adaptation": 0.10,
        },
    ),
    TaskSpec(
        task_id="task_hard",
        name="Cascading Disruptions Under Tight Budget Constraint",
        difficulty="hard",
        description=(
            "A supply chain crisis simulation: 21 days, high disruption frequency (45%), "
            "a tight budget ($600K — roughly 70% of what the medium task allows), "
            "double the order volume, and geopolitical uncertainty raising spot market premiums "
            "to 2x+ baseline. Multiple disruptions will overlap simultaneously. "
            "Some critical orders will be physically impossible to fulfill on time due to cascading failures — "
            "the agent must triage intelligently, accepting some SLA breaches to prevent budget collapse "
            "that would leave later critical orders completely unfulfilled."
        ),
        objective=(
            "Maximize total weighted value delivered within budget. This requires hard trade-offs: "
            "prioritizing high-value critical orders, accepting late delivery on low-value flexible orders, "
            "using split shipments to spread risk, and knowing when deferral is wiser than spot-market panic. "
            "A perfect score is not achievable — excellence means making the best possible decisions "
            "under genuinely conflicting constraints."
        ),
        max_steps=130,
        budget_usd=600_000,
        passing_score=0.50,
        grading_criteria={
            "on_time_delivery_rate": 0.25,
            "budget_utilization":    0.15,
            "mean_reward":           0.15,
            "sla_compliance":        0.25,
            "disruption_adaptation": 0.15,
            "cost_per_unit":         0.05,
        },
    ),
]

TASK_MAP = {t.task_id: t for t in TASKS}
