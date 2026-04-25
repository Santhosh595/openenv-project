# 🎥 Video Script — Supply Chain Multi-Agent OpenEnv
## Target: 90–120 seconds

---

**[0:00 – 0:12] Hook (problem)**

> "Every day, global supply chains face crisis: a typhoon hits Shanghai, a strike shuts Rotterdam, a port closes in LA — all at once. Humans can't track the cascading effects fast enough. This environment teaches AI to."

*[Screen: world map with disruption icons popping up in real time]*

---

**[0:12 – 0:30] The environment**

> "We built a fully realistic simulation: 3 suppliers, 3 warehouses, 3 demand nodes, 12 freight lanes. Orders arrive daily across 3 SLA tiers — critical, standard, flexible. Disruptions fire stochastically, blocking lanes and raising spot-market costs. The agent has 8 routing decisions and a hard budget limit."

*[Screen: network diagram animating order flow, then a disruption blocking lanes]*

---

**[0:30 – 0:55] Multi-agent architecture**

> "What makes this v2 unique is the three-agent architecture. A ProducerAgent monitors supplier health and signals when to switch sources. A WarehouseAgent tracks inventory and tells us when to pre-position stock. A LogisticsAgent synthesizes both signals into a final routing decision. They genuinely influence each other — watch."

*[Screen: demo.py output — the three-agent decision panel appearing for one step]*

```
ProducerAgent  → SUP_MX_MTY  buffer=90%  conf=40%
  └ All preferred suppliers disrupted. Using Monterrey fallback.
WarehouseAgent → WH_US_LAX  pre_position=True  safety=7d
  └ 2 active disruptions. Recommend safety stock at LA.
LogisticsAgent → ROUTING: SPLIT_SHIPMENT  [influenced by warehouse]
  └ WarehouseAgent pre_position=True on critical order → upgraded routing
```

---

**[0:55 – 1:15] Training evidence**

> "We trained a REINFORCE policy against this environment. In under 5 minutes on a free Colab runtime, the agent goes from random — getting 32% on-time delivery — to 82% on-time, outperforming the hand-crafted heuristic. Here's the learning curve."

*[Screen: reward_vs_steps_task_medium.png — show the training curve rising from random baseline]*

---

**[1:15 – 1:35] Live API demo**

> "The whole thing runs as a live API on Hugging Face Spaces. Hit /multi-agent/step and you get structured decisions from all three agents plus a decomposed reward breakdown. Hit /grader to get a reproducible score."

*[Screen: browser showing /docs or a curl response with the multi-agent JSON output]*

---

**[1:35 – 1:50] Close**

> "OpenEnv-compliant, deterministic grading, reproducible training in 5 minutes, three cooperating agents, and a real-world problem that matters. That's the Supply Chain Disruption Management environment."

*[Screen: GitHub repo + HF Space link]*

---

## 📋 Recording Tips

- Screen resolution: 1920×1080
- Terminal font: 16pt, dark background
- Run: `python demo.py --task task_medium --steps 6` for live recording
- Pause at each agent decision panel — that's the key visual
- Show the training plot for ≥5 seconds — judges need to see learning
- Keep cursor movement slow and deliberate

## 🛠️ Commands to Run During Recording

```bash
# Terminal 1 — show the demo
python demo.py --task task_medium --steps 6

# Terminal 2 — show the API response
curl -s http://localhost:7860/multi-agent/demo?task_id=task_medium&steps=3 | python -m json.tool | head -80

# Browser — show /docs
open http://localhost:7860/docs
```
