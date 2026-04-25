#!/usr/bin/env python3
"""
training.py — Supply Chain RL Training Pipeline
================================================
Colab-compatible training script. Implements REINFORCE (policy gradient)
to learn an improved routing policy versus the heuristic baseline.

DESIGN CHOICES (hackathon-appropriate):
  - Pure Python / NumPy — no GPU required, runs in Colab free tier
  - Trains in < 5 minutes for 3 tasks
  - Produces reward_vs_steps.png comparison plot
  - Saves policy weights to policy_weights.json

USAGE:
  python training.py                    # full training run
  python training.py --quick            # 100 episodes for fast demo
  python training.py --plot-only        # regenerate plot from saved data

COLAB:
  !pip install numpy matplotlib
  !python training.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Ensure local modules importable ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────
# Policy: a lightweight linear policy over hand-crafted features
# ─────────────────────────────────────────────────────────────

ACTIONS = [
    "standard_route",
    "express_route",
    "spot_market",
    "split_shipment",
    "defer_24h",
    "defer_48h",
    "source_alternative",
    "partial_fulfill",
]
N_ACTIONS = len(ACTIONS)
N_FEATURES = 10   # see extract_features()


def extract_features(obs: dict, order: dict) -> np.ndarray:
    """
    Hand-crafted feature vector — captures the key decision signals.

    Feature index:
      0: normalized slack (deadline_day - episode_day) / max_days
      1: sla_tier one-hot critical
      2: sla_tier one-hot standard
      3: sla_tier one-hot flexible
      4: budget_fraction (budget_remaining / initial_budget)
      5: disruption_count (capped at 3, normalized)
      6: spot_market_premium (normalized, capped at 3x)
      7: on_time_delivery_rate
      8: service_level (critical SLA fraction)
      9: units_fraction (normalized by max expected)
    """
    day       = obs.get("episode_day", 0)
    max_days  = obs.get("max_days", 14)
    budget    = obs.get("budget_remaining", 500_000)
    disrupts  = len(obs.get("active_disruptions", []))
    spot      = obs.get("spot_market_premium", 1.0)
    on_time   = obs.get("on_time_delivery_rate", 1.0)
    svc_lvl   = obs.get("service_level", 1.0)

    deadline  = order.get("deadline_day", day + 5)
    sla       = order.get("sla_tier", "standard")
    units     = order.get("units_required", 500)

    slack     = max(0, deadline - day)

    # Infer initial budget from task (approximate)
    initial_budget = max(budget, 1)  # can't know without task context; normalize relatively

    feat = np.array([
        min(slack / max(max_days, 1), 1.0),            # 0
        1.0 if sla == "critical"  else 0.0,            # 1
        1.0 if sla == "standard"  else 0.0,            # 2
        1.0 if sla == "flexible"  else 0.0,            # 3
        min(budget / 800_000, 1.0),                    # 4
        min(disrupts / 3.0, 1.0),                      # 5
        min((spot - 1.0) / 2.0, 1.0),                  # 6
        on_time,                                       # 7
        svc_lvl,                                       # 8
        min(units / 1000.0, 1.0),                      # 9
    ], dtype=np.float32)

    return feat


class LinearPolicy:
    """
    Softmax linear policy: logits = W @ features + b
    W: (N_ACTIONS, N_FEATURES), b: (N_ACTIONS,)
    """

    def __init__(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(N_ACTIONS, N_FEATURES).astype(np.float32) * 0.1
        self.b = np.zeros(N_ACTIONS, dtype=np.float32)

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self.W @ features + self.b

    def action_probs(self, features: np.ndarray) -> np.ndarray:
        lg = self.logits(features)
        lg -= lg.max()  # numerical stability
        exp = np.exp(lg)
        return exp / exp.sum()

    def sample(self, features: np.ndarray) -> Tuple[int, float]:
        """Return (action_index, log_prob)."""
        probs = self.action_probs(features)
        idx   = np.random.choice(N_ACTIONS, p=probs)
        return idx, float(np.log(probs[idx] + 1e-8))

    def greedy(self, features: np.ndarray) -> int:
        return int(np.argmax(self.action_probs(features)))

    def save(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self.W = data["W"]
        self.b = data["b"]

    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearPolicy":
        p = cls()
        p.W = np.array(d["W"], dtype=np.float32)
        p.b = np.array(d["b"], dtype=np.float32)
        return p


# ─────────────────────────────────────────────────────────────
# REINFORCE trainer
# ─────────────────────────────────────────────────────────────

class REINFORCETrainer:
    """
    Vanilla REINFORCE with baseline subtraction.
    One episode = one full task run.
    """

    def __init__(
        self,
        task_id: str,
        lr: float = 0.02,
        gamma: float = 0.95,
        baseline_decay: float = 0.9,
        max_steps_per_episode: int = 60,
    ):
        self.task_id  = task_id
        self.lr       = lr
        self.gamma    = gamma
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.max_steps      = max_steps_per_episode
        self.policy   = LinearPolicy(seed=42)

    def _run_episode(self, explore: bool = True) -> Tuple[float, List[Tuple]]:
        """
        Run one episode. Returns (total_reward, trajectory).
        trajectory = list of (features, action_idx, reward, log_prob)
        """
        from environment import SupplyChainEnv
        from models import Action

        env = SupplyChainEnv(task_id=self.task_id)
        obs = env.reset()

        trajectory = []
        total_reward = 0.0
        steps = 0

        while not env._is_done() and steps < self.max_steps:
            obs_dict = obs.model_dump()
            orders   = obs_dict.get("pending_orders", [])

            if not orders:
                env._advance_day()
                obs = env._build_observation()
                continue

            # Pick most urgent order
            sla_pri = {"critical": 0, "standard": 1, "flexible": 2}
            day     = obs_dict["episode_day"]
            order   = min(orders, key=lambda o: (sla_pri[o["sla_tier"]], o["deadline_day"] - day))

            feat      = extract_features(obs_dict, order)
            if explore:
                act_idx, log_prob = self.policy.sample(feat)
            else:
                act_idx   = self.policy.greedy(feat)
                log_prob  = float(np.log(self.policy.action_probs(feat)[act_idx] + 1e-8))

            routing = ACTIONS[act_idx]

            try:
                action = Action(
                    order_id=order["order_id"],
                    routing_decision=routing,  # type: ignore
                    alternate_supplier=None,
                    reasoning=f"Policy action: {routing}",
                )
                result    = env.step(action)
                r         = float(result.reward.total)
                obs       = result.observation
            except Exception:
                r   = -0.5
                obs = env._build_observation()

            trajectory.append((feat, act_idx, r, log_prob))
            total_reward += r
            steps += 1

        return total_reward, trajectory

    def _policy_gradient_update(self, trajectory: List[Tuple]) -> None:
        """REINFORCE update with reward-to-go and baseline subtraction."""
        if not trajectory:
            return

        # Compute discounted rewards-to-go
        T = len(trajectory)
        returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in reversed(range(T)):
            running = trajectory[t][2] + self.gamma * running
            returns[t] = running

        # Update running baseline
        ep_mean = float(returns.mean())
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * ep_mean

        # Policy gradient update: W += lr * (G_t - b) * grad_log_pi
        for t, (feat, act_idx, _, log_prob) in enumerate(trajectory):
            advantage = returns[t] - self.baseline
            if abs(advantage) < 1e-6:
                continue

            # grad log π(a|s) for softmax policy
            probs = self.policy.action_probs(feat)
            # ∇_W log π(a|s) = feat * (1[a] - π(a|s)) for row act_idx
            one_hot = np.zeros(N_ACTIONS, dtype=np.float32)
            one_hot[act_idx] = 1.0
            delta = advantage * (one_hot - probs)   # (N_ACTIONS,)

            # Weight gradient: outer product scaled by advantage
            self.policy.W += self.lr * np.outer(delta, feat)
            self.policy.b += self.lr * delta * 0.1  # smaller bias update

    def train(
        self,
        n_episodes: int = 300,
        eval_every: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training run. Returns history dict with:
          train_rewards, eval_rewards, episodes
        """
        history = {
            "train_rewards": [],
            "eval_rewards":  [],
            "episodes":      [],
        }

        best_eval = -float("inf")

        for ep in range(1, n_episodes + 1):
            total_r, trajectory = self._run_episode(explore=True)
            self._policy_gradient_update(trajectory)
            history["train_rewards"].append(total_r)

            if ep % eval_every == 0:
                eval_rewards = []
                for _ in range(3):   # 3 eval episodes
                    r, _ = self._run_episode(explore=False)
                    eval_rewards.append(r)
                eval_mean = float(np.mean(eval_rewards))
                history["eval_rewards"].append(eval_mean)
                history["episodes"].append(ep)

                if eval_mean > best_eval:
                    best_eval = eval_mean
                    self.policy.save(f"policy_{self.task_id}_best")

                if verbose:
                    smoothed = float(np.mean(history["train_rewards"][-eval_every:]))
                    print(
                        f"  Ep {ep:>4}/{n_episodes} | "
                        f"train_avg={smoothed:+.3f} | "
                        f"eval_mean={eval_mean:+.3f} | "
                        f"baseline={self.baseline:+.3f}"
                    )

        return history


# ─────────────────────────────────────────────────────────────
# Random baseline (for comparison)
# ─────────────────────────────────────────────────────────────

def run_random_baseline(task_id: str, n_episodes: int = 30) -> List[float]:
    """Run a random policy as lower bound for comparison."""
    from environment import SupplyChainEnv
    from models import Action

    rewards = []
    for _ in range(n_episodes):
        env = SupplyChainEnv(task_id=task_id)
        obs = env.reset()
        total = 0.0
        steps = 0
        while not env._is_done() and steps < 60:
            obs_dict = obs.model_dump()
            orders   = obs_dict.get("pending_orders", [])
            if not orders:
                env._advance_day()
                obs = env._build_observation()
                continue
            order   = orders[0]
            routing = random.choice(ACTIONS)
            try:
                action = Action(
                    order_id=order["order_id"],
                    routing_decision=routing,  # type: ignore
                    alternate_supplier=None,
                    reasoning="Random policy",
                )
                result  = env.step(action)
                total  += result.reward.total
                obs     = result.observation
            except Exception:
                total -= 0.5
                obs = env._build_observation()
            steps += 1
        rewards.append(total)
    return rewards


def run_heuristic_baseline(task_id: str, n_episodes: int = 10) -> List[float]:
    """Run the existing heuristic agent as the 'trained' comparison baseline."""
    from environment import SupplyChainEnv
    from models import Action

    def heuristic(obs_dict: dict) -> dict:
        orders = obs_dict.get("pending_orders", [])
        day    = obs_dict["episode_day"]
        budget = obs_dict["budget_remaining"]
        disruptions = obs_dict.get("active_disruptions", [])
        sla_priority = {"critical": 0, "standard": 1, "flexible": 2}
        order = min(orders, key=lambda o: (sla_priority[o["sla_tier"]], o["deadline_day"] - day))
        slack = order["deadline_day"] - day
        sla   = order["sla_tier"]
        if budget < 5000:                                         decision = "partial_fulfill"
        elif disruptions and sla == "critical" and slack <= 2:   decision = "spot_market"
        elif sla == "critical" and slack <= 2:                   decision = "express_route"
        elif disruptions and sla == "standard" and slack <= 3:   decision = "split_shipment"
        elif slack >= 5:                                          decision = "standard_route"
        elif sla == "flexible" and slack <= 2:                   decision = "defer_24h"
        else:                                                     decision = "standard_route"
        return {"order_id": order["order_id"], "routing_decision": decision}

    rewards = []
    for _ in range(n_episodes):
        env = SupplyChainEnv(task_id=task_id)
        obs = env.reset()
        total = 0.0
        steps = 0
        while not env._is_done() and steps < 60:
            obs_dict = obs.model_dump()
            orders = obs_dict.get("pending_orders", [])
            if not orders:
                env._advance_day(); obs = env._build_observation(); continue
            action_data = heuristic(obs_dict)
            try:
                action = Action(
                    order_id=action_data["order_id"],
                    routing_decision=action_data["routing_decision"],  # type: ignore
                    alternate_supplier=None,
                    reasoning=None,
                )
                result = env.step(action)
                total += result.reward.total
                obs    = result.observation
            except Exception:
                total -= 0.1
                obs = env._build_observation()
            steps += 1
        rewards.append(total)
    return rewards


# ─────────────────────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────────────────────

def generate_plots(
    task_id: str,
    train_history: Dict[str, List[float]],
    random_rewards: List[float],
    heuristic_rewards: List[float],
    outdir: str = ".",
) -> str:
    """
    Generate two plots:
      1. reward_vs_steps_{task_id}.png  — training curve with smoothed MA
      2. comparison_{task_id}.png       — random vs heuristic vs trained box plot
    Returns path to main plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Supply Chain RL — Training Results ({task_id})",
        fontsize=14, fontweight="bold"
    )

    # ── Plot 1: Training curve ───────────────────────────────
    ax1 = axes[0]
    train_r = train_history["train_rewards"]
    episodes = list(range(1, len(train_r) + 1))

    # Smoothed moving average
    window = max(1, len(train_r) // 20)
    smoothed = np.convolve(train_r, np.ones(window) / window, mode="valid")
    sm_ep    = episodes[window - 1:]

    ax1.plot(episodes, train_r, alpha=0.25, color="#4C9BE8", linewidth=0.8, label="Episode reward")
    ax1.plot(sm_ep, smoothed, color="#1a5fa8", linewidth=2.0, label=f"Smoothed (MA{window})")

    # Eval checkpoints
    if train_history.get("eval_rewards"):
        ax1.scatter(
            train_history["episodes"],
            train_history["eval_rewards"],
            color="#E8793E", s=50, zorder=5, label="Eval (greedy, n=3)"
        )

    # Random baseline reference line
    rand_mean = float(np.mean(random_rewards)) if random_rewards else -3.0
    ax1.axhline(rand_mean, color="#c0392b", linestyle="--", alpha=0.7, label=f"Random baseline ({rand_mean:.2f})")

    ax1.set_xlabel("Training Episode", fontsize=11)
    ax1.set_ylabel("Cumulative Episode Reward", fontsize=11)
    ax1.set_title("Reward vs Training Steps", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Comparison box plot ──────────────────────────
    ax2 = axes[1]

    # Trained policy: use last 20% of training episodes
    cutoff = max(1, len(train_r) * 4 // 5)
    trained_rewards = train_r[cutoff:]

    data   = [random_rewards, heuristic_rewards, trained_rewards]
    labels = ["Random\nPolicy", "Heuristic\nBaseline", "Trained\nPolicy"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    bp = ax2.boxplot(data, patch_artist=True, notch=False, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Cumulative Episode Reward", fontsize=11)
    ax2.set_title("Random vs Heuristic vs Trained", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate means
    for i, (d, label) in enumerate(zip(data, labels), start=1):
        if d:
            m = float(np.mean(d))
            ax2.annotate(f"μ={m:.2f}", xy=(i, m), xytext=(i + 0.35, m),
                         fontsize=9, color="navy", va="center")

    plt.tight_layout()
    out_path = os.path.join(outdir, f"reward_vs_steps_{task_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supply Chain RL Training Pipeline")
    parser.add_argument("--task",       default="task_medium",
                        choices=["task_easy", "task_medium", "task_hard", "all"])
    parser.add_argument("--episodes",   type=int, default=300)
    parser.add_argument("--quick",      action="store_true",
                        help="Quick demo: 100 episodes")
    parser.add_argument("--plot-only",  action="store_true",
                        help="Regenerate plots from saved training_results.json")
    parser.add_argument("--outdir",     default=".", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.quick:
        args.episodes = 100

    tasks = ["task_easy", "task_medium", "task_hard"] if args.task == "all" else [args.task]

    if args.plot_only:
        with open("training_results.json") as f:
            saved = json.load(f)
        for task_id, data in saved.items():
            generate_plots(
                task_id,
                data["history"],
                data["random_rewards"],
                data["heuristic_rewards"],
                outdir=args.outdir,
            )
        return

    all_results = {}

    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {task_id} | {args.episodes} episodes")
        print(f"{'='*60}")

        # 1. Random baseline (fast)
        print(f"  Running random baseline (30 episodes)...")
        random_rewards = run_random_baseline(task_id, n_episodes=30)
        print(f"    Random: μ={np.mean(random_rewards):+.3f}  σ={np.std(random_rewards):.3f}")

        # 2. Heuristic baseline
        print(f"  Running heuristic baseline (10 episodes)...")
        heuristic_rewards = run_heuristic_baseline(task_id, n_episodes=10)
        print(f"    Heuristic: μ={np.mean(heuristic_rewards):+.3f}  σ={np.std(heuristic_rewards):.3f}")

        # 3. REINFORCE training
        print(f"  Training REINFORCE policy...")
        t0 = time.time()
        trainer = REINFORCETrainer(task_id=task_id, lr=0.02, gamma=0.95)
        history = trainer.train(
            n_episodes=args.episodes,
            eval_every=max(10, args.episodes // 15),
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"  Training complete in {elapsed:.1f}s")

        # 4. Final eval
        trained_rewards = [trainer._run_episode(explore=False)[0] for _ in range(5)]
        print(f"    Trained (greedy): μ={np.mean(trained_rewards):+.3f}  σ={np.std(trained_rewards):.3f}")

        # 5. Save policy
        policy_path = os.path.join(args.outdir, f"policy_{task_id}.json")
        with open(policy_path, "w") as f:
            json.dump(trainer.policy.to_dict(), f)
        print(f"  Policy saved → {policy_path}")

        # 6. Generate plots
        generate_plots(task_id, history, random_rewards, heuristic_rewards, outdir=args.outdir)

        # 7. Compute improvement
        rand_mean     = float(np.mean(random_rewards))
        heuristic_mean = float(np.mean(heuristic_rewards))
        trained_mean  = float(np.mean(trained_rewards))
        improvement   = ((trained_mean - rand_mean) / abs(rand_mean)) * 100 if rand_mean != 0 else 0

        print(f"\n  ┌─ RESULTS SUMMARY ─────────────────────────────────")
        print(f"  │  Random baseline:    {rand_mean:+.3f}")
        print(f"  │  Heuristic baseline: {heuristic_mean:+.3f}")
        print(f"  │  Trained policy:     {trained_mean:+.3f}")
        print(f"  │  Improvement over random: {improvement:+.1f}%")
        print(f"  └───────────────────────────────────────────────────")

        all_results[task_id] = {
            "history":          history,
            "random_rewards":   random_rewards,
            "heuristic_rewards": heuristic_rewards,
            "trained_rewards":  trained_rewards,
            "summary": {
                "random_mean":     rand_mean,
                "heuristic_mean":  heuristic_mean,
                "trained_mean":    trained_mean,
                "improvement_pct": improvement,
                "episodes":        args.episodes,
                "time_seconds":    elapsed,
            }
        }

    # Save all results
    results_path = os.path.join(args.outdir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] training_results.json → {results_path}")

    # Final summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<16} {'Random':>8} {'Heuristic':>10} {'Trained':>8} {'Δ%':>8}")
    print(f"  {'-'*55}")
    for task_id, data in all_results.items():
        s = data["summary"]
        print(
            f"  {task_id:<16} {s['random_mean']:>+8.3f} "
            f"{s['heuristic_mean']:>+10.3f} "
            f"{s['trained_mean']:>+8.3f} "
            f"{s['improvement_pct']:>+7.1f}%"
        )


if __name__ == "__main__":
    main()
