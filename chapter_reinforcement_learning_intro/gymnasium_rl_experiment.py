"""Run small Gymnasium reinforcement-learning experiments for the intro-RL chapter.

The script intentionally uses standard Gymnasium environments and
Stable-Baselines3 so the chapter can report real DQN or PPO runs without
maintaining custom reader-facing RL algorithm implementations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RUNTIME_PACKAGES = {
    "gymnasium": "gymnasium",
    "stable_baselines3": "stable-baselines3",
    "torch": "torch",
    "numpy": "numpy",
}
OPTIONAL_ENV_PACKAGES = {
    "Box2D": "box2d",
    "minigrid": "minigrid",
    "ale_py": "ale-py",
}


def package_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_report() -> dict[str, Any]:
    packages = {**RUNTIME_PACKAGES, **OPTIONAL_ENV_PACKAGES}
    return {
        "required": {
            import_name: {
                "available": package_available(import_name),
                "version": package_version(distribution_name),
            }
            for import_name, distribution_name in RUNTIME_PACKAGES.items()
        },
        "optional_environment_packages": {
            import_name: {
                "available": package_available(import_name),
                "version": package_version(distribution_name),
            }
            for import_name, distribution_name in OPTIONAL_ENV_PACKAGES.items()
        },
        "all_versions": {
            import_name: package_version(distribution_name)
            for import_name, distribution_name in packages.items()
        },
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def summarize_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {str(key): summarize_observation(value) for key, value in observation.items()}
    if isinstance(observation, tuple):
        return [summarize_observation(value) for value in observation]

    summary: dict[str, Any] = {"type": type(observation).__name__}
    shape = getattr(observation, "shape", None)
    if shape is not None:
        summary["shape"] = [int(dim) for dim in shape]
    dtype = getattr(observation, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)

    flatten = getattr(observation, "reshape", None)
    if callable(flatten) and shape is not None:
        flat = observation.reshape(-1)
        count = int(getattr(flat, "shape", [len(flat)])[0])
        summary["sample"] = to_jsonable(flat[: min(8, count)])
        if count:
            try:
                summary["min"] = float(flat.min())
                summary["max"] = float(flat.max())
            except (TypeError, ValueError):
                pass
        return summary

    summary["value"] = to_jsonable(observation)
    return summary


def wrap_env(env: Any, env_wrapper: str) -> Any:
    if env_wrapper == "none":
        return env
    if env_wrapper == "minigrid-img":
        from minigrid.wrappers import ImgObsWrapper

        return ImgObsWrapper(env)
    raise ValueError(f"unknown environment wrapper {env_wrapper}")


def make_env(env_id: str, seed: int, render_mode: str | None = None, env_wrapper: str = "none") -> Any:
    import gymnasium as gym

    if env_id.startswith("MiniGrid-"):
        import minigrid  # noqa: F401
    if env_id.startswith("ALE/"):
        import ale_py  # noqa: F401

    env = gym.make(env_id, render_mode=render_mode)
    env = wrap_env(env, env_wrapper)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def run_episode(
    env: Any,
    policy: Any,
    seed: int,
    max_steps: int | None,
    deterministic: bool,
    success_return_threshold: float | None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        if max_steps is not None and step >= max_steps:
            truncated = True
            break
        if policy == "random":
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(observation, deterministic=deterministic)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        step += 1
        total_reward += float(reward)
        rows.append(
            {
                "step": step,
                "action": to_jsonable(action),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        observation = next_observation
    return {
        "return": total_reward,
        "steps": step,
        "success": int(total_reward >= success_return_threshold) if success_return_threshold is not None else 0,
        "rollout": rows,
    }


def evaluate_policy(
    env_id: str,
    policy: Any,
    episodes: int,
    seed: int,
    max_steps: int | None,
    deterministic: bool,
    success_return_threshold: float | None,
    env_wrapper: str,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    first_rollout: list[dict[str, Any]] = []
    returns: list[float] = []
    steps: list[int] = []
    successes = 0
    env = make_env(env_id, seed=seed, env_wrapper=env_wrapper)
    try:
        for episode in range(episodes):
            result = run_episode(env, policy, seed + episode, max_steps, deterministic, success_return_threshold)
            returns.append(float(result["return"]))
            steps.append(int(result["steps"]))
            successes += int(result["success"])
            episode_rows.append(
                {
                    "episode": episode,
                    "return": round(float(result["return"]), 6),
                    "steps": int(result["steps"]),
                    "success": int(result["success"]),
                }
            )
            if episode == 0:
                first_rollout = result["rollout"]
    finally:
        env.close()
    return (
        {
            "mean_return": sum(returns) / episodes,
            "min_return": min(returns),
            "max_return": max(returns),
            "success_rate": successes / episodes,
            "mean_steps": sum(steps) / episodes,
        },
        episode_rows,
        first_rollout,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def smoke_transition(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.env_id, seed=args.seed, env_wrapper=args.env_wrapper)
    try:
        observation, reset_info = env.reset(seed=args.seed)
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        payload = {
            "environment": args.env_id,
            "env_wrapper": args.env_wrapper,
            "seed": args.seed,
            "action_space": repr(env.action_space),
            "observation_space": repr(env.observation_space),
            "reset_info": to_jsonable(reset_info),
            "action": to_jsonable(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_info": to_jsonable(step_info),
            "observation": summarize_observation(observation),
            "next_observation": summarize_observation(next_observation),
        }
    finally:
        env.close()
    (output_dir / "first_transition.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def mps_available(torch: Any) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return "cuda"
    if requested == "mps":
        if not mps_available(torch):
            raise RuntimeError(
                "--device mps requested, but PyTorch MPS is not available"
            )
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if mps_available(torch):
        return "mps"
    return "cpu"


def train_sb3(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.env_util import make_vec_env

    algorithm = args.algorithm.lower()
    model_cls = {"dqn": DQN, "ppo": PPO}[algorithm]
    policy_name = args.policy
    device = resolve_device(torch, args.device)
    monitor_dir = args.output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def make_wrapped_env() -> Any:
        return make_env(args.env_id, seed=args.seed, env_wrapper=args.env_wrapper)

    env = make_vec_env(make_wrapped_env, n_envs=args.n_envs, seed=args.seed, monitor_dir=str(monitor_dir))
    start = time.perf_counter()
    model = model_cls(
        policy_name,
        env,
        seed=args.seed,
        device=device,
        learning_rate=args.learning_rate,
        verbose=0,
        **json.loads(args.model_kwargs),
    )
    model.learn(total_timesteps=args.train_steps, progress_bar=False)
    elapsed = time.perf_counter() - start
    env.close()
    return model, {"training_seconds": elapsed, "device": str(model.device)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random_summary, random_rows, random_rollout = evaluate_policy(
        args.env_id,
        "random",
        args.eval_episodes,
        args.seed + 1000,
        args.max_episode_steps,
        deterministic=True,
        success_return_threshold=args.success_return_threshold,
        env_wrapper=args.env_wrapper,
    )
    model, training_summary = train_sb3(args)
    trained_summary, trained_rows, trained_rollout = evaluate_policy(
        args.env_id,
        model,
        args.eval_episodes,
        args.seed + 2000,
        args.max_episode_steps,
        deterministic=args.deterministic_eval,
        success_return_threshold=args.success_return_threshold,
        env_wrapper=args.env_wrapper,
    )

    package_versions = {
        "python": sys.version.split()[0],
        "gymnasium": package_version("gymnasium"),
        "stable_baselines3": package_version("stable-baselines3"),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
    }
    summary: dict[str, Any] = {
        "environment": args.env_id,
        "algorithm": args.algorithm.lower(),
        "policy": args.policy,
        "env_wrapper": args.env_wrapper,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "n_envs": args.n_envs,
        "learning_rate": args.learning_rate,
        "success_return_threshold": args.success_return_threshold,
        "model_kwargs": json.loads(args.model_kwargs),
        "random_baseline": random_summary,
        "trained_policy": trained_summary,
        "training": training_summary,
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": "not-recorded",
            "packages": package_versions,
            "nvidia_smi": command_output(["nvidia-smi"]),
        },
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "config": str(output_dir / "config.json"),
            "evaluation_log": str(output_dir / "evaluation_episodes.csv"),
            "training_monitor_dir": str(output_dir / "monitor"),
            "random_rollout": str(output_dir / "random_rollout.csv"),
            "trained_rollout": str(output_dir / "trained_rollout.csv"),
        },
    }
    write_csv(
        output_dir / "evaluation_episodes.csv",
        [{"policy": "random", **row} for row in random_rows]
        + [{"policy": "trained", **row} for row in trained_rows],
        ["policy", "episode", "return", "steps", "success"],
    )
    write_csv(output_dir / "random_rollout.csv", random_rollout, ["step", "action", "reward", "terminated", "truncated"])
    write_csv(
        output_dir / "trained_rollout.csv",
        trained_rollout,
        ["step", "action", "reward", "terminated", "truncated"],
    )
    config = {
        "env_id": args.env_id,
        "algorithm": args.algorithm,
        "policy": args.policy,
        "env_wrapper": args.env_wrapper,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "max_episode_steps": args.max_episode_steps,
        "deterministic_eval": args.deterministic_eval,
        "success_return_threshold": args.success_return_threshold,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    parser.add_argument("--smoke-transition", action="store_true")
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="ppo")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--env-wrapper", choices=("none", "minigrid-img"), default="none")
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--model-kwargs", default="{}")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--success-return-threshold", type=float, default=200)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gymnasium-rl-experiment"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_deps:
        report = dependency_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        missing = [
            name
            for name, status in report["required"].items()
            if not status["available"]
        ]
        if missing and not args.allow_missing_deps:
            raise SystemExit("Missing required runtime packages: " + ", ".join(missing))
        return

    if args.smoke_transition:
        summary = smoke_transition(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
