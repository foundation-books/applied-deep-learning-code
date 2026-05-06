# Introduction to Reinforcement Learning Companion Code

This directory contains the Gymnasium and Stable-Baselines3 runner for the
introductory reinforcement-learning chapter. The chapter uses small simulated
and control environments as reproducible laboratories, with library implementations
of DQN and PPO rather than custom reader-facing RL algorithms.

The main file is:

```text
gymnasium_rl_experiment.py
```

## Notebook Walkthrough

Open `rl_intro_walkthrough.ipynb` in Jupyter, Colab, or Kaggle when you want
to inspect the Gymnasium interaction loop one cell at a time. The notebook
reuses `gymnasium_rl_experiment.py` for the actual baseline, training,
evaluation, and artifact-writing logic, and intentionally has no saved outputs.

Use `gymnasium_rl_experiment.py` for repeatable command-line runs, homework
submissions, smoke checks, and fixed experiment directories.

Repository-safe checks:

```sh
python3 gymnasium_rl_experiment.py --check-deps --allow-missing-deps
```

From the companion-code root, `make rl-intro-code-check` also parses the script
without writing `__pycache__` bytecode.

For a full local run, install the optional RL dependencies from the shared
repository environment:

```sh
cd ..
poetry install --with rl-intro
```

The default homework environment is `LunarLander-v3`, which uses Gymnasium's
Box2D environments. The shared `rl-intro` Poetry group installs Gymnasium with
the Box2D extra so the default environment is covered by the locked setup.

If you want notebook plots, also install the shared figure dependencies:

```sh
poetry install --with rl-intro,figures
```

A quick dependency-backed Gymnasium smoke run for the homework default is:

```sh
poetry run python chapter_reinforcement_learning_intro/gymnasium_rl_experiment.py \
  --smoke-transition --env-id LunarLander-v3 \
  --output-dir /tmp/adl-rl-first-transition
```

A short LunarLander PPO homework run looks like:

```sh
poetry run python chapter_reinforcement_learning_intro/gymnasium_rl_experiment.py \
  --env-id LunarLander-v3 --algorithm ppo \
  --n-envs 4 --train-steps 100000 --eval-episodes 20 --seed 7 \
  --deterministic-eval --success-return-threshold 200 \
  --model-kwargs '{"n_steps":1024,"batch_size":64,"n_epochs":4}' \
  --output-dir runs/student-lunarlander-ppo
```

The named MiniGrid fallback avoids Box2D and gives a more puzzle-like agent
task:

```sh
poetry run python chapter_reinforcement_learning_intro/gymnasium_rl_experiment.py \
  --env-id MiniGrid-DoorKey-6x6-v0 --env-wrapper minigrid-img \
  --algorithm ppo --policy MlpPolicy \
  --n-envs 4 --train-steps 100000 --eval-episodes 20 --seed 7 \
  --model-kwargs '{"n_steps":256,"batch_size":64,"n_epochs":4}' \
  --output-dir runs/student-minigrid-doorkey-ppo
```

The runner writes `summary.json`, `config.json`, `evaluation_episodes.csv`,
`random_rollout.csv`, `trained_rollout.csv`, and Stable-Baselines3 monitor logs.
The summary records the environment, seed, training steps, random baseline,
trained-policy return, success rate, elapsed time, device, and package versions.
