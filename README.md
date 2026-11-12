# AlphaZero-style Connect Four

A from-scratch AlphaZero-style agent for Connect Four: a policy/value CNN
guides Monte Carlo Tree Search, trained entirely through self-play (no
human game data).

**Result** (see `progress/`): after 340 training iterations the trained
network beats a fixed pure-MCTS baseline (search only, no learning) 19-1,
and against a deliberately strong baseline its win rate climbs from **22%
at iteration 50 to a stable 85-100% after iteration 200** — direct
evidence the learned policy/value guidance is doing real work, not just
riding on search depth.

## Folder layout

```
game.py, network.py, mcts.py, batched_mcts.py   core game/model/search
self_play.py, batched_self_play.py              self-play generation
arena.py, batched_arena.py, agents.py           head-to-head matches
baselines.py, replay_buffer.py                  non-learned baselines, training buffer
train.py, evaluate.py, play.py                  entry points (see below)
plot_progress.py, tournament.py                 post-hoc analysis (see below)
tests/           engine tests, incl. a differential test of win detection
checkpoints/     saved networks (best.pt, current.pt, best_iter{N}.pt snapshots)
logs/            raw per-iteration training logs (one file per training run)
progress/        parsed history + plots + tournament results (generated, not hand-edited)
```

## Tests

```bash
python tests/test_game.py       # or: pytest tests/
```

`check_winner` was rewritten after training from a full-board scan to an
O(1) check of the four lines through the last-placed piece — win detection
runs after every move of every rollout, so it is hot. The original
implementation is preserved in `tests/test_game.py` as a reference oracle
and the two are asserted equal across **~32,000 positions from 1,500
random playouts**. The test suite also pins the optimisation's
precondition: it inspects only lines through `last_move`, so a
hand-constructed board without one returns 0 even when a win is present.
Every in-play state comes from `apply_move`, which sets it.

## How the code fits together

- `game.py` — board state, legal moves, win detection, and the "my pieces
  / their pieces" canonical encoding fed to the network.
- `network.py` — a small residual CNN with two heads: policy (move
  distribution) and value (expected outcome for the player to move).
  `predict_batch` evaluates many positions in one GPU call.
- `mcts.py` — single-position PUCT tree search. Any object with
  `.evaluate(game) -> (probs, value)` can guide it — the network via
  `NetworkEvaluator`, or `baselines.py`'s random-rollout evaluator
  (classic "pure MCTS" with no learning).
- `batched_mcts.py` — the same search, but running many independent trees
  in lockstep so every simulation step's leaf evaluations batch into one
  GPU call instead of one board at a time. This is the throughput fix that
  took training from ~150s/iteration to ~20s/iteration (see "Performance
  history" below) — batch-1 GPU inference is latency-bound, not
  compute-bound, for a network this small.
- `self_play.py` / `batched_self_play.py` — play games with MCTS providing
  an improved policy at each move, record `(state, mcts_policy, outcome)`
  as training data. The batched version runs many games concurrently.
- `replay_buffer.py` — stores self-play examples (with free horizontal-flip
  augmentation, since Connect Four is left-right symmetric).
- `train.py` — the main loop: self-play with the current best network →
  train a candidate → arena match candidate vs. best → promote only on a
  decisive win. Self-play always uses the frozen best network, so one bad
  training step can't poison the data.
- `arena.py` / `batched_arena.py` / `agents.py` — head-to-head matches
  between any two agents (the batched version groups games by which
  network is currently on move so each group still batches).
- `evaluate.py` — trained network vs. random mover and vs. pure MCTS
  (rollout, no learning) — the real test of whether training helped.
- `play.py` — play against a checkpoint from the terminal.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python -u train.py --iterations 300 --games-per-iter 64 --simulations 80 \
    --arena-simulations 50 --arena-games 20 --train-steps-per-iter 200 \
    --resume > logs/training_run3.txt 2>&1
```

(`-u` for unbuffered output so the redirected log streams live instead of
appearing only in large chunks or at exit.)

Each iteration: generates self-play games (batched), trains on the replay
buffer, then runs a batched arena match to decide whether to promote.
Checkpoints land in `checkpoints/` (`best.pt` = current best network,
`current.pt` = latest trained candidate, `best_iter{N}.pt` snapshots every
10 iterations — note the snapshot filenames use each *run's own* local
iteration counter, so a later `--resume` run will overwrite an earlier
run's same-numbered snapshots; see `tournament.py`'s docstring for how the
analysis scripts account for this).

Resume a training run with `--resume` (loads `best.pt`).

## Evaluating

```bash
python evaluate.py --checkpoint best.pt --games 30
```

Reports win rate against a random mover and against pure MCTS (random
rollouts, same simulation budget but no learned guidance). Beating the
rollout baseline is the meaningful signal — beating random alone is a low
bar even for an untrained network with a few hundred search simulations.

## Playing against it

```bash
python play.py --checkpoint best.pt --simulations 200
```

## Analyzing training progress

```bash
python plot_progress.py                      # logs/*.txt -> progress/training_history.csv + training_progress.png
python tournament.py --rollout-sims 600      # every best_iter*.pt vs a fixed baseline -> progress/tournament_*_r600.*
```

`plot_progress.py` charts training loss, arena promotion history, and
self-play outcome rates over the whole (possibly multi-run) training
history. `tournament.py` gives the more direct signal — actual playing
strength against a fixed non-learned opponent — since the loss curves and
noisy 20-game arena results are proxies for that, not the thing itself.

### Choose a baseline that can't saturate

`progress/` contains two tournament runs, and the difference between them
is the lesson:

* **`*_r100.*`** — baseline at 100 rollout simulations. **Not useful.** The
  network already beats it 70-100% from the very first snapshot tested
  (iteration 50) and never separates further. A flat line at ~95% looks
  like "training did nothing" when it actually means the ruler is too
  short to measure with.
* **`*_r600.*`** — baseline at 600 rollout simulations, ~10x the network's
  own 60-simulation budget. This has enough headroom to show the real
  curve: **22% at iteration 50 -> parity around iteration 80-100 -> a
  stable 85-100% band from iteration 200 onward.**

A ceiling effect is easy to mistake for a null result. If a metric is
pinned near its maximum, make the task harder before concluding anything.

### The two-phase learning curve

Three independent signals agree on a phase change around iteration ~230,
which is stronger evidence than any single curve:

1. **Iterations 0-100** — player-1 win rate is high but volatile (0.35-0.85).
   Weak, noisy play where whoever finds a tactic first usually wins.
2. **Iterations 100-230** — P1 win rate *falls* to ~0.3-0.4 while the draw
   rate climbs to ~0.2-0.3 and games lengthen from ~18 to ~32 moves. The
   network is learning **defence**: fewer cheap blunders, so the
   first-player edge is temporarily masked.
3. **Iterations 230-340** — P1 win rate climbs to ~0.7-0.8, draws collapse,
   games shorten slightly. The network learns to **convert** won positions
   rather than merely avoid losing — reasserting Connect Four's
   game-theoretic first-player advantage.

The r600 tournament corroborates this from the outside: strength dips at
iterations 170-190 (the defensive phase) before the strong finish.

## Design notes / what to watch for

- **Value convention**: every state's value is always from the
  perspective of the player about to move there. This is why values flip
  sign when backed up from a child to its parent in MCTS — see the
  docstring in `mcts.py`.
- **Self-play uses the frozen best network**, not the network currently
  being trained — a single bad gradient step can't corrupt the data other
  games are being generated from.
- **Arena promotion requires a decisive win rate** (default 55%, draws
  excluded from the decisive-game count) — guards against promoting a
  network that only "improved" by noise. With a small number of arena
  games this is still a coarse, noisy gate (expect it to swing between 0%,
  50%, and 100% iteration to iteration) — `tournament.py` gives a cleaner
  strength signal across many more games per checkpoint.
- **Connect Four is a solved first-player win**, not a game that draws
  with perfect play (unlike Tic-Tac-Toe). So as the agent improves, expect
  player-1 win rate in self-play to trend up, not toward parity — see
  `progress/training_progress.png` panel 3, where it visibly dips (as the
  network learns defense, temporarily narrowing the first-player edge)
  before recovering as endgame conversion technique catches up.
- **Expect early instability**: with a randomly initialized network,
  early self-play games are close to random play, and MCTS's PUCT
  exploration term dominates until the value head has anything meaningful
  to say.

## Performance history

The first working version ran self-play and arena games one board at a
time, at ~150-160s/iteration with GPU utilization around 39% (batch-1 GPU
inference is latency-bound — kernel launch overhead dominates actual
compute for a network this small). `batched_mcts.py` fixed this by running
many independent search trees in lockstep and batching every simulation
step's leaf evaluations into a single GPU call; measured throughput went
from ~267 network evaluations/sec to ~4,320/sec (~16x), bringing iteration
time down to ~20s even with deeper search and more self-play games per
iteration.

## Stretch goals

- Vectorize or multiprocess the MCTS tree-selection walk itself — it's now
  the bottleneck (GPU utilization tops out around 25-40% even with
  batching, since selection is sequential Python per game between batched
  GPU calls).
- Track Elo across checkpoints instead of pairwise win rates.
- Scale up to a larger game (Gomoku, small-board Go) once the pipeline is
  validated here.
