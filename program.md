# autoresearch — Dream++ track

This is an experiment to have an autonomous AI agent run research that improves the
**Hermes Dreaming** (Dream++) self-improvement engine. We fork the base nanochat
training loop but bias experiments toward architectures, optimizers, and techniques
that transfer well to the Dream++ use case: staged proposal generation, memory
consolidation, and skill update synthesis.

## Context: Dream++

Dream++ is `hermes-dreaming` — a staged self-improvement engine for Hermes-style
memory, user, skill, and fact updates. It reads source inputs, stages proposed
changes in a reviewable artifact, and only writes to live state after explicit
approval. The engine needs efficient LLM-driven proposal generation — this
research explores better model architectures for that workload.

Relevant repo: https://github.com/Newstex/hermes-dreaming

## Setup

To set up a new experiment, work with the agent to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may29`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards
   and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
   The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once confirmed, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (NVIDIA GB10 Blackwell). The training script
runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding
startup/compilation). Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  model architecture, optimizer, hyperparameters, training loop, batch size, model
  size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**Goal: lowest val_bpb.** Since the time budget is fixed, optimize for the best
model within the time constraint. Experiments that show improvements in
architecture efficiency, memory footprint, or training speed are especially
valuable for the Dream++ use case (efficient LLM inference for proposal
generation).

**Platform note**: This is an NVIDIA GB10 (Blackwell, cap 12.1).
- FA3 from kernels-community is used (Hopper cap 9.0 check → false)
- **Known issue**: FA3 + torch.compile causes FakeTensor errors on this arch.
  The agent must fix this: try disabling torch.compile for the attn module,
  using a fallback attention, or patching the kernel import.
- This is the first research challenge — the baseline won't run until the
  attention mechanism is fixed for Blackwell.

**VRAM** is a soft constraint. ~100GB available headroom on this platform.
Some increase is acceptable for meaningful val_bpb gains.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Conversely, removing
something and getting equal or better results is a great outcome.

**The first run**: Establish the baseline — run the training script as-is.

## Output format

Once the script finishes it prints a summary like:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract metric: `grep "^val_bpb:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Header + 5 columns:

```
commit\tval_bpb\tmemory_gb\tstatus\tdescription
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (0.000000 for crashes)
3. peak memory in GB, round to .1f (0.0 for crashes)
4. status: `keep`, `discard`, or `crash`
5. short text description

## The experiment loop

Runs on a dedicated branch (e.g. `autoresearch/may29`). LOOP FOREVER:

1. Look at git state: current branch/commit
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If grep output empty → crash. Read `tail -n 50 run.log`, attempt fix.
7. Record in results.tsv (leave tsv untracked by git)
8. If val_bpb improved (lower), advance the branch
9. If equal or worse, git reset back to start

**Timeout**: Each experiment ~5 minutes. If >10 minutes, kill + discard.

**Crashes**: Fix dumb bugs and re-run. Log fundamental failures as "crash" and
move on.

**NEVER STOP**: Do not pause to ask the human. Continue indefinitely until
manually stopped. If out of ideas, re-read papers, combine near-misses, try
radical changes.
