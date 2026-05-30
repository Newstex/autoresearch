#!/usr/bin/env python3
"""
Autonomous research loop controller for Dream++ autoresearch.

This script acts as the AI researcher:
1. Reads program.md for context
2. For each experiment iteration: proposes a change to train.py, runs it,
   evaluates results, keeps/discards, and loops.
3. Uses the Hermes gateway API for LLM decisions.

Runs with maximum niceness and stops when peak hours begin.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
WORKSPACE = os.environ.get("AUTORESEARCH_WORKSPACE", "/home/newstex/workspace/autoresearch")
HERMES_API = os.environ.get("HERMES_API_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("AUTORESEARCH_MODEL", "deepseek-v4-flash")
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "50"))
TIMEOUT_PER_EXPERIMENT = 600  # 10 minutes max per experiment
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "14400"))  # 4 hours
OFFPEAK_END_HOUR = 7  # UTC hour when off-peak ends

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "autoresearch-loop.log")

def log(msg, level="INFO"):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{timestamp}] {level}: {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ──────────────────────────────────────────────────────────────
# LLM interaction via Hermes gateway
# ──────────────────────────────────────────────────────────────
def ask_llm(system_prompt, user_prompt, max_tokens=4096):
    """Send a prompt to the LLM and get the response."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    req = urllib.request.Request(
        f"{HERMES_API}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            log(f"LLM call failed (attempt {attempt+1}): {e}", "WARN")
            time.sleep(5)

    log("LLM call failed after 3 attempts", "ERROR")
    return None


def propose_experiment(context, history, n_try):
    """Ask the LLM to propose the next experiment."""
    sys_prompt = """You are optimizing nanochat train.py for lowest val_bpb in 5min.
Modify ANYTHING in train.py. DO NOT modify prepare.py or add deps.
Simpler is better. Blackwell GB10 (cap 12.1): FA3 FakeTensor bug, work around it.
Output JSON: {"idea": str, "patch": str (full train.py content), "patch_type": "replace|none", "expected_effect": str}"""

    user_prompt = f"""Experiment #{n_try}.
History (last 20):
{history[-20:] if history else 'None yet.'}

Code context (first 2000 chars):
{context[:2000]}

Propose next experiment. Output ONLY valid JSON."""
    
    response = ask_llm(sys_prompt, user_prompt)
    if not response:
        return None

    # Parse JSON from response
    try:
        # Find JSON block
        json_match = re.search(r"\{[^}]*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # Try parsing the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        log(f"Failed to parse LLM response as JSON: {response[:200]}", "ERROR")
        return None


def evaluate_run(run_log):
    """Parse the training run output for key metrics."""
    metrics = {}
    
    # Check for crash
    if "Traceback" in run_log or "Error" in run_log or "RuntimeError" in run_log:
        metrics["status"] = "crash"
        # Extract error message
        error_match = re.search(r"(\w+Error): (.+?)(?:\n|$)", run_log)
        if error_match:
            metrics["error"] = f"{error_match.group(1)}: {error_match.group(2)[:200]}"
        return metrics

    # Parse metrics
    for line in run_log.split("\n"):
        line = line.strip()
        if line.startswith("val_bpb:"):
            metrics["val_bpb"] = float(line.split()[-1])
        elif line.startswith("peak_vram_mb:"):
            metrics["peak_vram_mb"] = float(line.split()[-1])
        elif line.startswith("training_seconds:"):
            metrics["training_seconds"] = float(line.split()[-1])
        elif line.startswith("mfu_percent:"):
            metrics["mfu_percent"] = float(line.split()[-1])

    if "val_bpb" in metrics:
        metrics["status"] = "completed"
    else:
        metrics["status"] = "unknown"

    return metrics


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────
def main():
    log("Autoresearch loop starting")
    log(f"Workspace: {WORKSPACE}")
    log(f"Model: {MODEL}")
    log(f"Max experiments: {MAX_EXPERIMENTS}")

    os.chdir(WORKSPACE)
    start_time = time.time()

    # Read program.md for context
    program_md = Path("program.md").read_text()
    train_py = Path("train.py").read_text()
    context = f"program.md:\n{program_md[:2000]}\n\ntrain.py (first 100 lines):\n{train_py[:2000]}"

    # Read any existing results
    results_path = Path("results.tsv")
    if results_path.exists():
        history = results_path.read_text().strip().split("\n")[1:]  # skip header
    else:
        history = []
        # Write header
        results_path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

    # Create experiment branch
    tag = time.strftime("%b%d", time.gmtime()).lower()
    branch = f"autoresearch/{tag}"
    subprocess.run(["git", "checkout", "-b", branch], capture_output=True)

    # Experiment loop
    for n_try in range(1, MAX_EXPERIMENTS + 1):
        # Check time limits
        elapsed = time.time() - start_time
        if elapsed > MAX_RUN_SECONDS:
            log(f"Reached max run time ({elapsed:.0f}s >= {MAX_RUN_SECONDS}s), stopping")
            break

        # Check peak hours
        current_hour = time.gmtime().tm_hour
        if current_hour >= OFFPEAK_END_HOUR and current_hour < 23:
            log(f"Peak hour ({current_hour} UTC), stopping")
            break

        log(f"Experiment #{n_try} (elapsed: {elapsed:.0f}s)")

        # Propose experiment
        proposal = propose_experiment(context, history, n_try)
        if proposal is None:
            log("No proposal generated, stopping", "WARN")
            break

        idea = proposal.get("idea", "unknown")
        patch_content = proposal.get("patch", "none")
        patch_type = proposal.get("patch_type", "none")

        log(f"Idea: {idea}")

        if patch_type == "replace" and patch_content != "none":
            # Apply patch to train.py
            # For simplicity, write the full proposed train.py
            # (the LLM can output the full file or a diff)
            # We handle both cases
            current = Path("train.py").read_text()
            
            # Check if it's a diff/patch or full replacement
            if patch_content.startswith("---") or "old_string" in patch_content:
                # It's a diff — we'd need proper patch logic
                # For now, write the full content
                pass
            
            # Write the proposed content
            if len(patch_content) > 50:  # reasonable patch
                Path("train.py").write_text(patch_content)
                log(f"Applied patch to train.py")
            else:
                log(f"Patch too short ({len(patch_content)} chars), skipping")
                continue

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-a", "-m", f"exp #{n_try}: {idea}"],
            capture_output=True, text=True
        )
        commit_short = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
        log(f"Committed as {commit_short}")

        # Run experiment
        log("Running experiment...")
        run_result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True, text=True,
            timeout=TIMEOUT_PER_EXPERIMENT,
        )
        run_log = run_result.stdout + run_result.stderr

        # Evaluate
        metrics = evaluate_run(run_log)

        if metrics.get("status") == "crash":
            log(f"Crash: {metrics.get('error', 'unknown')}")
            status = "crash"
            val_bpb = 0.0
            memory_gb = 0.0
            # Revert
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)
            log("Reverted crash commit")
        elif metrics.get("status") == "completed":
            val_bpb = metrics["val_bpb"]
            memory_gb = metrics.get("peak_vram_mb", 0) / 1024
            log(f"val_bpb={val_bpb:.6f}, memory={memory_gb:.1f}GB")

            # Compare to previous best
            best_so_far = float("inf")
            for h in history[-20:]:
                parts = h.split("\t")
                if len(parts) >= 2 and parts[1] != "val_bpb":
                    try:
                        v = float(parts[1])
                        if v > 0:
                            best_so_far = min(best_so_far, v)
                    except ValueError:
                        pass

            if val_bpb < best_so_far and val_bpb > 0:
                status = "keep"
                log("Improved! Keeping commit")
            else:
                status = "discard"
                log("No improvement, reverting")
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)
        else:
            status = "unknown"
            val_bpb = 0.0
            memory_gb = 0.0
            log("Unknown experiment outcome")

        # Record result
        entry = f"{commit_short}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{idea}\n"
        with open("results.tsv", "a") as f:
            f.write(entry)
        history.append(entry.strip())

        # Brief pause between experiments
        time.sleep(5)

    log(f"Loop finished after {n_try} experiments ({time.time() - start_time:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
