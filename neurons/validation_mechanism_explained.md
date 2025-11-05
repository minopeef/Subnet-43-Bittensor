## System Architecture Overview

### 1. **Problem Configuration & Bins & Best Scores** (`envs.utils`)
- Defines different TSP problem types (TSP, mTSP, cmTSP, etc.)
- Each problem type has multiple size bins (small, medium, large, xlarge)
- Each bin has configurable thresholds for convergence and rewards
- Manages persistent storage of best agents and their metadata in `best_agents/` directory

### 2. **Performance Tracking** (`MinerPerformance`)
```python
@dataclass
class MinerPerformance:
    uid: int
    scores: List[float]  # Accumulates scores across rounds
```
- Tracks each miner's scores across multiple evaluation rounds
- Computes statistics: mean, variance, std_dev
- Determines if more runs needed based on old variance threshold (now replaced by ranking stability)

### 3. **Ranking Stability System** (`RankStability`)
**How it works:**
- **Tracks ranking history**: Stores the ordered list of miner UIDs (best→worst) after each round
- **Per-miner position variance**: Tracks how much each miner's rank position changes across rounds
- **Kendall Distance**: Measures pairwise ranking disagreement between consecutive rounds
  - If miner A beats B in round 1 but loses in round 2, that's a disagreement
  - Normalized to 0-1 (0 = perfect agreement, 1 = complete reversal)

**Convergence criteria** (all must pass OR Kendall alone passes):
```python
def has_converged(self,
                  mean_std_threshold: float = 2.0,    # Avg miner moves <2 positions
                  max_std_threshold: float = 6.0,     # No miner jumps >6 positions  
                  kendall_threshold: float = 0.02):   # ≤2% pairwise disagreement
```

### 4. **Adaptive Evaluation Loop** (`run_adaptive_evaluation`)
**Round-by-round process:**
```
Round 1: Generate problem → Evaluate all miners → Build ranking → Check stability
Round 2: Generate NEW problem → Evaluate all miners → Build ranking → Check stability
Round 3: ...
Continue until: ranking converges OR max_rounds reached
```

**Key features:**
- **Fresh problem each round**: Different problem instances test robustness
- **Concurrent evaluation**: All miners evaluated in parallel using `asyncio.create_task()`
- **Blocking calls handled properly**: `container.solve_problem()` is blocking, so wrapped in `loop.run_in_executor(None, func)`
- **Ranking stability tracking**: After each round, computes current ranking and adds to history

**Example flow:**
```
Round 1: Rankings = [uid3, uid1, uid2, uid4] → Add to history
Round 2: Rankings = [uid3, uid1, uid2, uid4] → Kendall distance = 0.0 (perfect match)
Round 3: Rankings = [uid1, uid3, uid2, uid4] → Small change detected
...
Round 8: mean_std=1.8, max_std=5.2, kendall=0.015 → CONVERGED!
```

### 5. **Reward Calculation** (`calculate_rewards`)

**Two-stage filtering:**

**Stage 1: Eligibility check**

```python
if historical_best == float('inf'):
    # First epoch - all valid scores eligible
    eligible.append((uid, perf.mean_score))
else:
    # Must beat historical best by improvement_threshold
    improvement = (historical_best - perf.mean_score) / historical_best
    if improvement >= improvement_threshold:  # e.g., 2% better
        eligible.append((uid, perf.mean_score))
```
```
Assuming a 5% improvement requirement threshold:

Example:
Epoch 1: No historical best → All valid solutions eligible
         → Winner UID 5 (score 100.0) saved to best_agents/

Epoch 2: Historical best = 100.0 (UID 5)
         → UID 7 scores 95.0 → improvement = 5% ✓ eligible
         → UID 7 wins → New best saved, old file removed

Epoch 3: Historical best = 95.0 (UID 7)
         → UID 12 scores 96.0 → improvement = -1% ✗ not eligible
         → No one beats threshold → No rewards this epoch
         → Historical best unchanged
```

**Stage 2: Winner-takes-all**
```python
if not eligible:
    return {uid: 0.0 for all}  # No one gets rewarded

best_uid, best_score = min(eligible, key=lambda x: x[1])
rewards[best_uid] = 1.0  # Only best miner gets reward
# All others get 0.0
```

**Historical tracking:**
```python
BEST_SCORES[problem_key] = best_score  # Update if improved
```

### 6. **Main Validator Loop**

**Full cycle:**
```
1. Connect to subtensor → Load metagraph → Get all UIDs

2. Pull agent code for each UID from GitHub gists
   └─> Store in agents/{uid}/{block}/agent.py

3. Create Container for each agent
   └─> Starts Docker container with agent code
   └─> Copies agent.py + tools.py into container
   └─> Starts FastAPI server in container

4. FOR EACH problem type (TSP, mTSP, etc.):
     FOR EACH bin (small, medium, large, xlarge):
       ├─> Run adaptive evaluation until ranking converges
       ├─> Calculate rewards (0 or 1 per miner)
       └─> Accumulate rewards in all_rewards[uid]

5. Aggregate rewards across all problems:
   final_weights[idx] = mean(all_rewards[uid])

6. Set weights on blockchain:
   sub.set_weights(wallet, netuid, weights, uids)

7. Cleanup containers and repeat
```

### 7. **Key Optimizations**
1. **No reward if no improvement**: 
   ```python
   if not eligible:  # No one beat historical best by threshold
       return {uid: 0.0 for all}
   ```
2. **Concurrent evaluation**: All miners evaluated in parallel per round
3. **Fresh problems each round**: Tests robustness across problem instances

### 8. **How Convergence Works**

**Example scenario:**
```
Round 1: [uid2, uid5, uid1, uid3, uid4] → History: [one ranking]
Round 2: [uid2, uid5, uid1, uid4, uid3] → Small swap uid3↔uid4
         Kendall = 0.1 (10% disagreement), mean_std = 0.5
         NOT converged (kendall > 0.02)

Round 3: [uid2, uid5, uid1, uid4, uid3] → No change
         Kendall = 0.0, mean_std = 0.33
         NOT converged yet (need more history)

...

Round 10: [uid2, uid5, uid1, uid4, uid3] → Stable for 8 rounds
          Kendall = 0.0, mean_std = 0.8, max_std = 2.1
          ✅ CONVERGED! (kendall < 0.02 AND mean_std < 2.0)
```

### 9. **Heartbeat & Watchdog**

```python
HEARTBEAT = time.monotonic()  # Updated frequently during execution

async def watchdog(timeout: int = 300):
    if time.monotonic() - HEARTBEAT > timeout:
        os._exit(1)  # Kill process if stalled >5min
```

Prevents the validator from getting stuck indefinitely.

---

### 10. **Restart Behavior**
On validator restart:
- Loads best_agents/metadata.json
- Verifies all agent files still exist
- Continues comparing against historical bests
- Saves winning agent files persistently
- Cleans up old agent files when superseded
- Maintains continuity across restarts

## Summary

Validation mechanism is implemented as a **sophisticated adaptive evaluation system**:

1. **Generates fresh problems each round** to test robustness
2. **Evaluates all miners concurrently** for efficiency  
3. **Tracks ranking stability** using Kendall distance + position variance
4. **Converges when rankings stabilize** (not just individual scores)
5. **Rewards only miners beating historical best** by improvement threshold
6. **Winner-takes-all**: Only the single best miner gets reward=1.0
7. **Aggregates across all problem types/bins** for final weights

The key insight: **Instead of just checking if individual miner scores stabilize, we check if the relative RANKING of miners stabilizes across different problem instances.** This is much more robust and meaningful for a competitive evaluation system.
