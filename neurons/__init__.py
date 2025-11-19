from __future__ import annotations
import os
import re
import time
import click
import random
import aiohttp
import asyncio
import aiofiles
import traceback
import logging
import bittensor as bt
import requests
import numpy as np
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Sequence, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from dotenv import load_dotenv
from sn43 import Container, tools
from envs.utils import PROBLEM_CONFIGS, ProblemBin, ProblemType, ProblemTypeConfig
from functools import partial

logger = logging.getLogger("neurons")

NETUID = 43

# ---------------- Persistent Storage ----------------
BEST_AGENTS_DIR = Path("best_agents")
BEST_AGENTS_DIR.mkdir(exist_ok=True)

@dataclass
class BestAgentRecord:
    """Record of the best agent for a problem type/bin."""
    problem_key: str
    uid: int
    score: float
    agent_file_path: str
    timestamp: float
    epoch: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BestAgentRecord':
        return cls(**data)

class BestAgentStorage:
    """Manages persistent storage of best agents."""
    
    def __init__(self, storage_dir: Path = BEST_AGENTS_DIR):
        self.storage_dir = storage_dir
        self.metadata_file = storage_dir / "metadata.json"
        self.records: Dict[str, BestAgentRecord] = {}
        self.load()
    
    def load(self) -> None:
        """Load metadata from disk."""
        if not self.metadata_file.exists():
            logger.info("No existing best agents metadata found")
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for problem_key, record_dict in data.items():
                record = BestAgentRecord.from_dict(record_dict)
                # Verify agent file still exists
                if Path(record.agent_file_path).exists():
                    self.records[problem_key] = record
                    logger.info(
                        f"Loaded best agent for {problem_key}: "
                        f"UID {record.uid}, score {record.score:.4f}"
                    )
                else:
                    logger.warning(
                        f"Agent file missing for {problem_key}: {record.agent_file_path}"
                    )
        except Exception as e:
            logger.error(f"Failed to load best agents metadata: {e}")
    
    def save(self) -> None:
        """Save metadata to disk."""
        try:
            data = {
                problem_key: record.to_dict()
                for problem_key, record in self.records.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.records)} best agent records")
        except Exception as e:
            logger.error(f"Failed to save best agents metadata: {e}")
    
    def get_best(self, problem_key: str) -> Optional[BestAgentRecord]:
        """Get the best agent record for a problem."""
        return self.records.get(problem_key)
    
    def update_best(
        self,
        problem_key: str,
        uid: int,
        score: float,
        agent_source_path: str,
        epoch: int
    ) -> bool:
        """
        Update the best agent for a problem if the new score is better.
        Returns True if updated, False otherwise.
        """
        existing = self.records.get(problem_key)
        
        # Check if this is actually better
        if existing and score >= existing.score:
            logger.info(
                f"{problem_key}: New score {score:.4f} not better than "
                f"existing {existing.score:.4f}"
            )
            return False
        
        # Copy agent file to persistent storage
        dest_filename = f"{problem_key}_uid{uid}_epoch{epoch}.py"
        dest_path = self.storage_dir / dest_filename
        
        try:
            shutil.copy2(agent_source_path, dest_path)
            logger.info(f"Saved best agent to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy agent file: {e}")
            return False
        
        # Create new record
        new_record = BestAgentRecord(
            problem_key=problem_key,
            uid=uid,
            score=score,
            agent_file_path=str(dest_path),
            timestamp=time.time(),
            epoch=epoch
        )
        
        # Remove old agent file if exists
        if existing and Path(existing.agent_file_path).exists():
            try:
                Path(existing.agent_file_path).unlink()
                logger.info(f"Removed old agent file: {existing.agent_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old agent file: {e}")
        
        self.records[problem_key] = new_record
        self.save()
        
        prev_score_str = f"{existing.score:.4f}" if existing else "N/A"
        logger.info(
            f"Updated best agent for {problem_key}: "
            f"UID {uid}, score {score:.4f} (prev: {prev_score_str})"
        )
        
        return True
    
    def get_all_best_scores(self) -> Dict[str, float]:
        """Get dictionary of all best scores."""
        return {
            problem_key: record.score
            for problem_key, record in self.records.items()
        }

# Global storage instance
BEST_AGENT_STORAGE = BestAgentStorage()

# ---------------- MinerPerformance ----------------
@dataclass
class MinerPerformance:
    """Track performance for a miner on a specific problem."""
    uid: int
    scores: List[float] = field(default_factory=list)
    
    @property
    def run_count(self) -> int:
        return len(self.scores)
    
    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float('inf')
    
    @property
    def variance(self) -> float:
        return float(np.var(self.scores)) if len(self.scores) > 1 else float('inf')
    
    @property
    def std_dev(self) -> float:
        return float(np.std(self.scores)) if len(self.scores) > 1 else float('inf')
    
    def add_score(self, score: float) -> None:
        self.scores.append(score)
    
    def needs_more_runs(self, variance_threshold: float, min_runs: int) -> bool:
        if self.run_count < min_runs:
            return True
        return self.variance > variance_threshold

# ---------------- Ranking Stability ----------------
def normalized_kendall_distance(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    n = len(rank_a)
    pos_a = {miner: i for i, miner in enumerate(rank_a)}
    pos_b = {miner: i for i, miner in enumerate(rank_b)}
    discord = 0
    total = n * (n - 1) // 2
    miners = list(rank_a)
    for i in range(n):
        for j in range(i+1, n):
            m1 = miners[i]
            m2 = miners[j]
            if pos_b[m1] > pos_b[m2]:
                discord += 1
    return discord / total

class RankStability:
    def __init__(self, uids: List[int], min_window: int = 10):
        """
        Args:
            uids: List of actual miner UIDs (e.g., [4, 5, 6, 130])
            min_window: minimum window size
        """
        self.uids = list(uids)
        self.n = len(self.uids)
        self.uid_to_idx = {uid: idx for idx, uid in enumerate(self.uids)}
        self.min_window = min_window
        self.history = []

    def add_ranking(self, ranking: Sequence[int]):
        """ranking: list/tuple of UIDs in order from best (0) to worst (n-1)."""
        assert len(ranking) == self.n, f"Expected {self.n} UIDs, got {len(ranking)}"
        assert all(uid in self.uid_to_idx for uid in ranking), \
            f"Invalid UIDs in ranking: {set(ranking) - set(self.uids)}"
        self.history.append(list(ranking))

    def per_miner_std(self):
        """Return array of std dev of rank positions per miner across history window."""
        if len(self.history) < 2:
            return np.full(self.n, np.inf)
        
        T = len(self.history)
        pos = np.zeros((T, self.n), dtype=float)
        
        for t, ranking in enumerate(self.history):
            for rank_position, uid in enumerate(ranking):
                miner_idx = self.uid_to_idx[uid]
                pos[t, miner_idx] = rank_position
        
        return np.std(pos, axis=0, ddof=0)

    def aggregate_per_miner_stats(self):
        stds = self.per_miner_std()
        return {
            "mean_std": float(np.mean(stds)),
            "median_std": float(np.median(stds)),
            "max_std": float(np.max(stds)),
            "stds": stds  # numpy array
        }

    def avg_normalized_kendall(self):
        """Average of normalized Kendall distances between consecutive rankings in window."""
        if len(self.history) < self.min_window:
            return 1.0  # completely unstable / not enough data
        dists = []
        for i in range(len(self.history)-1):
            d = normalized_kendall_distance(self.history[i], self.history[i+1])
            dists.append(d)
        return float(np.mean(dists))

    def has_converged(self,
                      mean_std_threshold: float = 2.0,
                      max_std_threshold: float = 6.0,
                      kendall_threshold: float = 0.02) -> bool:
        """
        Default thresholds:
         - mean_std_threshold=2.0: on average miner moves <2 positions (tight)
         - max_std_threshold=6.0: no miner jumps more than ~6 positions
         - kendall_threshold=0.02: consecutive rankings disagree on <=2% of pairwise orders
        """
        if len(self.history) < self.min_window:
            return False
        
        stats = self.aggregate_per_miner_stats()
        avg_kendall = self.avg_normalized_kendall()
        ok_mean = stats["mean_std"] < mean_std_threshold
        ok_max = stats["max_std"] < max_std_threshold
        ok_kendall = avg_kendall < kendall_threshold
        # require both per-miner stability and pairwise stability
        return (ok_mean and ok_max) and ok_kendall

# ---------------- Subtensor ----------------
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR is None:
        logger.debug("Making Bittensor connection...")
        if bt is None:
            raise RuntimeError("bittensor not installed")
        SUBTENSOR = bt.async_subtensor(os.getenv('SUBTENSOR_ENDPOINT', 'wss://lite.sub.latent.to:443'))
        try:
            await SUBTENSOR.initialize()
            logger.debug("Connected")
        except Exception as e:
            logger.error(f"Failed to initialize subtensor: {e}")
            os._exit(1)
    return SUBTENSOR

# ---------------- Get Agent ----------------
async def pull_agent(uid: int) -> Optional[str]:
    try:
        sub = await get_subtensor()
        commit = await sub.get_revealed_commitment(netuid=NETUID, uid=uid)
        g = commit[0][1]
        block = commit[0][0]
        if g.startswith("http") and "api.github.com" not in g:
            g = f"https://api.github.com/gists/{g.rstrip('/').split('/')[-1]}"
        if not g.startswith("http"):
            g = f"https://api.github.com/gists/{g}"
        async with aiohttp.ClientSession() as s:
            async with s.get(g) as r:
                data = await r.json()
            meta = next(iter(data["files"].values()))
            content = meta.get("content")
            if content is None or meta.get("truncated"):
                async with s.get(meta["raw_url"]) as r:
                    content = await r.text()
        dir = f"agents/{uid}/{block}/"
        Path(dir).mkdir(parents=True, exist_ok=True)
        name = f"{dir}agent.py"
        async with aiofiles.open(name, "w", encoding="utf-8") as f:
            await f.write(content or "")
        resolved_path = str(Path(name).resolve())
        return resolved_path
    except Exception as e:
        logger.warning(f'Failed pulling agent on UID: {uid} with error: {e}')
        return None

# ---------------- CLI ----------------
@click.group()
@click.option('--log-level', type=click.Choice(['CRITICAL','ERROR','WARNING','INFO','DEBUG'], case_sensitive=False), default=None, help='Logging level (or set LOG_LEVEL env)')
def cli(log_level: Optional[str]):
    global NETUID
    load_dotenv(override=True)
    level_name = (log_level or os.getenv('LOG_LEVEL') or 'INFO').upper()
    NETUID = int(os.getenv('NETUID', NETUID))
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@cli.command("push")
@click.argument("path", default="agents/base_agent.py")
def push(path: str):
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value
    coldkey = require_env("BT_WALLET_COLD")
    hotkey = require_env("BT_WALLET_HOT")
    github_token = require_env("GITHUB_TOKEN")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    async def main():
        logger.info('Loading chain state ...')
        sub = await get_subtensor()
        metagraph = await sub.metagraph(NETUID)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            logger.warning(f"Not registered, first register your wallet `btcli subnet register --netuid {NETUID} --wallet.name {coldkey} --hotkey {hotkey}`")
            os._exit(1)
        logger.info(f'UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}')

        with open(path, 'r') as f:
            content = f.read()
        scheme = "token" if github_token.startswith(("ghp_", "github_pat_")) else "Bearer"
        headers = {
            "Authorization": f"{scheme} {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "neurons-cli"
        }
        gist_data = {"description": "Agent code", "public": True, "files": {os.path.basename(path): {"content": content}}}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.github.com/gists", json=gist_data, headers=headers) as resp:
                if resp.status != 201:
                    try:
                        error_json = await resp.json()
                        error_msg = error_json.get("message") or str(error_json)
                    except Exception:
                        error_msg = await resp.text()
                    raise RuntimeError(
                        f"Failed to create gist ({resp.status}): {error_msg}. Ensure your GITHUB_TOKEN is valid and has 'gist' scope, visit: https://github.com/settings/tokens/new"
                    )
                gist_url = (await resp.json())["html_url"]
                logger.info(f"Created gist: {gist_url}")

        await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=gist_url, blocks_until_reveal=1)
        logger.info(f"Committed gist URL to blockchain.")

    asyncio.run(main())

@cli.command("pull")
@click.argument("uid", type=int, required=False)
def pull(uid: int = None):
    if uid is not None:
        asyncio.run(pull_agent(uid))
    else:
        async def pull_all():
            sub = await get_subtensor()
            metagraph = await sub.metagraph(NETUID)
            for uid in metagraph.uids:
                await pull_agent(int(uid))
        asyncio.run(pull_all())

# ---------------- Watchdog ----------------
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(max(1, timeout // 3))
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)

# ---------------- Validator Helper Functions ----------------
async def evaluate_miner_on_problem(
    container: Container,
    uid: int,
    problem_data: Dict[str, Any],
    timeout: float = 60.0
) -> Optional[Dict[str, Any]]:
    try:
        try:
            test_response = requests.get(f"{container.base_url}/healthz", timeout=5.0)
        except Exception as e:
            return None
        
        loop = asyncio.get_running_loop()
        func = partial(container.solve_problem, problem_data, timeout)

        result = await loop.run_in_executor(None, func)
        
        if result is None:
            return None
            
        return result
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug(f"UID {uid}: Error in solve_problem - {e}")
        return None

async def run_adaptive_evaluation(
    uids: List[int],
    containers: Dict[int, Container],
    agent_paths: Dict[int, str],  # Added: track agent file paths
    problem_type_str: str,
    bin_config: ProblemBin,
    config: ProblemTypeConfig,
    rank_window: int = 10,
    mean_std_threshold: float = 2.0,
    max_std_threshold: float = 6.0,
    kendall_threshold: float = 0.02,
    max_rounds: int = 20
) -> Tuple[Dict[int, MinerPerformance], Dict[int, str]]:  # Returns performances and agent paths
    """
    Run adaptive evaluation until ranking stability converges or max_rounds is reached.
    Returns both performances and the agent file paths for each UID.
    """
    global HEARTBEAT

    # initialize trackers
    performances: Dict[int, MinerPerformance] = {
        uid: MinerPerformance(uid=uid) for uid in uids
    }
    stability = RankStability(uids=uids, min_window=rank_window)

    round_num = 0

    while round_num < max_rounds:
        round_num += 1
        HEARTBEAT = time.monotonic()

        # generate fresh problem each round (so ranking stability measures across problems)
        n_nodes = random.randint(bin_config.min_nodes, bin_config.max_nodes)
        logger.info(f"[Round {round_num}] Generating {problem_type_str} problem with {n_nodes} nodes (bin: {bin_config.bin_id})")
        problem_info = tools.generate_problem(
            problem_type_str=problem_type_str,
            n_nodes=n_nodes
        )

        if "problem_data" not in problem_info:
            logger.error(f"[Round {round_num}] Failed to generate problem; skipping round")
            continue

        problem_data = problem_info["problem_data"]

        # Launch all evaluations concurrently (each evaluation uses threadpool internally)
        tasks = {}
        for uid in uids:
            container = containers.get(uid)
            if container is None:
                tasks[uid] = None
                continue
            tasks[uid] = asyncio.create_task(
                evaluate_miner_on_problem(container, uid, problem_data, timeout=300.0)
            )

                # gather results for ALL uids
        for uid in uids:  # Iterate over ALL uids, not just tasks.keys()
            HEARTBEAT = time.monotonic()
            
            task = tasks.get(uid)
            if task is None:
                # No container for this uid
                performances[uid].add_score(float('inf'))
                logger.debug(f"[Round {round_num}] UID {uid}: No container, scored inf")
                continue
                
            try:
                result = await task
            except Exception as e:
                logger.debug(f"[Round {round_num}] UID {uid}: evaluation exception: {e}")
                performances[uid].add_score(float('inf'))
                continue

            if result is None:
                performances[uid].add_score(float('inf'))
                logger.debug(f"[Round {round_num}] UID {uid}: No result, scored inf")
                continue

            # Score the solution
            if isinstance(result, dict) and result.get("solution"):
                try:
                    score_result = tools.score_solution(solution=result["solution"], problem_data=problem_data)
                    score = score_result.get("score", float('inf'))
                    performances[uid].add_score(score)
                    perf = performances[uid]
                    logger.info(
                        f"[Round {round_num}] UID {uid}: score={score:.4f}, runs={perf.run_count}, mean={perf.mean_score:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"[Round {round_num}] UID {uid}: Scoring error - {e}")
                    performances[uid].add_score(float('inf'))
            else:
                logger.warning(f"[Round {round_num}] UID {uid}: Invalid solution format")
                performances[uid].add_score(float('inf'))

        # Build ranking this round using mean_score (lower is better).
        # If two miners have identical mean_score, break ties by uid to keep deterministic order.
        sorted_miners = sorted(
            uids,
            key=lambda uid: (performances[uid].mean_score, uid)
        )

        # `sorted_miners` is list of uids from best -> worst
        stability.add_ranking(sorted_miners)

        # Log stability diagnostics
        if len(stability.history) >= 2:
            agg = stability.aggregate_per_miner_stats()
            avg_kendall = stability.avg_normalized_kendall()
            logger.info(
                f"[Round {round_num}] rank_window={rank_window} mean_std={agg['mean_std']:.4f} "
                f"median_std={agg['median_std']:.4f} max_std={agg['max_std']:.4f} "
                f"avg_kendall={avg_kendall:.6f}"
            )
        else:
            logger.info(f"[Round {round_num}] rank_window={rank_window} (need 2+ rounds for stability metrics)")

        # Check convergence
        if stability.has_converged(
            mean_std_threshold=mean_std_threshold,
            max_std_threshold=max_std_threshold,
            kendall_threshold=kendall_threshold
        ):
            logger.info(
                f"[Round {round_num}] Rankings converged with thresholds: "
                f"mean_std<{mean_std_threshold}, max_std<{max_std_threshold}, kendall<{kendall_threshold}"
            )
            break

    # End loop
    return performances, agent_paths

def calculate_rewards(
    performances: Dict[int, MinerPerformance],
    agent_paths: Dict[int, str],
    problem_key: str,
    improvement_threshold: float,
    epoch: int
) -> Dict[int, float]:
    """
    Calculate rewards based on performance.
    Only miners beating the historical best by improvement_threshold get rewards.
    Updates best agent storage if a new best is found.
    """
    global BEST_AGENT_STORAGE
    
    rewards = {uid: 0.0 for uid in performances.keys()}
    
    # Get historical best from persistent storage
    best_record = BEST_AGENT_STORAGE.get_best(problem_key)
    historical_best = best_record.score if best_record else float('inf')
    
    if best_record:
        logger.info(
            f"{problem_key}: Historical best - UID {best_record.uid}, "
            f"score {historical_best:.4f} (epoch {best_record.epoch})"
        )
    else:
        logger.info(f"{problem_key}: No historical best (first epoch)")
    
    # Find eligible miners
    eligible = []
    for uid, perf in performances.items():
        if perf.mean_score == float('inf'):
            logger.debug(f"UID {uid}: Excluded (inf score - failed/broken agent)")
            continue
        
        if historical_best == float('inf'):
            # First epoch - all valid scores are eligible
            eligible.append((uid, perf.mean_score))
        else:
            # Must beat historical best by improvement threshold
            improvement = (historical_best - perf.mean_score) / historical_best
            if improvement >= improvement_threshold:
                eligible.append((uid, perf.mean_score))
                logger.info(
                    f"UID {uid}: Eligible with score {perf.mean_score:.4f}, "
                    f"improvement {improvement*100:.2f}%"
                )
            else:
                logger.debug(
                    f"UID {uid}: score {perf.mean_score:.4f}, "
                    f"improvement {improvement*100:.2f}% (below threshold {improvement_threshold*100:.1f}%)"
                )

    if not eligible:
        logger.info(f"No miners eligible for {problem_key} - no rewards this epoch")
        return rewards
    
    # Award best miner
    best_uid, best_score = min(eligible, key=lambda x: x[1])
    rewards[best_uid] = 1.0
    
    logger.info(
        f"{problem_key}: Winner UID {best_uid} with score {best_score:.4f}"
    )
    
    # Update persistent storage if this is a new best
    if best_score < historical_best:
        agent_path = agent_paths.get(best_uid)
        if agent_path and Path(agent_path).exists():
            BEST_AGENT_STORAGE.update_best(
                problem_key=problem_key,
                uid=best_uid,
                score=best_score,
                agent_source_path=agent_path,
                epoch=epoch
            )
        else:
            logger.warning(
                f"Cannot save best agent for UID {best_uid}: "
                f"agent path not found or invalid"
            )

    return rewards

# ---------------- Main Validator ----------------
@cli.command("validator")
def validator():
    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value
    
    coldkey = require_env("BT_WALLET_COLD")
    hotkey = require_env("BT_WALLET_HOT")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)
    logger.debug(f"Validator initialized with wallet: {coldkey}/{hotkey}")

    async def _run():
        global HEARTBEAT

        # IMPORTANT: Ensure local sn43 server is running with TSP tools loaded
        from sn43 import ensure_server_running, load_env
        
        # Load the TSP environment which registers tools
        try:
            env_spec = load_env("envs/tsp")  # or just "tsp" if using module path
        except Exception as e:
            raise
        
        # Ensure the server is running
        ensure_server_running(host="0.0.0.0", port=5005)

        # Issue a token for the validator to call its own tools
        from sn43 import issue_token
        validator_token = issue_token(ttl_s=365*24*3600, per_token_limit=200)
        os.environ["sn43_TOKEN"] = validator_token
        
        epoch = 0
        
        while True:
            try:
                epoch += 1
                HEARTBEAT = time.monotonic()

                sub = await get_subtensor()

                metagraph = await sub.metagraph(NETUID)
                uids = [int(uid) for uid in metagraph.uids]
                
                # Initialize containers for all miners
                containers: Dict[int, Container] = {}
                agent_paths: Dict[int, str] = {}
                
                for uid in uids:
                    HEARTBEAT = time.monotonic()
                    
                    gen_tmp_file = await pull_agent(uid)
                    HEARTBEAT = time.monotonic()

                    if gen_tmp_file is None:
                        logger.warning(f"Could not pull agent for UID {uid}, skipping")
                        continue
                    
                    agent_paths[uid] = gen_tmp_file  # Store path for later
                    
                    container = None
                    try:
                        HEARTBEAT = time.monotonic()
                        container = Container(gen_tmp_file, token=validator_token)
                        HEARTBEAT = time.monotonic()
                        containers[uid] = container
                    except Exception as e:
                        # Extra safety: if container object exists but init failed, try destroying
                        if container is not None and hasattr(container, 'container_id') and container.container_id:
                            try:
                                container.destroy()
                            except Exception as cleanup_e:
                                logger.error(f"Additional cleanup failed: {cleanup_e}")
                        
                        HEARTBEAT = time.monotonic()
                        continue
                
                logger.info(f"Initialized {len(containers)} containers")
                
                # Iterate through all problem types and bins
                all_rewards: Dict[int, List[float]] = defaultdict(list)
                
                for problem_type, config in PROBLEM_CONFIGS.items():

                    for bin_config in config.bins:
                        HEARTBEAT = time.monotonic()
                        problem_type_str = problem_type.value if isinstance(problem_type, ProblemType) else problem_type

                        problem_key = f"{problem_type}_{bin_config.bin_id}"
                        
                        # Run adaptive evaluation
                        performances, paths = await run_adaptive_evaluation(
                            uids=uids,
                            containers=containers,
                            agent_paths=agent_paths,
                            problem_type_str=problem_type_str,
                            bin_config=bin_config,
                            config=config,
                            rank_window=config.rank_window,
                            mean_std_threshold=config.mean_std_threshold,
                            max_std_threshold=config.max_std_threshold,
                            kendall_threshold=config.kendall_threshold,
                            max_rounds=config.max_rounds
                        )

                        HEARTBEAT = time.monotonic()
                        
                        # Calculate rewards (this will update best agent storage)
                        rewards = calculate_rewards(
                            performances=performances,
                            agent_paths=paths,
                            problem_key=problem_key,
                            improvement_threshold=config.improvement_threshold,
                            epoch=epoch
                        )

                        HEARTBEAT = time.monotonic()
                        
                        # Accumulate rewards
                        for uid, reward in rewards.items():
                            all_rewards[uid].append(reward)

                        HEARTBEAT = time.monotonic()
                        
                        # Log results
                        logger.info(f"\nResults for {problem_key}:")
                        for uid in sorted(performances.keys()):
                            perf = performances[uid]
                            reward = rewards.get(uid, 0.0)
                            logger.info(
                                f"UID {uid}: mean={perf.mean_score:.4f}, "
                                f"runs={perf.run_count}, reward={reward:.1f}"
                            )
                        
                        HEARTBEAT = time.monotonic()
                
                # Cleanup containers
                for container in containers.values():
                    try:
                        container.destroy()
                    except Exception as e:
                        logger.warning(f"Failed to destroy container: {e}")
                
                # Aggregate rewards (average across all problem types/bins)
                final_weights = [0.0] * len(uids)
                for idx, uid in enumerate(uids):
                    if uid in all_rewards and all_rewards[uid]:
                        final_weights[idx] = float(np.mean(all_rewards[uid]))
                
                
                # Set weights on chain
                logger.info("Setting weights on chain...")
                await sub.set_weights(
                    wallet=wallet,
                    netuid=NETUID,
                    weights=final_weights,
                    uids=uids,
                    wait_for_inclusion=False,
                    wait_for_finalization=False
                )
                logger.info(f"Weights successfully set for epoch {epoch}")

            except asyncio.CancelledError:
                logger.debug("Validator loop cancelled")
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Runner error: {e}; retrying...")
                await asyncio.sleep(5)

    async def main():
        logger.debug("Starting validator with watchdog")
        await asyncio.gather(
            _run(),
            watchdog(timeout=600),
            return_exceptions=True
        )
    
    asyncio.run(main())
