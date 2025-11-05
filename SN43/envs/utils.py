"""
Problem type definitions and binning configurations for TSP variants.
Defines bin sizes and problem generation parameters for each problem type.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum

class ProblemType(Enum):
    TSP = "Metric TSP"
    MTSP = "Metric mTSP"
    MDMTSP = "Metric mTSP"
    CMDMTSP = "Metric cmTSP"
    RCMDMTSP = "Metric cmTSP"
    RCMDMTSPTW = "Metric cmTSPTW"

@dataclass
class ProblemBin:
    """Defines a bin size range for a problem type."""
    bin_id: str
    min_nodes: int
    max_nodes: int
    weight: float  # Probability weight for selecting this bin
    
@dataclass
class ProblemTypeConfig:
    """Configuration for a specific problem type."""
    problem_type: ProblemType
    bins: List[ProblemBin]
    min_runs: int  # Minimum number of times to evaluate each miner
    variance_threshold: float  # Stop adding runs when variance drops below this
    improvement_threshold: float  # Min % improvement over baseline to get rewards
    selection_weight: float  # Probability weight for selecting this problem type

    # Optional adaptive evaluation thresholds
    rank_window: int = 10
    mean_std_threshold: float = 2.0
    max_std_threshold: float = 6.0
    kendall_threshold: float = 0.02
    max_rounds: int = 1000

# Define bin configurations for each problem type
TSP_BINS = [
    ProblemBin("small", 100, 500, 0.2),
    ProblemBin("medium", 500, 1500, 0.3),
    ProblemBin("large", 1500, 3000, 0.3),
    ProblemBin("xlarge", 3000, 5000, 0.2),
]

MTSP_BINS = [
    ProblemBin("small", 100, 400, 0.3),
    ProblemBin("medium", 400, 1000, 0.4),
    ProblemBin("large", 1000, 2000, 0.3),
]

MDMTSP_BINS = [
    ProblemBin("small", 100, 400, 0.3),
    ProblemBin("medium", 400, 1000, 0.4),
    ProblemBin("large", 1000, 2000, 0.3),
]

CMDMTSP_BINS = [
    ProblemBin("small", 100, 400, 0.3),
    ProblemBin("medium", 400, 1000, 0.4),
    ProblemBin("large", 1000, 2000, 0.3),
]

RCMDMTSP_BINS = [
    ProblemBin("small", 100, 400, 0.4),
    ProblemBin("medium", 400, 800, 0.4),
    ProblemBin("large", 800, 1500, 0.2),
]

RCMDMTSPTW_BINS = [
    ProblemBin("small", 100, 400, 0.4),
    ProblemBin("medium", 400, 800, 0.4),
    ProblemBin("large", 800, 1500, 0.2),
]

# Master configuration for all problem types
PROBLEM_CONFIGS = {
    ProblemType.TSP: ProblemTypeConfig(  # Metric TSP
        problem_type=ProblemType.TSP,
        bins=TSP_BINS,
        min_runs=3,
        variance_threshold=0.02,       # tighter, because TSP is stable
        improvement_threshold=0.01,    # small improvement expected
        selection_weight=0.15,
        mean_std_threshold=1.2,
        kendall_threshold=0.01,        # rank stability should be high
        max_rounds=600
    ),
    ProblemType.MTSP: ProblemTypeConfig(  # Multi-TSP
        problem_type=ProblemType.MTSP,
        bins=MTSP_BINS,
        min_runs=4,
        variance_threshold=0.03,       # slightly looser, multiple salesmen increases variance
        improvement_threshold=0.02,
        selection_weight=0.18,
        mean_std_threshold=1.5,
        kendall_threshold=0.015,
        max_rounds=700
    ),
    ProblemType.MDMTSP: ProblemTypeConfig(  # Multi-Depot Multi-TSP
        problem_type=ProblemType.MDMTSP,
        bins=MDMTSP_BINS,
        min_runs=5,
        variance_threshold=0.035,
        improvement_threshold=0.025,
        selection_weight=0.18,
        mean_std_threshold=1.8,
        kendall_threshold=0.017,
        max_rounds=800
    ),
    ProblemType.CMDMTSP: ProblemTypeConfig(  # Constrained MDMTSP
        problem_type=ProblemType.CMDMTSP,
        bins=CMDMTSP_BINS,
        min_runs=6,
        variance_threshold=0.04,
        improvement_threshold=0.03,
        selection_weight=0.2,
        mean_std_threshold=2.0,
        kendall_threshold=0.018,
        max_rounds=900
    ),
    ProblemType.RCMDMTSP: ProblemTypeConfig(  # Randomized Constrained MDMTSP
        problem_type=ProblemType.RCMDMTSP,
        bins=RCMDMTSP_BINS,
        min_runs=7,
        variance_threshold=0.045,
        improvement_threshold=0.035,
        selection_weight=0.22,
        mean_std_threshold=2.2,
        kendall_threshold=0.02,
        max_rounds=1000
    ),
    ProblemType.RCMDMTSPTW: ProblemTypeConfig(  # Randomized Constrained MDMTSP with Time Windows
        problem_type=ProblemType.RCMDMTSPTW,
        bins=RCMDMTSPTW_BINS,
        min_runs=8,
        variance_threshold=0.05,       # most complex → allow wider variance
        improvement_threshold=0.04,
        selection_weight=0.25,
        mean_std_threshold=2.5,
        kendall_threshold=0.025,       # allow slightly more rank movement
        max_rounds=1200
    ),
}


def select_problem_type() -> ProblemTypeConfig:
    """Randomly select a problem type based on weights."""
    import random
    types = list(PROBLEM_CONFIGS.values())
    weights = [cfg.selection_weight for cfg in types]
    return random.choices(types, weights=weights)[0]

def select_bin(config: ProblemTypeConfig) -> ProblemBin:
    """Randomly select a bin from a problem type config."""
    import random
    weights = [b.weight for b in config.bins]
    return random.choices(config.bins, weights=weights)[0]