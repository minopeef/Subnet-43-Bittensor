"""
Tools for TSP problem generation and solution scoring.
These run in the validator's container and are called by miner agents.
"""
import sn43
import random
import math
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from envs.utils import ProblemType, ProblemTypeConfig, select_problem_type, select_bin

# Import from your existing codebase
from sn43.graphite.protocol import (
    GraphV2Problem, GraphV2ProblemMulti, GraphV2ProblemMultiConstrained,
    GraphV2ProblemMultiConstrainedTW
)
from sn43.graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
from sn43.graphite.utils.graph_utils import (
    get_multi_minmax_tour_distance, get_multi_minmax_tour_distance_tw,
    is_valid_solution, get_tour_distance
)
from sn43.graphite.solvers.greedy_solver_multi_4 import NearestNeighbourMultiSolver4
from sn43.graphite.data.dataset_utils import load_default_dataset

# Global dataset storage
LOADED_DATASETS = {}

def init_datasets():
    """Initialize datasets on container startup."""
    global LOADED_DATASETS
    print("init_datasets called", flush=True)
    class MockNeuron:
        pass
    neuron = MockNeuron()
    print("About to load_default_dataset", flush=True)
    load_default_dataset(neuron)
    print(f"Datasets loaded: {len(neuron.loaded_datasets)} datasets", flush=True)
    LOADED_DATASETS = neuron.loaded_datasets
    print("init_datasets complete", flush=True)

@sn43.tool
async def generate_problem(
    ctx: sn43.Context,
    problem_type_str: Optional[str] = None,
    n_nodes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a TSP problem instance.
    
    Args:
        ctx: Execution context
        problem_type: One of TSP, MTSP, MDMTSP, CMTSP, CMTSPTW (or None for random)
        n_nodes: Number of nodes (or None for random within bin)
    
    Returns:
        Dictionary with problem specification
    """
    import logging
    
    # Select problem type if not specified
    if problem_type_str is None:
        config = select_problem_type()
    else:
        from envs.utils import PROBLEM_CONFIGS
        config = PROBLEM_CONFIGS[ProblemType.from_value(problem_type_str)]
    
    # Select bin and n_nodes
    selected_bin = select_bin(config)
    if n_nodes is None:
        n_nodes = random.randint(selected_bin.min_nodes, selected_bin.max_nodes)
    
    # Select dataset
    dataset_ref = random.choice(list(LOADED_DATASETS.keys()))
    selected_ids = random.sample(range(len(LOADED_DATASETS[dataset_ref]['data'])), n_nodes)
    
    # Randomly select salesmen
    m = random.randint(2, 10)

    # Create problem based on type
    if config.problem_type == ProblemType.TSP:
        problem = GraphV2Problem(
            problem_type="Metric TSP",
            n_nodes=n_nodes,
            selected_ids=selected_ids,
            cost_function="Geom",
            dataset_ref=dataset_ref,
        )
    
    elif config.problem_type == ProblemType.MTSP:
        problem = GraphV2ProblemMulti(
            problem_type="Metric mTSP",
            n_nodes=n_nodes,
            selected_ids=selected_ids,
            cost_function="Geom",
            dataset_ref=dataset_ref,
            n_salesmen=m,
            depots=[0 for _ in range(m)],
            single_depot=True
        )
    
    elif config.problem_type == ProblemType.MDMTSP:
        problem = GraphV2ProblemMulti(
            problem_type="Metric mTSP",
            n_nodes=n_nodes,
            selected_ids=selected_ids,
            cost_function="Geom",
            dataset_ref=dataset_ref,
            n_salesmen=m,
            depots=sorted(random.sample(list(range(n_nodes)), k=m)),
            single_depot=False
        )
    
    elif config.problem_type == ProblemType.CMDMTSP:
        problem = await _generate_cmdmtsp(n_nodes, selected_ids, dataset_ref)

    elif config.problem_type == ProblemType.RCMDMTSP:
        problem = await _generate_rcmdmtsp(n_nodes, selected_ids, dataset_ref)
    
    elif config.problem_type == ProblemType.RCMDMTSPTW:
        problem = await _generate_rcmdmtsptw(n_nodes, selected_ids, dataset_ref)
    
    # Store problem in context
    ctx.set("current_problem", problem)
    ctx.set("problem_config", config.__dict__)
    ctx.set("bin_id", selected_bin.bin_id)

    problem.edges = await recreate_edges(problem)
    if problem.edges is not None:
        problem.edges = problem.edges.tolist()
    problem_dict = problem.dict()
    
    return {
        "problem_type": config.problem_type.value,
        "bin_id": selected_bin.bin_id,
        "n_nodes": n_nodes,
        "problem_data": problem_dict
    }

async def _generate_cmdmtsp(n_nodes: int, selected_ids: List[int], dataset_ref: str):
    """Generate a constrained mTSP with feasibility check."""
    solution_found = False
    attempts = 0
    max_attempts = 10
    
    while not solution_found and attempts < max_attempts:
        attempts += 1
        m = random.randint(2, 10)
        depots = sorted(random.sample(list(range(n_nodes)), k=m))
        demand = [1 for _ in range(n_nodes)]
        for depot in depots:
            demand[depot] = 0
        
        total_demand_padded = sum(demand) + 9 * m
        constraint = [(math.ceil(total_demand_padded/m) + random.randint(0, int(total_demand_padded/m * 0.3)) - 
                      random.randint(0, int(total_demand_padded/m * 0.2))) for _ in range(m-1)]
        last_constraint = total_demand_padded - sum(constraint) + random.randint(
            int(total_demand_padded/m * 0.2), int(total_demand_padded/m * 0.3))
        constraint.append(last_constraint)
        
        test_problem = GraphV2ProblemMultiConstrained(
            problem_type="Metric cmTSP",
            n_nodes=n_nodes,
            selected_ids=selected_ids,
            cost_function="Geom",
            dataset_ref=dataset_ref,
            n_salesmen=m,
            depots=depots,
            single_depot=False,
            demand=demand,
            constraint=constraint
        )
        
        # Verify feasibility with greedy solver
        test_problem.edges = await recreate_edges(test_problem)
        solver = NearestNeighbourMultiSolver4(problem_types=[test_problem])
        try:
            route = await asyncio.wait_for(solver.solve_problem(test_problem), timeout=10)
            if route is not None:
                solution_found = True
                test_problem.edges = None
                return test_problem
        except asyncio.TimeoutError:
            continue
    
    raise RuntimeError("Failed to generate feasible cmTSP after max attempts")

async def _generate_rcmdmtsp(n_nodes: int, selected_ids: List[int], dataset_ref: str):
    """Generate a constrained mTSP with feasibility check."""
    solution_found = False
    attempts = 0
    max_attempts = 10
    
    while not solution_found and attempts < max_attempts:
        attempts += 1
        m = random.randint(2, 10)
        depots = sorted(random.sample(list(range(n_nodes)), k=m))
        demand = [random.randint(1, 9) for _ in range(n_nodes)]
        for depot in depots:
            demand[depot] = 0
        
        total_demand_padded = sum(demand) + 9 * m
        constraint = [(math.ceil(total_demand_padded/m) + random.randint(0, int(total_demand_padded/m * 0.3)) - 
                      random.randint(0, int(total_demand_padded/m * 0.2))) for _ in range(m-1)]
        last_constraint = total_demand_padded - sum(constraint) + random.randint(
            int(total_demand_padded/m * 0.2), int(total_demand_padded/m * 0.3))
        constraint.append(last_constraint)
        
        test_problem = GraphV2ProblemMultiConstrained(
            problem_type="Metric cmTSP",
            n_nodes=n_nodes,
            selected_ids=selected_ids,
            cost_function="Geom",
            dataset_ref=dataset_ref,
            n_salesmen=m,
            depots=depots,
            single_depot=False,
            demand=demand,
            constraint=constraint
        )
        
        # Verify feasibility with greedy solver
        test_problem.edges = await recreate_edges(test_problem)
        solver = NearestNeighbourMultiSolver4(problem_types=[test_problem])
        try:
            route = await asyncio.wait_for(solver.solve_problem(test_problem), timeout=10)
            if route is not None:
                solution_found = True
                test_problem.edges = None
                return test_problem
        except asyncio.TimeoutError:
            continue
    
    raise RuntimeError("Failed to generate feasible cmTSP after max attempts")

async def _generate_rcmdmtsptw(n_nodes: int, selected_ids: List[int], dataset_ref: str):
    """Generate a constrained mTSP with time windows."""
    # First generate base cmTSP
    base_problem = await _generate_rcmdmtsp(n_nodes, selected_ids, dataset_ref)
    
    # Solve it to get time windows
    base_problem.edges = await recreate_edges(base_problem)
    solver = NearestNeighbourMultiSolver4(problem_types=[base_problem])
    greedy_solution = await asyncio.wait_for(solver.solve_problem(base_problem), timeout=10)
    
    # Create time windows from greedy solution
    expected_improvement = 0.3
    travel_time_min = (base_problem.edges / 1000 / 50 * 60)
    travel_times = [0 for _ in range(n_nodes)]
    
    for route in greedy_solution:
        travel_time_elapsed = 0
        for idx, node in enumerate(route[:-1]):
            if idx == 0:
                travel_times[node] = 0
            else:
                travel_time_elapsed += travel_time_min[route[idx-1], node]
                travel_times[node] = travel_time_elapsed * (1 - expected_improvement)
    
    time_windows = [(round(float(t)*0.9, 5), round(float(t), 5)) for t in travel_times]
    
    return GraphV2ProblemMultiConstrainedTW(
        problem_type="Metric cmTSPTW",
        n_nodes=n_nodes,
        selected_ids=selected_ids,
        cost_function="Geom",
        dataset_ref=dataset_ref,
        n_salesmen=base_problem.n_salesmen,
        depots=base_problem.depots,
        single_depot=False,
        demand=base_problem.demand,
        constraint=base_problem.constraint,
        time_windows=time_windows
    )

@sn43.tool
async def recreate_edges(problem):
    """Recreate distance matrix from node coordinates."""
    node_coords_np = LOADED_DATASETS[problem.dataset_ref]["data"]
    node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
    if problem.cost_function == "Geom":
        return geom_edges(node_coords)
    elif problem.cost_function == "Euclidean2D":
        return euc_2d_edges(node_coords)
    elif problem.cost_function == "Manhatten2D":
        return man_2d_edges(node_coords)
    return None

@sn43.tool
async def score_solution(
    ctx: sn43.Context,
    solution: List[Any],
    problem_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Score a solution for the current problem.
    
    Args:
        ctx: Execution context
        solution: The solution to score (route or routes)
        problem_data: Optional problem data (uses ctx if not provided)
    
    Returns:
        Dictionary with score and validity information
    """
    # Get problem from context or parameter
    if problem_data is None:
        problem = ctx.get("current_problem")
    else:
        # Reconstruct problem from dict
        if "cmTSPTW" in problem_data.get("problem_type", ""):
            problem = GraphV2ProblemMultiConstrainedTW(**problem_data)
        elif "cmTSP" in problem_data.get("problem_type", ""):
            problem = GraphV2ProblemMultiConstrained(**problem_data)
        elif "mTSP" in problem_data.get("problem_type", ""):
            problem = GraphV2ProblemMulti(**problem_data)
        else:
            problem = GraphV2Problem(**problem_data)
    
    # Validate solution
    is_valid = is_valid_solution(problem, solution)
    if not is_valid:
        return {
            "valid": False,
            "score": float('inf'),
            "error": "Invalid solution"
        }
    
    # Recreate edges
    problem.edges = await recreate_edges(problem)
    
    # Calculate score based on problem type
    if isinstance(problem, GraphV2ProblemMultiConstrainedTW):
        score = get_multi_minmax_tour_distance_tw(problem, solution, problem.edges)
    elif isinstance(problem, (GraphV2ProblemMulti, GraphV2ProblemMultiConstrained)):
        score = get_multi_minmax_tour_distance(problem, solution)
    else:
        score = get_tour_distance(problem, solution)
    
    return {
        "valid": True,
        "score": float(score),
        "solution": solution
    }

@sn43.tool
async def get_baseline_score(ctx: sn43.Context) -> float:
    """
    Get the baseline (greedy) score for the current problem.
    Used for computing improvement thresholds.
    """
    problem = ctx.get("current_problem")
    problem.edges = await recreate_edges(problem)
    
    # Use appropriate greedy solver
    if isinstance(problem, (GraphV2ProblemMultiConstrained, GraphV2ProblemMultiConstrainedTW)):
        solver = NearestNeighbourMultiSolver4(problem_types=[problem])
    else:
        from sn43.graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
        solver = NearestNeighbourSolverVali()
    
    baseline_solution = await asyncio.wait_for(
        solver.solve_problem(problem), 
        timeout=30
    )
    
    
    if isinstance(problem, GraphV2ProblemMultiConstrainedTW):
        score = get_multi_minmax_tour_distance_tw(problem, baseline_solution, problem.edges)
    elif isinstance(problem, (GraphV2ProblemMulti, GraphV2ProblemMultiConstrained)):
        score = get_multi_minmax_tour_distance(problem, baseline_solution)
    else:
        score = get_tour_distance(problem, baseline_solution)
    
    ctx.set("baseline_score", float(score))
    return float(score)

# Initialize datasets when module loads
init_datasets()