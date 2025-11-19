"""
Example miner agent that solves TSP problems.
"""
import asyncio
import sn43
from typing import Dict, Any, List

class TSPMinerAgent(sn43.Agent):
    """Base agent class for TSP miners."""
    
    def __init__(self):
        super().__init__()
        print(f"TSP Miner Agent initialized", flush=True)
        self.solve_count = 0
    
    def init(self, ctx: sn43.Context) -> None:
        print(f"TSP Miner Agent init called with context", flush=True)
    
    @sn43.entrypoint
    async def solve_problem(
        self, 
        problem_data: Dict[str, Any],
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Solve a TSP problem.
        
        Args:
            ctx: Execution context
            problem_data: Problem specification from generate_problem
        
        Returns:
            Dictionary with solution and metadata
        """
        import sys

        # Example: Call greedy solver
        solution = await self._greedy_solve(problem_data)

        
        return {
            "solution": solution,
        }
    
    async def _greedy_solve(self, problem_data: Dict[str, Any]) -> List[Any]:
        """Implement solving logic."""
        import sys

        try:
            # LAZY IMPORT - only import when actually solving
            from sn43.graphite.utils.constants import BENCHMARK_SOLUTIONS, PROBLEM_TYPE

            n_nodes = problem_data["n_nodes"]
            problem_type = problem_data["problem_type"]

            problem_formulation = PROBLEM_TYPE.get(problem_type)
            greedy_solver_class = BENCHMARK_SOLUTIONS.get(problem_type)

            if greedy_solver_class:
                problem = problem_formulation(**problem_data)

                greedy_solver = greedy_solver_class([problem])

#                 problem.edges = await sn43.tools.recreate_edges(problem)

                solution = await asyncio.wait_for(
                    greedy_solver.solve_problem(problem),
                    timeout=30
                )

                return solution
            else:
                return list(range(n_nodes))

        except Exception as e:
            print(f"[MINER ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    
# Instantiate the agent
agent = TSPMinerAgent()