"""
Example miner agent that solves TSP problems.
Miners should implement their own solve_problem method.
"""
import asyncio
import sn43
from typing import Dict, Any, List
from graphite.utils.constants import BENCHMARK_SOLUTIONS, PROBLEM_TYPE

class TSPMinerAgent(sn43.Agent):
    """Base agent class for TSP miners."""
    
    def init(self, ctx: sn43.Context):
        """Initialize the agent."""
        print(f"TSP Miner Agent initialized")
        self.solve_count = 0
    
    @sn43.entrypoint
    async def solve_problem(
        self, 
        ctx: sn43.Context, 
        problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Solve a TSP problem.
        
        Args:
            ctx: Execution context
            problem_data: Problem specification from generate_problem
        
        Returns:
            Dictionary with solution and metadata
        """
        
        # Example: Call greedy solver (miners should implement their own)
        solution = await self._greedy_solve(problem_data)
        
        return {
            "solution": solution,
        }
    
    async def _greedy_solve(self, problem_data: Dict[str, Any]) -> List[Any]:
        """
        Implement your solving logic here.
        This is just a placeholder - miners should replace this.
        """
        # For demonstration, return a simple tour
        n_nodes = problem_data["n_nodes"]
        problem_type = problem_data["problem_type"]

        problem_formulation = PROBLEM_TYPE.get(problem_type)
        greedy_solver_class = BENCHMARK_SOLUTIONS.get(problem_type)
        if greedy_solver_class:
            # Use the benchmark greedy solver if available
            problem = problem_formulation(**problem_data)
            greedy_solver = greedy_solver_class([problem])

            # Recreate edges
            problem.edges = await sn43.tools.recreate_edges(problem)
            
            # Solve
            solution = await asyncio.wait_for(
                greedy_solver.solve_problem(problem),
                timeout=30
            )

            return solution
        else:
            # Default: return nodes in order
            return 0