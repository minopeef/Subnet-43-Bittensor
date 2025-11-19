# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2023 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from sn43.graphite.utils.graph_utils import (
    get_tour_distance,
    get_multi_minmax_tour_distance,
    get_multi_minmax_tour_distance_tw,
)
from sn43.graphite.protocol import (
    GraphV2Problem,
    GraphV2ProblemMulti,
    GraphV2ProblemMultiConstrained,
    GraphV2ProblemMultiConstrainedTW,
)

# ---- LAZY IMPORT HELPERS TO AVOID CIRCULAR IMPORT ----

def _nn_solver():
    from sn43.graphite.solvers.greedy_solver import NearestNeighbourSolver
    return NearestNeighbourSolver

def _nn_multi_2():
    from sn43.graphite.solvers.greedy_solver_multi_2 import NearestNeighbourMultiSolver2
    return NearestNeighbourMultiSolver2

def _nn_multi_4():
    from sn43.graphite.solvers.greedy_solver_multi_4 import NearestNeighbourMultiSolver4
    return NearestNeighbourMultiSolver4


# Wrappers so existing code using classes still works
class LazySolver:
    def __init__(self, loader):
        self._loader = loader
        self._cls = None

    def __call__(self, *args, **kwargs):
        if self._cls is None:
            self._cls = self._loader()
        return self._cls(*args, **kwargs)

    @property
    def cls(self):
        if self._cls is None:
            self._cls = self._loader()
        return self._cls


# Lazy solver objects (drop-in replacements for class references)
NN = LazySolver(_nn_solver)
NN2 = LazySolver(_nn_multi_2)
NN4 = LazySolver(_nn_multi_4)


# ---- BENCHMARK MAPPINGS ----

BENCHMARK_SOLUTIONS = {
    'Metric TSP': NN,
    'General TSP': NN,
    'Metric mTSP': NN2,
    'General mTSP': NN2,
    'Metric cmTSP': NN4,
    'General cmTSP': NN4,
    'Metric cmTSPTW': NN4,
    'General cmTSPTW': NN4,
}

PROBLEM_TYPE = {
    'Metric TSP': GraphV2Problem,
    'General TSP': GraphV2Problem,
    'Metric mTSP': GraphV2ProblemMulti,
    'General mTSP': GraphV2ProblemMulti,
    'Metric cmTSP': GraphV2ProblemMultiConstrained,
    'General cmTSP': GraphV2ProblemMultiConstrained,
    'Metric cmTSPTW': GraphV2ProblemMultiConstrainedTW,
    'General cmTSPTW': GraphV2ProblemMultiConstrainedTW,
}

COST_FUNCTIONS = {
    'Metric TSP': get_tour_distance,
    'General TSP': get_tour_distance,
    'Metric mTSP': get_multi_minmax_tour_distance,
    'General mTSP': get_multi_minmax_tour_distance,
    'Metric cmTSP': get_multi_minmax_tour_distance,
    'General cmTSP': get_multi_minmax_tour_distance,
    'Metric cmTSPTW': get_multi_minmax_tour_distance_tw,
    'General cmTSPTW': get_multi_minmax_tour_distance_tw,
}