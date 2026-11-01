"""
mvmd -- Multi-Vehicle Multi-Drone delivery with independent and truck-supported
drone operations (temporary truck-waiting synchronisation).

Companion code of the manuscript
"Sequence-Aware Adaptive Large Neighborhood Search for Multi-Vehicle Multi-Drone
Delivery with Independent and Truck-Supported Drone Operations".

Modules
-------
instance   : instance definition, generators (toy network / Solomon C101), clustering, JSON I/O
solution   : solution representation and exact evaluation (objective = MILP objective)
construct  : TWCC-GRH two-stage constructive heuristic (Section 5.1)
operators  : destroy / repair operators (Shaw removal, cost-based removal, greedy, regret-2)
alns       : ALNS-I (independent weights), ALNS-P (pair weights, = OSD with Len=1),
             OSD-ALNS (sequence weights, Len>=2)                       (Section 5.2)
vns, ga    : baseline metaheuristics                                   (Section 6.2)
milp       : arc-based MILP (Section 4) with optional valid inequalities (Section 4.5),
             HiGHS or Gurobi backend
"""
__version__ = "1.0.0"
