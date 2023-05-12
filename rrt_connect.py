# This script use the RRT-Connect from OMPL
# Have to install OMPL by runing the script install-ompl-ubuntu.sh

from planning import *
import math
import sys

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl-1.5.2/py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og


class RRTConnect:
    def __init__(self, pl_env: Planning):
        space = ob.RealVectorStateSpace(pl_env.dof)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(pl_env.dof)
        bounds.setLow(-math.pi)
        bounds.setHigh(math.pi)
        space.setBounds(bounds)

        # create a simple setup object
        self.ss = og.SimpleSetup(space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

        self.pl_env = pl_env
        self.start = ob.State(space)
        self.goal = ob.State(space)
        self.obstacles = None

    def set_planning_request(self, pl_req):
        for i in range(self.pl_env.dof):
            self.start[i] = pl_req.start[i]
            self.goal[i] = pl_req.goal[i]

        self.obstacles = pl_req.obstacles

    def is_state_valid(self, state):
        is_valid = self.pl_env.manipulator.check_validity(state, self.obstacles)
        return is_valid

    def planning(self, pl_req: PlanningRequest, solve_time=5.0):
        self.set_planning_request(pl_req)
        si = self.ss.getSpaceInformation()
        planner = og.RRTConnect(si)
        self.ss.setPlanner(planner)
        self.ss.setStartAndGoalStates(self.start, self.goal)
        solved = self.ss.solve(solve_time)
        if solved:
            solution = self.get_solution()
            if np.linalg.norm(solution[-1] - pl_req.goal) < 1e-6:
                return True
        return False

    def get_solution(self, interpolate=False):
        solution_path = self.ss.getSolutionPath()
        if interpolate:
            solution_path.interpolate(30)
        ompl_solution = list(solution_path.getStates())
        solution = []
        for state in ompl_solution:
            np_state = np.zeros(self.pl_env.dof)
            for i in range(self.pl_env.dof):
                np_state[i] = state[i]
            solution.append(np_state)
        return solution

    def get_planning_time(self):
        return self.ss.getLastPlanComputationTime()
