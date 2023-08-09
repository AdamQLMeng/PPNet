import os
import time
import json
import math

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from ompl.geometric import SimpleSetup
from ompl import control as oc
from math import sqrt
from scipy.spatial import distance
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


def time_synchronized():
    return time.time()


class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius


def parse_obstacle(s):
    try:
        x, y, radius = map(float, s.split(','))
        return Obstacle(x, y, radius)
    except Exception:
        raise argparse.ArgumentTypeError("Obstacles must be x,y,radius")


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, obstacles):
        super(ValidityChecker, self).__init__(si)
        self.obstacles = obstacles

    # Returns whether the given state's position overlaps the
    # circular obstacle
    def isValid(self, state):
        x = state.getX()
        y = state.getY()
        for obstacle in self.obstacles:
            if sqrt(pow(x - obstacle.x, 2) + pow(y - obstacle.y, 2)) - obstacle.radius <= 0:
                return False

        return True

    # Returns the distance from the given state's position to the
    # boundary of the circular obstacle.
    def clearance(self, state):
        if len(self.obstacles) == 0:
            return 1

        # Extract the robot's (x,y) position from its state
        x = state.getX()
        y = state.getY()

        clearance = 0
        # Distance formula between two points, offset by the circle's
        # radius
        for obstacle in self.obstacles:

            clearance += (sqrt(pow(x - obstacle.x, 2) + pow(y - obstacle.y, 2)) - obstacle.radius)
            if clearance <= 0:
                return 0

        return clearance


def absolute_distance_between_angles(angle1, angle2):
    fabs = math.fabs(math.atan2(math.sin(angle1 - angle2), math.cos(angle1 - angle2)))
    return fabs


class MyDecomposition(oc.GridDecomposition):
    def __init__(self, length, bounds):
        super(MyDecomposition, self).__init__(length, 2, bounds)

    def project(self, s, coord):
        coord[0] = s.getX()
        coord[1] = s.getY()

    def sampleFullState(self, sampler, coord, s):
        sampler.sampleUniform(s)
        s.setXY(coord[0], coord[1])


def get_path_length_objective(si):
    return ob.PathLengthOptimizationObjective(si)


def get_threshold_path_length_objective(si):
    obj = ob.PathLengthOptimizationObjective(si)
    # obj.setCostThreshold(ob.Cost(8))
    return obj


class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    def stateCost(self, s):
        if (self.si_.getStateValidityChecker().clearance(s) == 0):
            return sys.maxsize
        return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s))**0.5)

class MinTurningObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(MinTurningObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to minimize turning. 
    def motionCost(self, s1, s2):
        return ob.Cost(math.fabs(s2.getYaw() - s1.getYaw()))

class WindObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(WindObjective, self).__init__(si, True)
        self.si_ = si

    # TODO: add requirement about wind alignment
    def motionCost(self, s1, s2):
        direction = math.atan2(s2.getY() - s1.getY(), s2.getX() - s1.getX())
        diff = absolute_distance_between_angles(direction, math.radians(80))
        return diff * ((s1.getX() - s2.getX())**2 + (s1.getY() - s2.getY())**2)


class TackingObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(TackingObjective, self).__init__(si, True)
        self.si_ = si

    # TODO: Cost based on tacking paper
    def motionCost(self, s1, s2):
        direction = math.atan2(s2.getY() - s1.getY(), s2.getX() - s1.getX())
        distance = ((s2.getY() - s1.getY())**2 + (s2.getX() - s1.getX())**2)**0.5
        wind_direction = math.radians(225)
        relative_wind_direction = wind_direction - direction

        upwind_angle = math.radians(45)
        downwind_angle = math.radians(30)

        if math.fabs(relative_wind_direction) < upwind_angle:
            multiplier = 2.0
        elif math.fabs(relative_wind_direction - math.radians(180)) < downwind_angle:
            multiplier = 1.5
        else:
            multiplier = 0.0

        return multiplier * distance

def get_clearance_objective(si):
    return ClearanceObjective(si)


def get_path_length_obj_with_cost_to_go(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "abitstar":
        return og.ABITstar(si)
    elif plannerType.lower() == "aitstar":
        return og.AITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

# Keep these in alphabetical order and all lower case
def allocate_planner(si, planner_type, decomp):
    if planner_type.lower() == "est":
        return og.EST(si)
    elif planner_type.lower() == "kpiece":
        return og.KPIECE1(si)
    elif planner_type.lower() == "pdst":
        return og.PDST(si)
    elif planner_type.lower() == "rrt":
        return og.RRT(si)
    elif planner_type.lower() == "syclopest":
        return og.SyclopEST(si, decomp)
    elif planner_type.lower() == "sycloprrt":
        return og.SyclopRRT(si, decomp)

    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def getBalancedObjective(si):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)
    minTurnObj = MinTurningObjective(si)
    windObj = WindObjective(si)
    tackingObj = TackingObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 1.0)
    opt.addObjective(clearObj, 80.0)
    opt.addObjective(minTurnObj, 0.0)
    opt.addObjective(windObj, 0.0)
    opt.addObjective(tackingObj, 0.0)
    # opt.setCostThreshold(ob.Cost(5))

    return opt

# Keep these in alphabetical order and all lower case
def allocate_objective(si, objectiveType):
    if objectiveType.lower() == "pathclearance":
        return get_clearance_objective(si)
    elif objectiveType.lower() == "pathlength":
        return get_path_length_objective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return get_threshold_path_length_objective(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")


def create_numpy_path(states):
    lines = states.splitlines()
    length = len(lines) - 1
    array = np.zeros((length, 2))

    for i in range(length):
        array[i][0] = float(lines[i].split(" ")[0])
        array[i][1] = float(lines[i].split(" ")[1])
    return array


class TerminationCondition:
    def __init__(self, ss: SimpleSetup = None, target_path_length=60, epsilon=0.1):
        assert ss is not None, 'ss is required parameter'
        assert target_path_length != 0, "target path length couldn't be 0, it will lead to failure of convergence"
        self.ss = ss
        self.target_path_length = target_path_length
        self.epsilon = epsilon
        self.length_fn = None

    def __call__(self):
        if self.ss.getPlanner().getName().lower() == "prmstar":
            self.length_fn = self.ss.getPlanner().getBestCost
        else:
            self.length_fn = self.ss.getPlanner().bestCost().value
        length = float(self.length_fn())
        if length <= (1 + self.epsilon) * self.target_path_length:
            return True
        return False


def plan(run_time, planner_type, objective_type, map_size, init, end, obstacles, clearance, target_path_length, epsilon, index):
    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^2.

    space = ob.SE2StateSpace()

    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, 0)
    bounds.setLow(1, 0)

    bounds.setHigh(0, map_size[0])
    bounds.setHigh(1, map_size[1])

    # Set the bounds of space to be in [0,1].
    space.setBounds(bounds)

    # define a simple setup class
    ss = og.SimpleSetup(space)

    # Construct a space information instance for this state space
    si = ss.getSpaceInformation()
    pdef = ob.ProblemDefinition(si)
    # Set resolution of state validity checking. This is fraction of space's extent.
    # si.setStateValidityCheckingResolution(0.001)

    # Set the object used to check which states in the space are valid
    validity_checker = ValidityChecker(si, obstacles)
    ss.setStateValidityChecker(validity_checker)

    # Set our robot's starting state to be the bottom-left corner of
    # the environment, or (0,0).
    start = ob.State(space)
    start[0] = init[0]
    start[1] = init[1]
    start[2] = math.pi / 4
    # Set our robot's goal state to be the top-right corner of the
    # environment, or (1,1).
    goal = ob.State(space)
    goal[0] = end[0]
    goal[1] = end[1]
    goal[2] = math.pi / 4

    # Set the start and goal states
    ss.setStartAndGoalStates(start, goal)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    objective = allocate_objective(si, objective_type)
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)
    minTurnObj = MinTurningObjective(si)
    windObj = WindObjective(si)
    tackingObj = TackingObjective(si)
    ss.setOptimizationObjective(objective)

    print("ss.getProblemDefinition().hasOptimizationObjective( ){}".format(ss.getProblemDefinition().hasOptimizationObjective()))
    print("ss.getProblemDefinition().hasOptimizedSolution() {}".format(ss.getProblemDefinition().hasOptimizedSolution()))

    # Create a decomposition of the state space
    decomp = MyDecomposition(256, bounds)

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    # optimizing_planner = allocate_planner(si, planner_type, decomp)
    optimizing_planner = allocatePlanner(si, planner_type)

    ss.setPlanner(optimizing_planner)

    # set planner termination condition
    ptc_fn = ob.PlannerTerminationConditionFn(
                    TerminationCondition(ss, target_path_length=target_path_length, epsilon=epsilon)
    )
    ptc1 = ob.PlannerTerminationCondition(ptc_fn)
    ptc2 = ob.timedPlannerTerminationCondition(run_time)
    ptc = ob.plannerOrTerminationCondition(ptc1, ptc2)

    # attempt to solve the planning problem in the given runtime
    print(ss.getPlanner().getName().lower())
    t_start = time_synchronized()
    solved = ss.solve(ptc if ss.getPlanner().getName().lower() not in ["bfmt", "fmt"] else run_time)
    t_end = time_synchronized()
    print(t_end-t_start)
    if solved:
        # Output the length of the path found
        print('{0} found solution of path length {1:.4f} with an optimization '
              'objective value of {2:.4f}'.format(ss.getPlanner().getName(),
                                                  ss.getSolutionPath().length(),
                                                  0.1))
        solution_path = ss.getSolutionPath()
        matrix = solution_path.printAsMatrix()
        path = create_numpy_path(matrix)
        length = solution_path.length()
        states = solution_path.getStates()
        prevState = states[0]
        for state in states[1:]:
            space.validSegmentCount(prevState, state)
            prevState = state

        print("ss.getSolutionPath().printAsMatrix() = {}".format(ss.getSolutionPath().printAsMatrix()))
        print("ss.haveSolutionPath() = {}".format(ss.haveSolutionPath()))
        print("ss.haveExactSolutionPath() = {}".format(ss.haveExactSolutionPath()))
        print("***")
        print("ss.getSolutionPath().length() = {}".format(ss.getSolutionPath().length()))
        print("ss.getSolutionPath().check() = {}".format(ss.getSolutionPath().check()))
        print("ss.getSolutionPath().clearance() = {}".format(ss.getSolutionPath().clearance()))
        print("ss.getSolutionPath().cost(objective).value() = {}".format(ss.getSolutionPath().cost(objective).value()))
        print("ss.getSolutionPath().cost(lengthObj).value() = {}".format(ss.getSolutionPath().cost(lengthObj).value()))
        print("ss.getSolutionPath().cost(clearObj).value() = {}".format(ss.getSolutionPath().cost(clearObj).value()))
        print("ss.getSolutionPath().cost(minTurnObj).value() = {}".format(ss.getSolutionPath().cost(minTurnObj).value()))
        print("ss.getSolutionPath().cost(windObj ).value() = {}".format(ss.getSolutionPath().cost(windObj).value()))
        print("ss.getSolutionPath().cost(tackingObj ).value() = {}".format(ss.getSolutionPath().cost(tackingObj).value()))
        print("ss.getProblemDefinition().hasOptimizedSolution() {}".format(ss.getProblemDefinition().hasOptimizedSolution()))
        print("***")
        return path, length, t_end-t_start
    else:
        print("No solution found.")
        return None, None, None


def plot_path(path, size, start, goal, clearance, obstacles, file_name=None):
    x, y = path.T
    ax = plt.gca()
    ax.axis(xmin=0, xmax=size[0], ymin=size[1], ymax=0)
    ax.plot(x, y, 'b--')
    # ax.plot(x, y, '')
    plot_map(size, start, goal, clearance, obstacles, None)
    if file_name:
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.savefig('./{}'.format(file_name), dpi=1000)
        plt.clf()


def plot_map(size, init, end, clearance, obstacles, file_name=None):
    ax = plt.gca()
    ax.axis(xmin=0, xmax=size[0], ymin=size[1], ymax=0)
    for obstacle in obstacles:
        # plt.plot(obstacle.x, obstacle.y, 'yo')
        ax.add_patch(plt.Circle(xy=(obstacle.x, obstacle.y), radius=obstacle.radius-clearance, fc='black', ec='black'))
    ax.plot(init[0], init[1], 'rs')
    ax.plot(end[0], end[1], 'rs')
    # if file_name:
    #     plt.savefig(r'./temp.jpg', dpi=90)
    #     img = Image.open(r'./temp.jpg').convert('RGB')
    #     img = T.ToTensor()(img)#T.CenterCrop((224, 224))(T.ToTensor()(img))
    #     img = img[:, 53:383, 73:517]
    #     img = T.Resize([224, 224])(img)
    #     init = init/map_size*224
    #     end = end/map_size*224
    #     img = add_init_end_single(img, [224-init[1], init[0]], [224-end[1], end[0]])
    #     torchvision.utils.save_image(img, file_name)
        # T.ToPILImage()(img).save('./{}'.format(file_name))


def random_state_space(init, end, num_obs, obs_size, map_size, clearance):
    obs = np.concatenate([(np.random.random(num_obs)*map_size[0]).reshape([-1, 1]), (np.random.random(num_obs)*map_size[1]).reshape([-1, 1]), (np.random.random(num_obs)*obs_size).reshape([-1, 1])], axis=1)
    print(obs.shape)
    obstacles = []
    print('Obstacles data:')
    for item in obs:
        coord = item[:2]
        size = item[-1].item()
        print(coord, size)
        if distance.euclidean(coord, init) > math.pi/4 + size + clearance and distance.euclidean(coord, end) > math.pi/4 + size + clearance:
            obstacles.append(list(item))
    print('number of obstacles', len(obstacles))
    return obstacles


def add_init_end_single(image, init, end):
    red = torch.tensor([255, 0, 0])
    assert len(image.shape) == 3, "Image shape incorrect"
    assert init is not None, "Init is None"
    assert end is not None, "End is None"

    # image = np.array(image).copy().reshape([image.shape[1], image.shape[2], image.shape[0]])
    resolution = image.shape[1]
    for j in range(4):
        for k in range(4):
            if 0 <= int(np.round(init[0]) + j) < resolution and 0 <= int(np.round(init[1]) + k) < resolution:
                image[:, int(np.round(init[0]) + j), int(np.round(init[1]) + k)] = red
            if 0 <= int(np.round(init[0]) + j) < resolution and 0 <= int(np.round(init[1]) - k) < resolution:
                image[:, int(np.round(init[0]) + j), int(np.round(init[1]) - k)] = red
            if 0 <= int(np.round(init[0]) - j) < resolution and 0 <= int(np.round(init[1]) + k) < resolution:
                image[:, int(np.round(init[0]) - j), int(np.round(init[1]) + k)] = red
            if 0 <= int(np.round(init[0]) - j) < resolution and 0 <= int(np.round(init[1]) - k) < resolution:
                image[:, int(np.round(init[0]) - j), int(np.round(init[1]) - k)] = red
            if 0 <= int(np.round(end[0]) + j) < resolution and 0 <= int(np.round(end[1]) + k) < resolution:
                image[:, int(np.round(end[0]) + j), int(np.round(end[1]) + k)] = red
            if 0 <= int(np.round(end[0]) + j) < resolution and 0 <= int(np.round(end[1]) - k) < resolution:
                image[:, int(np.round(end[0]) + j), int(np.round(end[1]) - k)] = red
            if 0 <= int(np.round(end[0]) - j) < resolution and 0 <= int(np.round(end[1]) + k) < resolution:
                image[:, int(np.round(end[0]) - j), int(np.round(end[1]) + k)] = red
            if 0 <= int(np.round(end[0]) - j) < resolution and 0 <= int(np.round(end[1]) - k) < resolution:
                image[:, int(np.round(end[0]) - j), int(np.round(end[1]) - k)] = red
    return image


def generate_random_problems(num_prob, num_obs, obs_size, map_size, clearance, root="./data"):
    unsolved_problems_txt = "{}/unsolved_problems.txt".format(root)
    for i in range(num_prob):
        while True:
            init_state = np.random.random(2) * map_size
            end_state = np.random.random(2) * map_size
            if distance.euclidean(init_state, end_state) > math.sqrt(map_size[0] ** 2 + map_size[1] ** 2) / 2:
                break
        obstacles = random_state_space(init=init_state,
                                                 end=end_state,
                                                 num_obs=num_obs,
                                                 obs_size=obs_size,
                                                 map_size=map_size,
                                                 clearance=clearance)
        problem = {"Init": list(init_state), "End": list(end_state), "Obstacles": obstacles}
        obstacles = [parse_obstacle(str(o[0]) + ',' + str(o[1]) + ',' + str(o[2])) for o in problem["Obstacles"]]
        plot_map(size=map_size, init=init_state, end=end_state, obstacles=obstacles, clearance=clearance, file_name='{}/{}.jpg'.format(root, i))
        plt.clf()
        with open(unsolved_problems_txt, "a") as f:
            f.write(json.dumps(problem) + "\n")


def find_solution_planners(planners, solved_problems_txt, p_list, runtime, map_size, clearance, epsilon, root="./data"):
    solved_problems_comparison_txt = "{}/solved_problems_comparison.txt".format(root)
    problem_compared = []
    with open(solved_problems_txt, 'r', encoding='utf-8') as f:
        content_unsolved = f.readlines()
    print("{} problems are waiting for solving!".format(len(content_unsolved)))
    if os.path.exists(solved_problems_comparison_txt):
        with open(solved_problems_comparison_txt, 'r', encoding='utf-8') as f:
            content_solved = f.readlines()
        for i, line in enumerate(content_solved):
            problem_compared.append(json.loads(line)["Index"])
    print("{} problems has been compared".format(len(problem_compared)))
    for i, line in enumerate(content_unsolved):
        problem = json.loads(line)
        # if problem["Index"] not in p_list:
        #     continue
        if problem["Index"] in problem_compared:
            print(i, 'Problem(Index:{}) has been compared!'.format(int(problem["Index"])))
            continue
        print("Going to compare problem N0.{}".format(problem["Index"]))
        init_state = [problem["Init"][1], problem["Init"][0]]
        if init_state[0] <= 0:
            init_state[0] = 0.1
        if init_state[0] >= 224:
            init_state[0] = 223.9
        if init_state[1] <= 0:
            init_state[1] = 0.1
        if init_state[1] >= 224:
            init_state[1] = 223.9
        end_state = [problem["End"][1], problem["End"][0]]
        if end_state[0] <= 0:
            end_state[0] = 0.1
        if end_state[0] >= 224:
            end_state[0] = 223.9
        if end_state[1] <= 0:
            end_state[1] = 0.1
        if end_state[1] >= 224:
            end_state[1] = 223.9
        target_path_length = problem["Length"]
        # target_path = problem["Waypoint"]
        # target_path_length = sum([distance.euclidean(target_path[i], target_path[i+1]) for i in range(len(target_path)-1)])
        print('init end', init_state, end_state, target_path_length, problem["Length"])
        invalid = False
        for obs in problem["Obstacles"]:
            if distance.euclidean(obs[:2], init_state) - obs[-1] - clearance/2 < 0 \
                    or distance.euclidean(obs[:2], end_state) - obs[-1] - clearance/2 < 0:
                invalid = True
                break
        if invalid:
            print('Problem(Index:{}) is invalid!'.format(int(problem["Index"])))
            continue
        obstacles = [parse_obstacle(str(o[0]) + ',' + str(o[1]) + ',' + str(o[2]+clearance/2)) for o in problem["Obstacles"]]
        # plot_map((224, 224), init_state, end_state, clearance, obstacles)
        # plt.plot(np.array(target_path).T[1], np.array(target_path).T[0])
        # plt.show()
        Solution = []
        for planner in planners:
            # Solve the planning problem
            path, length, planning_time = plan(runtime, planner, 'PathLength',
                                               map_size, init_state, end_state, obstacles,
                                               clearance, target_path_length, epsilon, i)
            # plot_path(path, map_size, init_state, end_state, clearance, obstacles, file_name=None)  # '{}/{}_{}.jpg'.format(root, planner, i))
            # plt.plot(np.array(target_path).T[1], np.array(target_path).T[0])
            # plt.show()
            if path is not None:
                path = [list(p) for p in list(path)]
            Solution.append({"Planner": planner, "Waypoint": path, "Length": length, "Time": planning_time})
        problem["Solution"] = Solution
        with open(solved_problems_comparison_txt, "a") as f:
            f.write(json.dumps(problem) + "\n")


if __name__ == "__main__":
    map_size = [224, 224]
    # init_state = [5, 5]
    # end_state = [40, 40]
    clearance = 1
    # target_path_length = 55
    epsilon = 0.00  # 5% range of target path length
    num_obs = 20
    num_prob = 1000
    runtime = 30
    planners = ['BITstar', 'ABITstar', 'InformedRRTstar', 'RRTstar']  # , 'AITstar', 'InformedRRTstar',  'PRMstar']
    info = 2

    # Check that time is positive
    if runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
            % (runtime,))

    # Set the log level
    if info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")

    # set random obstacles
    result_root = "/home/long/experiment/result_c{}_{}".format(clearance, num_obs)
    result_root = "/home/long/dataset/exp_diff_data_generation"
    result_root = "/home/long/dataset/exp_diff_algo"
    # result_root = "/home/long/dataset/exp_diff_algo_minseg4"
    p_list = []
    # supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # p_list = [int(i.split('.')[0].split('_')[-1]) for i in os.listdir(result_root+"/result_path_c1")
    #           if os.path.splitext(i)[-1] in supported]
    # print(len(p_list))
    # p_list.sort()
    # generate_random_problems(num_prob, num_obs, obs_size, map_size, clearance, root=data_root)
    solved_problems_txt = "{}/solved_problems.txt".format(result_root)
    find_solution_planners(planners, solved_problems_txt, p_list, runtime, map_size, clearance/50*224, epsilon, result_root)

    # for _ in range(num_prob):

