# This file include the context of motion planning using a graph
# To initiate the planning context, you have to first define a manipulator
# Run this script to see a planning request and a occupancy map
# If there is a solution in the given roadmap, this solution will also be visualized
import matplotlib.pyplot as plt

from manipulator import *
from utilities import *
import networkx as nx
from scipy.stats import qmc
import numpy as np

from heapq import heappush, heappop


class Planning:
    def __init__(self, manipulator: manipulator,
                 planning_range_max=None, planning_range_min=None, resolution=0.01) -> None:
        self.manipulator = manipulator
        self.planning_range_max = planning_range_max
        self.planning_range_min = planning_range_min
        self.resolution = resolution
        self.G = None
        self.original_G = None
        self.sampler = qmc.Halton(d=manipulator.dof, seed=37)
        self.static_obstacles = None
        self.idx_path = None
        self.dof = manipulator.dof

    def get_graph_from_file(self, graph_file: str):
        self.G = nx.read_graphml(graph_file)

    def generate_graph_halton(self, n_nodes, obstacles=None):
        self.G = nx.Graph()
        if obstacles is not None:
            self.static_obstacles = obstacles
        for i in range(n_nodes):
            valid_sample = self.generate_valid_sample(obstacles)
            self.add_node(valid_sample)

    def generate_valid_sample(self, obstacles):
        valid_sample = False
        while valid_sample is False:
            sample = self.sampler.random().squeeze()
            sample = sample * self.manipulator.jnt_ranges + self.manipulator.jnt_lower_bound
            valid_sample = self.manipulator.check_validity(jnt=sample, obstacles=obstacles)
        return sample

    def save_graph_to_file(self, graph_file: str):
        g = make_numpy_graph_to_graphml_ok(self.G, "state")
        # Save graph
        nx.write_graphml(g, graph_file)

    def check_validity(self, source_config, target_config, obstacles):
        # source_config = edge["source"]
        # target_config = edge["target"]

        target_valid = self.manipulator.check_validity(target_config, obstacles)
        if not target_valid:
            return False, 1

        diff = np.linalg.norm(target_config - source_config)
        if diff < 1e-6:
            return True, 0

        n_checking = int(diff // self.resolution)

        step = (target_config - source_config) * self.resolution / diff
        for i in range(n_checking):
            config = source_config + i * step
            valid = self.manipulator.check_validity(config, obstacles)
            if not valid:
                return False, i+2
        return True, n_checking+1

    def check_path_validity(self, path: list, obstacles):
        source = path[0]
        ppath = path[1:]
        for node in ppath:
            is_valid, _ = self.check_validity(source, node, obstacles)
            if not is_valid:
                return False
            source = node
        return True

    def add_node(self, node_state, idx=None, max_nn_number=20):
        max_nn_dist = np.pi

        if idx is None:
            idx = self.G.number_of_nodes()

        def is_nn(nn):
            close_enough = np.linalg.norm(self.G.nodes[nn]["state"] - node_state) < max_nn_dist
            if not close_enough:
                return False

            if self.static_obstacles is not None:
                valid_edge, _ = self.check_validity(self.G.nodes[nn]["state"], node_state, self.static_obstacles)

                return valid_edge
            return True

        temp_nns = sorted(self.G.nodes,
                          key=lambda x: np.linalg.norm(self.G.nodes[x]["state"] - node_state))

        nns = []
        cnt_nns = 0
        for nn in temp_nns:
            if is_nn(nn):
                nns.append(nn)
                cnt_nns += 1
            if cnt_nns > max_nn_dist:
                break

        if len(nns) > max_nn_number:
            nns = nns[:max_nn_number]

        self.G.add_nodes_from([(idx, {"state": node_state})])
        for nn in nns:
            self.G.add_edge(nn, idx)

    def remove_node(self, node_state, idx=None):
        if idx is None:
            idx = self.G.number_of_nodes() - 1
        node = self.G.nodes[idx]["state"]

        if np.linalg.norm(node - node_state) < 1e-6:
            # Ensure the node is the same
            self.G.remove_node(idx)

    def node_dist(self, node1, node2):
        node1_state = self.G.nodes[node1]["state"]
        node2_state = self.G.nodes[node2]["state"]
        return np.linalg.norm(node1_state - node2_state)

    def search(self, planning_request: PlanningRequest,
               idx_start=None, idx_goal=None, visualize=False) -> PlanningResult:
        """
        :param planning_request:
        :param visualize:
        :param idx_start:
        :param idx_goal:
        :param parallel:
        :return: (has_path, state_path, checking_cnt, checked_list )
        """

        start = planning_request.start
        goal = planning_request.goal
        obstacles = planning_request.obstacles
        # TODO(Xi): implement search
        num_of_nodes = self.G.number_of_nodes()
        idx_start = num_of_nodes if idx_start is None else idx_start
        idx_goal = num_of_nodes + 1 if idx_goal is None else idx_goal

        self.add_node(start, idx_start, max_nn_number=30)
        self.add_node(goal, idx_goal, max_nn_number=30)

        pr = PlanningResult()
        # Define functions for weights and heuristics
        def get_heuristics(source_node, target_node):
            return self.node_dist(source_node, target_node)

        if visualize: 
            idx_path, cnt, checked_dict = self.a_star_with_visualization(
                idx_start, idx_goal, heuristic=get_heuristics, obstacles=obstacles)
        else:
            idx_path, cnt, checked_dict = self.a_star(
                idx_start, idx_goal, heuristic=get_heuristics, obstacles=obstacles)
        
        # nx.astar_path(self.G, idx_start, idx_goal, heuristic=get_heuristics, weight=get_weight)
        if idx_path is not None:
            state_path = [self.G.nodes[idx]["state"] for idx in idx_path]
            self.G.remove_node(idx_start)
            self.G.remove_node(idx_goal)
            self.idx_path = idx_path

            pr.has_solution = True
            pr.solution_path = state_path
            pr.index_path = idx_path
            pr.checked_counts = cnt
            pr.checked_dict = checked_dict
            return pr

        self.G.remove_node(idx_start)
        self.G.remove_node(idx_goal)
        self.idx_path = None

        pr.has_solution = False
        pr.solution_path = []
        pr.index_path = []
        pr.checked_counts = cnt
        pr.checked_dict = checked_dict
        return pr

    def rewire_graph(self):
        n_nodes = self.G.number_of_nodes()
        for node in range(n_nodes):
            node_state = self.G.nodes[node]["state"]
            self.G.remove_node(node)
            self.add_node(node_state, node)

    def a_star(self, source, target, heuristic, obstacles):
        """
        Retrun: path, collision checking counts, a dictionary of checked nodes
        """
        if source not in self.G or target not in self.G:
            msg = f"Either source {source} or target {target} is not in G"
            raise nx.NodeNotFound(msg)

        if heuristic is None:
            # The default heuristic is h=0 - same as Dijkstra's algorithm
            def heuristic(u, v):
                return 0

        push = heappush
        pop = heappop
        # weight = _weight_function(G, weight)

        # The queue stores priority, node, cost to reach, and parent.
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        queue = []
        # f-value, count, node, cost, parent
        for neighbor, _ in self.G[source].items():
            h = heuristic(neighbor, target)
            dist = self.node_dist(neighbor, source)
            push(queue, (dist + h, (source, neighbor), dist, source))

        # Maps enqueued nodes to distance of discovered paths and the
        # computed heuristics to target. We avoid computing the heuristics
        # more than once and inserting the node into the queue too many times.
        enqueued = {}
        # Maps explored nodes to parent closest to the source.
        explored = {source: None}
        checked_dict = {}
        checked_cnt = 0

        while queue:
            # Pop the smallest item from queue.
            _, curr_edge, dist, parent = pop(queue)
            source_node, curr_node = curr_edge

            source_state = self.G.nodes[source_node]["state"]
            curr_state = self.G.nodes[curr_node]["state"]

            if curr_node in explored:
                # Do not override the parent of starting node
                # if explored[curr_node] is None:
                continue

                # Skip bad paths that were enqueued before finding a better one
                # qcost, h = enqueued[curr_node]
                # if qcost < dist:
                #     continue

            is_edge_valid, cnt = self.check_validity(source_state, curr_state, obstacles)
            checked_cnt += cnt
            if curr_node not in checked_dict and curr_node != target:
                checked_dict.update({curr_node: is_edge_valid})
            if is_edge_valid:
                if curr_node == target:
                    path = [curr_node]
                    node = parent
                    while node is not None:
                        path.append(node)
                        node = explored[node]
                    path.reverse()
                    return path, checked_cnt, checked_dict

                explored[curr_node] = parent
                for neighbor, _ in self.G[curr_node].items():
                    ncost = dist + self.node_dist(neighbor, curr_node)
                    # if neighbor in enqueued:
                    #     qcost, h = enqueued[neighbor]
                    #     # if qcost <= ncost, a less costly path from the
                    #     # neighbor to the source was already determined.
                    #     # Therefore, we won't attempt to push this neighbor
                    #     # to the queue
                    #     if qcost <= ncost:
                    #         continue
                    if (curr_node, neighbor) in enqueued \
                            or (neighbor, curr_node) in enqueued:
                        continue

                    if neighbor in explored:
                        continue

                    h = heuristic(neighbor, target)
                    enqueued[(curr_node, neighbor)] = ncost, h
                    push(queue, (ncost + h, (curr_node, neighbor), ncost, curr_node))
        
        return None, checked_cnt, checked_dict
    
    def a_star_with_visualization(self, source, target, heuristic, obstacles):
        if source not in self.G or target not in self.G:
            msg = f"Either source {source} or target {target} is not in G"
            raise nx.NodeNotFound(msg)

        if heuristic is None:
            # The default heuristic is h=0 - same as Dijkstra's algorithm
            def heuristic(u, v):
                return 0

        push = heappush
        pop = heappop

        queue = []
        # f-value, count, node, cost, parent
        for neighbor, _ in self.G[source].items():
            h = heuristic(neighbor, target)
            dist = self.node_dist(neighbor, source)
            push(queue, (dist + h, (source, neighbor), dist, source))

        enqueued = {}
        # Maps explored nodes to parent closest to the source.
        explored = {source: None}
        checked_list = []
        checked_cnt = 0

        # Visualize the graph and planning request before search 
        fig, axs = plt.subplots(
            nrows=1, ncols=1, sharex=False, sharey=False, figsize=(20, 14))
        edge_colors = ["gray"] * self.G.number_of_edges()
        edge_width = [1.0] * self.G.number_of_edges()
        state = nx.get_node_attributes(self.G, "state")
        
        # nx.draw_networkx_edges(
        #         self.G, state, alpha=0.4, ax=axs, edge_color=edge_colors)
        # nx.draw_networkx_nodes(
        #     self.G,
        #     state,
        #     nodelist=self.G.nodes,
        #     node_size=60,
        #     cmap=plt.cm.Reds_r,
        #     ax=axs
        # )
        # plt.show()

        edge_list = list(self.G.edges())
        while queue:
            # Pop the smallest item from queue.
            _, curr_edge, dist, parent = pop(queue)
            source_node, curr_node = curr_edge
            try:
                edge_idx = edge_list.index(curr_edge)
            except ValueError:
                edge_idx = edge_list.index((curr_node, source_node))

            source_state = self.G.nodes[source_node]["state"]
            curr_state = self.G.nodes[curr_node]["state"]

            if curr_node in explored:
                # Do not override the parent of starting node
                # if explored[curr_node] is None:
                continue

                # Skip bad paths that were enqueued before finding a better one
                # qcost, h = enqueued[curr_node]
                # if qcost < dist:
                #     continue

            if curr_node not in checked_list and curr_node != target:
                checked_list.append(curr_node)

            is_edge_valid, cnt = self.check_validity(source_state, curr_state, obstacles)
            checked_cnt += cnt
            if is_edge_valid:
                edge_colors[edge_idx] = "green"
                edge_width[edge_idx] = 2.0
                if curr_node == target:
                    node_color = ["gray"] * self.G.number_of_nodes()
                    node_color[-1] = "green"
                    node_color[-2] = "red"
                    nx.draw_networkx_nodes(
                                    self.G,
                                    state,
                                    nodelist=self.G.nodes,
                                    node_size=60,
                                    cmap=plt.cm.Reds_r,
                                    node_color=node_color,
                                    ax=axs
                                )
                    nx.draw_networkx_edges(
                        self.G, state, alpha=0.4, ax=axs, edge_color=edge_colors)
                    sum_links_length = 3.5
                    axs.set_xlim([-sum_links_length, sum_links_length])
                    axs.set_ylim([-sum_links_length, sum_links_length])
                    axs.set_aspect('equal', adjustable='box')
                    # plt.show() 
                    path = [curr_node]
                    node = parent
                    while node is not None:
                        path.append(node)
                        node = explored[node]
                    path.reverse()
                    return path, checked_cnt, checked_list

                explored[curr_node] = parent
                for neighbor, _ in self.G[curr_node].items():
                    ncost = dist + self.node_dist(neighbor, curr_node)
                    # if neighbor in enqueued:
                    #     qcost, h = enqueued[neighbor]
                    #     # if qcost <= ncost, a less costly path from the
                    #     # neighbor to the source was already determined.
                    #     # Therefore, we won't attempt to push this neighbor
                    #     # to the queue
                    #     if qcost <= ncost:
                    #         continue
                    if (curr_node, neighbor) in enqueued \
                            or (neighbor, curr_node) in enqueued:
                        continue

                    if neighbor in explored:
                        continue

                    h = heuristic(neighbor, target)
                    enqueued[(curr_node, neighbor)] = ncost, h
                    push(queue, (ncost + h, (curr_node, neighbor), ncost, curr_node))
            else:
                edge_colors[edge_idx] = "red"
                edge_width[edge_idx] = 2.0

        node_color = ["gray"] * self.G.number_of_nodes()
        for i in range(250, self.G.number_of_nodes()):
            node_color[i] = "blue"

        node_color[-1] = "green"
        node_color[-2] = "red"

        nx.draw_networkx_nodes(
                        self.G,
                        state,
                        nodelist=self.G.nodes,
                        node_size=60,
                        cmap=plt.cm.Reds_r,
                        node_color=node_color,
                        ax=axs
                    )
        nx.draw_networkx_edges(
            self.G, state, alpha=0.4, ax=axs, edge_color=edge_colors)
        sum_links_length = 3.5
        axs.set_xlim([-sum_links_length, sum_links_length])
        axs.set_ylim([-sum_links_length, sum_links_length])
        axs.set_aspect('equal', adjustable='box')
        # plt.show() 
        return None, checked_cnt, checked_list

    def visualize_request(self, planning_request: PlanningRequest, path=None):
        start = planning_request.start
        goal = planning_request.goal
        obstacles = planning_request.obstacles

        fk_start = self.manipulator.fk(start)
        fk_goal = self.manipulator.fk(goal)

        goal_plot_parameters = self.manipulator.get_links_plot_parameters(fk_goal, goal)
        start_plot_parameters = self.manipulator.get_links_plot_parameters(fk_start, start)

        # Visualize robots
        fig, ax = plt.subplots()
        ax.plot([0], [0])
        for plot_param in goal_plot_parameters:
            x = plot_param["x"]
            y = plot_param["y"]
            angle = plot_param["angle"]
            width = plot_param["width"]
            height = plot_param["height"]

            ax.add_patch(
                patches.Rectangle((x, y), height=height, width=width, angle=angle, facecolor="green"))
            # Have a look at this: https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html

        for plot_param in start_plot_parameters:
            x = plot_param["x"]
            y = plot_param["y"]
            angle = plot_param["angle"]
            width = plot_param["width"]
            height = plot_param["height"]
            ax.add_patch(
                patches.Rectangle((x, y), height=height, width=width, angle=angle, facecolor="blue"))

        for i in range(self.manipulator.dof):
            ax.add_patch(
                patches.Circle((fk_start[i][0], fk_start[i][1]), self.manipulator.links_w[i] * 0.55, facecolor='black'))
            ax.add_patch(
                patches.Circle((fk_goal[i][0], fk_goal[i][1]), self.manipulator.links_w[i] * 0.55, facecolor='black'))

        ax.add_patch(
            patches.Circle((fk_start[0][0], fk_start[0][1]), self.manipulator.links_w[0] * 0.65, facecolor='black'))

        sum_links_length = sum(self.manipulator.links) + 0.05
        ax.set_xlim([-sum_links_length, sum_links_length])
        ax.set_ylim([-sum_links_length, sum_links_length])
        ax.set_aspect('equal', adjustable='box')

        # Visualize the obstacles
        for obs in obstacles:
            x, y, r = obs["x"], obs["y"], obs["r"]
            ax.add_patch(patches.Circle((x, y), r, facecolor='orange'))
        ax.set_title("Green: Goal - Blue: Start")

    def visualize_path(self, req, path):
        obstacles = req.obstacles
        fig, ax = plt.subplots()
        ax.plot([0], [0])
        for i, jnt in enumerate(path):
            if i == 0:
                color = "green"
            elif i == len(path):
                color = "blue"
            else:
                color = "black"
            fk = self.manipulator.fk(jnt)
            plot_parameters = self.manipulator.get_links_plot_parameters(fk, jnt)

            for plot_param in plot_parameters:
                x = plot_param["x"]
                y = plot_param["y"]
                angle = plot_param["angle"]
                width = plot_param["width"]
                height = plot_param["height"]

                ax.add_patch(
                    patches.Rectangle((x, y), height=height, width=width, angle=angle, facecolor=color))
            # Have a look at this: https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html

        sum_links_length = sum(self.manipulator.links) + 0.05
        ax.set_xlim([-sum_links_length, sum_links_length])
        ax.set_ylim([-sum_links_length, sum_links_length])
        ax.set_aspect('equal', adjustable='box')

        # Visualize the obstacles
        for obs in obstacles:
            x, y, r = obs["x"], obs["y"], obs["r"]
            ax.add_patch(patches.Circle((x, y), r, facecolor='orange'))
        plt.show()

    def get_occupancy_map(self, req, resolution=0.05):
        # Knowing the number of grid per row / column
        sum_links_length = sum(self.manipulator.links) + 2*resolution
        n = int(np.ceil(2*sum_links_length/resolution))
        # Zero -> not occupied, One -> occupied
        occ_grid = np.zeros([n, n])

        # This implementation applies only to discs
        obstacles = req.obstacles
        for obs in obstacles:
            # Adding offset to x and y to make sure that the robot is in the center of the grid
            x = obs["x"] + sum_links_length
            y = obs["y"] + sum_links_length
            radius = obs["r"]

            # Get bounds of the obstacle
            left_x_index = max(0, int(np.floor((x - radius)/resolution)))
            left_y_index = max(0, int(np.floor((y - radius)/resolution)))
            right_x_index = min(n, int(np.ceil((x + radius)/resolution)))
            right_y_index = min(n, int(np.ceil((y + radius)/resolution)))

            # TODO: Could be better - maybe using a percentage
            for i in range(left_x_index, right_x_index):
                for j in range(left_y_index, right_y_index):
                    # Not use the lef and down...
                    dist_to_center = (((i+0.5)*resolution - x)**2 + ((j+0.5)*resolution - y)**2)**0.5
                    if dist_to_center < radius:
                        occ_grid[j, i] = 1
                    elif dist_to_center < radius + 0.5*resolution and occ_grid[j, i] == 0:
                        occ_grid[j, i] = 0.5

        return occ_grid

    def get_occupancy_map_with_robot(self, req, jnt, resolution=0.05):
        occ_grid_with_robot = self.get_occupancy_map(req, resolution)
        fk = self.manipulator.fk(jnt)

        sum_links_length = sum(self.manipulator.links) + 2 * resolution
        n = int(np.ceil(2 * sum_links_length / resolution))

        origin_x = n/2
        origin_y = n/2

        curr_x_idx = origin_x
        curr_y_idx = origin_y

        # Get grid coordinates of the robot
        for id in range(self.dof):
            next_x_idx = int(curr_x_idx + fk[id+1][0]/resolution)
            next_y_idx = int(curr_y_idx + fk[id+1][1]/resolution)

            min_x_index = int(np.floor(min(curr_x_idx, next_x_idx) - self.manipulator.links_w[id]/resolution))
            min_y_index = int(np.floor(min(curr_y_idx, next_y_idx) - self.manipulator.links_w[id]/resolution))
            max_x_index = int(np.ceil(max(curr_x_idx, next_x_idx) + self.manipulator.links_w[id]/resolution))
            max_y_index = int(np.ceil(max(curr_y_idx, next_y_idx) + self.manipulator.links_w[id]/resolution))

            # TODO: Could be better - maybe using a percentage
            for i in range(min_x_index, max_x_index):
                for j in range(min_y_index, max_y_index):
                    # Compute the distance to this point
                    center_point = ((i-origin_x+0.5)*resolution, (j-origin_y+0.5)*resolution)
                    dist_to_point = self.manipulator.get_min_distance_to_point(jnt, center_point)
                    if dist_to_point < 0:
                        occ_grid_with_robot[j, i] = 1
                    elif dist_to_point < resolution and occ_grid_with_robot[j, i] == 0:
                        occ_grid_with_robot[j, i] = 0.5
            curr_x_idx = next_x_idx
            curr_y_idx = next_y_idx

        return occ_grid_with_robot

    def visualize_occupancy_map(self, req, occ_grid):
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
        axs[0].pcolor(occ_grid, edgecolors='k')

        obstacles = req.obstacles
        # Visualize the obstacles
        for obs in obstacles:
            x, y, r = obs["x"], obs["y"], obs["r"]
            axs[1].add_patch(patches.Circle((x, y), r, facecolor='orange'))
        sum_links_length = sum(self.manipulator.links) + 0.05
        axs[1].set_xlim([-sum_links_length, sum_links_length])
        axs[1].set_ylim([-sum_links_length, sum_links_length])
        axs[0].set_aspect('equal', adjustable='box')
        axs[1].set_aspect('equal', adjustable='box')
        axs[0].set_title("Occupancy map")
        plt.show()

    def get_obstacle_space(self, req, contour_n=100):
        xs = np.linspace(-np.pi, np.pi, contour_n)
        ys = np.linspace(-np.pi, np.pi, contour_n)

        invalid_list = []
        nodes = [np.array([x_pos, y_pos]) for x_pos in xs for y_pos in ys]
        for i, node in enumerate(nodes):
            is_valid = self.manipulator.check_validity(node, req.obstacles)
            if not is_valid:
                invalid_list.append(node)
        return invalid_list

    def visualize_obstacle_space(self, req):
        obstacles_space = self.get_obstacle_space(req)
        _, ax = plt.subplots(nrows=1, ncols=1)

        obst_space_x = [ns[0] for ns in obstacles_space]
        obst_space_y = [ns[1] for ns in obstacles_space]
        ax.scatter(obst_space_x, obst_space_y, c="r", s=1)
        if self.G is not None:
            nx.draw_networkx_edges(self.G, pos=nx.get_node_attributes(self.G, "state"), ax=ax)
            nx.draw_networkx_nodes(self.G, pos=nx.get_node_attributes(self.G, "state"),
                                   ax=ax, node_size=15)
        # TODO: Add Cartesian space
        ax.set_title("Obstacle space - Red points are invalid")
        plt.show()


def get_2dof_planning_environment():
    # Define planning environment or planning agent
    dof = 2
    links = [0.5, 0.5]
    planning_range_max = np.array([np.pi, np.pi])
    planning_range_min = np.array([-np.pi, -np.pi])

    ma = manipulator(dof, links)
    pl_env = Planning(ma, planning_range_max, planning_range_min, resolution=0.05)
    return pl_env


if __name__ == "__main__":
    dof = 2
    links = [0.5, 0.5]
    ma = manipulator(dof, links)
    pl = Planning(ma)

    # Generate random request
    obstacles = generate_random_obstacles()
    req_random = ma.generate_random_queries(obstacles)

    # Get request from the data
    request_file = "../data/pl_req/hard_pl_req_250_nodes.json"
    hard_requests = load_planning_requests(request_file)
    req = hard_requests[72]
    pl.visualize_obstacle_space(req)
    """easy_request_file = "../data/pl_req/easy_pl_req_250_nodes.json"
    easy_requests = load_planning_requests(easy_request_file)
    req = easy_requests[7]

    # pl.visualize_obstacle_space(req)
    # pl.generate_graph_halton(100, obstacles=req.obstacles)
    # visualize_graphs([pl.G], "state", show_edges=True)

    pl.visualize_request(req)
    occ = pl.get_occupancy_map_with_robot(req, req.start)
    pl.visualize_occupancy_map(req, occ)


    # Generate a random halton Graph for searching
    pl.generate_graph_halton(n_nodes=10000)

    # Search for solution
    result = pl.search(req)
    if result.has_solution:
        solution_path = result.solution_path
        pl.visualize_path(req, solution_path)"""
