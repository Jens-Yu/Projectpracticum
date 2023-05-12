# This file is used to define the manipulator used for planning
# Running this script, you can see the Cartesian space of a 3D planner robot
# The closest distances are shown in the figure

from utilities import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class manipulator:
    def __init__(self, dof, links, links_width=None, motion_range = None, safety_margin=0.00) -> None:
        self.dof = dof
        self.links = links
        self.links_w = links_width if links_width is not None else [0.05]*len(links)
        self.safety_margin= safety_margin

        motion_range = motion_range if motion_range is not None else [[-np.pi, np.pi]]*dof
        self.jnt_lower_bound = np.array([jnt[0] for jnt in motion_range])
        self.jnt_ranges = np.array([jnt[1] - jnt[0] for jnt in motion_range])

    def fk(self, jnt) -> list:
        """Compute the forward kinematics

        Args:
            jnt (list): joint configurations

        Returns:
            list: Return a list of the joint positions
        """

        jnt_cart = []
        origin = np.zeros([2, 1])
        rot_mat = np.diag([1, 1])
        jnt_cart.append(origin)
        for i in range(self.dof):
            cos1 = np.cos(jnt[i])
            sin1 = np.sin(jnt[i])
    
            rot_mat = rot_mat * np.matrix([[cos1, -sin1],
                [sin1, cos1]])

            link = np.array([self.links[i], 0], ndmin=2).transpose()
            next_point = rot_mat*link + origin
            jnt_cart.append(next_point)
            origin = next_point
         
        return jnt_cart
    
    def compute_distance(self, link_start, link_end, link_w, obstacle):
        obs_scale = link_w*0.5 + self.safety_margin
        obstacle_r = obstacle['r']
        obstacle_x = obstacle['x']
        obstacle_y = obstacle['y']
        obs_origin = np.array([obstacle_x, obstacle_y], ndmin=2).transpose()
        

        dist_vec = self.compute_distance_vecotor(link_start, link_end, obstacle)
        link_obs_origin_dist = np.linalg.norm(dist_vec["link"] - obs_origin)

        dist = np.linalg.norm(dist_vec["link"] - dist_vec["obs"])
        if link_obs_origin_dist > obstacle_r:
            dist = dist - obs_scale
        else:
            dist=-dist - obs_scale
        return dist

    def compute_distance_to_point(self, link_start, link_end, link_w, point):
        obs_scale = link_w * 0.5 + self.safety_margin
        obstacle = {}
        obstacle["r"] = 0.0001
        obstacle["x"] = point[0]
        obstacle["y"] = point[1]
        obs_origin = np.array([point[0], point[1]], ndmin=2).transpose()

        dist_vec = self.compute_distance_vecotor(link_start, link_end, obstacle)
        link_obs_origin_dist = np.linalg.norm(dist_vec["link"] - obs_origin)

        dist = np.linalg.norm(dist_vec["link"] - dist_vec["obs"])
        dist = dist - obs_scale if link_obs_origin_dist > 0.0001 else -dist - obs_scale

        return dist
    
    def compute_distance_vecotor(self, link_start, link_end, obstacle):
        obstacle_x = obstacle['x']
        obstacle_y = obstacle['y']
        obstacle_r = obstacle['r']
        obstacle_xy = np.array([obstacle_x, obstacle_y], ndmin=2).transpose()

        link_vec = link_end - link_start
        start_obs_vec = obstacle_xy - link_start 

        link_length = np.linalg.norm(link_vec)

        projection_link_obs = \
            link_vec.transpose()*start_obs_vec / link_length
        
        projection_link_obs = projection_link_obs.item()

        if projection_link_obs > link_length:
            closest_point = link_end
        elif projection_link_obs < 0:
            closest_point = link_start
        else:
            closest_point = link_start + projection_link_obs/link_length * link_vec

        dist = np.linalg.norm((obstacle_xy - closest_point))
        obs_closest_point = (obstacle_xy - closest_point)*(1 - obstacle_r/dist) + closest_point
        dist_vec = {"link": closest_point, "obs": obs_closest_point}
        
        return dist_vec
    
    def check_self_validity(self, jnt):
        if self.dof == 2:
            return True

        jnt_cart = self.fk(jnt)
        for i in range(self.dof-2):
            x_1, y_1 = jnt_cart[i][0], jnt_cart[i][1]
            x_2, y_2 = jnt_cart[i+1][0], jnt_cart[i+1][1]
            for j in range(i+2, self.dof):
                x_3, y_3 = jnt_cart[j][0], jnt_cart[j][1]
                x_4, y_4 = jnt_cart[j+1][0], jnt_cart[j+1][1]
                # Intersection check according to https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
                t = ((x_1 - x_3)*(y_3 - y_4) - (y_1 - y_3)*(x_3 - x_4)) / ((x_1 - x_2)*(y_3 - y_4) - (y_1 - y_2)*(x_3 - x_4))
                u = ((x_1 - x_3)*(y_1 - y_2) - (y_1 - y_3)*(x_1 - x_2)) / ((x_1 - x_2)*(y_3 - y_4) - (y_1 - y_2)*(x_3 - x_4))

                length_width_ratio = self.links_w[i]/self.links[i]
                self_collision = (0-length_width_ratio < t < 1+length_width_ratio) and \
                                 (0-length_width_ratio < u < 1+length_width_ratio)
                if self_collision:
                    return False
        return True
    
    def check_validity(self, jnt, obstacles):
        if obstacles is None:
            return self.check_self_validity(jnt)

        fk_val = self.fk(jnt)
        for i in range(self.dof):
            for obs in obstacles:
                dist = self.compute_distance(fk_val[i], fk_val[i+1], self.links_w[i], obs)
                if dist < 0:
                    return False
        return True
    
    def get_min_distances(self, jnt, obstacles):
        fk_val = self.fk(jnt)
        min_dist_list = []
        for i in range(self.dof):
            min_dist = 100
            for obs in obstacles:
                dist = self.compute_distance(fk_val[i], fk_val[i+1], self.links_w[i], obs)
                min_dist = min(dist, min_dist)
            
            min_dist_list.append(min_dist)
        return min_dist_list

    def get_min_distance_to_point(self, jnt, point):
        fk_val = self.fk(jnt)
        min_dist = 100
        for i in range(self.dof):
            dist = self.compute_distance_to_point(fk_val[i], fk_val[i + 1], self.links_w[i], point)
            min_dist = min(dist, min_dist)
        return min_dist
    
    def get_dist_vectors(self, jnt, obstacles):
        fk_val = self.fk(jnt)
        dist_vec_list = []

        for i in range(self.dof):
            link_dist_list = []
            for obs in obstacles:
                dist_vec = self.compute_distance_vecotor(fk_val[i], fk_val[i+1], obs)
                link_dist_list.append(dist_vec)
            
            dist_vec_list.append(link_dist_list)
        return dist_vec_list
    
    def get_min_dist_vectors(self, jnt, obstacles):
        min_dist_vec_list = []
        dist_vec_list = self.get_dist_vectors(jnt, obstacles)
        for link_dist_vectors in dist_vec_list:
            min_vec = None
            min_dist = 100

            for i, vec in enumerate(link_dist_vectors):
                dist = np.linalg.norm(vec["link"] - vec["obs"])
                if dist < min_dist:
                    min_dist = dist
                    min_vec = vec
            
            min_dist_vec_list.append(min_vec)
        return min_dist_vec_list
    
    def visualize(self, jnt, obstacles=None, config=None):
        # dist_vectors = self.get_dist_vectors(jnt, obstacles)
        # fig = plt.figure()

        fk = self.fk(jnt)
        plot_parameters = self.get_links_plot_parameters(fk, jnt)
        
        # Visualize robots
        fig, ax = plt.subplots()
        ax.plot([0], [0])
        for plot_param in plot_parameters:
            x = plot_param["x"]
            y = plot_param["y"]
            angle = plot_param["angle"]
            width = plot_param["width"]
            height = plot_param["height"]

            ax.add_patch(patches.Rectangle((x, y), height=height, width=width, angle=angle))
            # Have a look at this: https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html

        for i in range(self.dof):
            ax.add_patch(patches.Circle((fk[i][0], fk[i][1]), self.links_w[i]*0.55, facecolor='black'))
        
        ax.add_patch(patches.Circle((fk[0][0], fk[0][1]), self.links_w[0]*0.75, facecolor='brown'))
         
        sum_links_length = sum(self.links) + 0.05
        ax.set_xlim([-sum_links_length, sum_links_length])
        ax.set_ylim([-sum_links_length, sum_links_length])
        ax.set_aspect('equal', adjustable='box')

        # Visualize the obstacles
        if obstacles is not None:
            for obs in obstacles:
                x, y, r= obs["x"], obs["y"], obs["r"]
                ax.add_patch(patches.Circle((x, y), r, facecolor='orange'))

            # Visualize the distance vectors
            dist_vectors = self.get_min_dist_vectors(jnt, obstacles)
            for dist_vec in dist_vectors:
                x = dist_vec["link"][0].item()
                y = dist_vec["link"][1].item()
                dx = dist_vec["obs"][0].item() - x
                dy = dist_vec["obs"][1].item() - y
                plt.arrow(x, y, dx, dy)

        # plt.grid()
        plt.show()
    
    def get_links_plot_parameters(self, fk, jnt):
        plot_parameters = []
        rotated_angle = 0
        adjusted_x = 0
        adjusted_y = 0
        jnt_value = 0
        for i in range(self.dof):
            rotated_angle += jnt[i]/np.pi*180
            jnt_value += jnt[i]

            adjusted_x =  np.sin(jnt_value)*self.links_w[i]/2 + fk[i][0].item()
            adjusted_y = -np.cos(jnt_value)*self.links_w[i]/2 + fk[i][1].item()

            plot_param = {"angle": rotated_angle}
            plot_param.update({"height": self.links_w[0]})
            plot_param.update({"width": self.links[0]})
            plot_param.update({"x": adjusted_x})
            plot_param.update({"y": adjusted_y})

            plot_parameters.append(plot_param)

        return plot_parameters

    def generate_random_queries(self, obstacles):
        # np.random.rand() generate random numbers from [0, 1]
        start_valid, goal_valid = False, False
        cnt = 0
        
        while start_valid is False and cnt < 100:
            start_config = np.random.rand(self.dof) * self.jnt_ranges
            start_config = start_config + self.jnt_lower_bound
            start_valid = self.check_validity(start_config, obstacles)
            cnt += 1
        
        if cnt >= 100:
            print("start configuration is not valid")
            return None
        
        cnt = 0
        while goal_valid is False and cnt < 100:
            goal_config = np.random.rand(self.dof) * self.jnt_ranges
            goal_config = goal_config + self.jnt_lower_bound
            goal_valid = self.check_validity(goal_config, obstacles)
            cnt += 1
        
        if cnt >= 100:
            print("Goal configuration is not valid")
            return None

        req = PlanningRequest(goal=goal_config, start=start_config)
        req.obstacles = obstacles
        req.dof = self.dof

        return req


if __name__ == "__main__":
    dof = 2
    links = [0.5, 0.5]
    
    ma = manipulator(dof, links)
    obstacles = generate_random_obstacles()
    request_file = "../data/pl_req/hard_pl_req_250_nodes.json"
    hard_requests = load_planning_requests(request_file)
    req = hard_requests[72]
    # ma.visualize(req.start, obstacles)

    jnt = np.array([np.pi*0.7, np.pi*0.62])
    print(ma.check_validity(jnt, obstacles=None))
    ma.visualize(jnt, obstacles)
