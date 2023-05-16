# Projektpraktikum
# Reproduction of Stein Variational Probabilistic Roadmaps
## Overview
[Background](#Background)

## Background
Efficient and reliable generation of global path plans are necessary for safe execution and deployment of autonomous systems.In order to generate planning graphs which resolve the topology of a given environment, many sampling-based motion planners resort to coarse, but heuristically-driven strategies which often fail to generalize to new and varied surroundings.The paper "Stein Variational Probabilistic Roadmaps" proposes a method for Probabilistic Roadmaps which relies on particle-based Variational Inference to efficiently cover the posterior distribution over feasible regions in configuration space. Stein Variational Probabilistic Roadmap (SV-PRM) results in sample-efficient generation of planning-graphs and large improvements over traditional sampling approaches.



This project is about a reproduction of "Stein Variational Probabilistic Roadmaps". This paper propose a method for Probabilistic Roadmaps which relies on particle-based Variational Inference to efficiently cover the posterior distribution over feasible regions in configuration space. This method is Stein Variational Gradient Descent(SVGD), it results in sample-efficient generation of planning-graphs and large improvements over traditional sampling approaches.
Variational inference is a technique used in Bayesian inference to approximate the posterior distribution of a model, here q(x) is the approximate distribution, p(x|z) is the true posterior distribution.A proposal distribution q(x), belonging to a family Q, is chosen to minimize the KL-divergence with the target posterior distribution p(x|z) over latent variable x.Stein Variational Gradient Descent avoids the challenge of determining an appropriate Q by leveraging a non-parameteric, particle based representation of the posterior distribution.
# manipulator.py
This file is used to define the manipulator used for planning .Running this script, you can see the Cartesian space of a planner robot.The closest distances between manipulators and obstacles are shown in the figure.These distances will be used to calculate the posterior probability.
# sv_prm.py
In this file we transfer the obstacles into configuration space of manipulator and calculate the posterior probability. After that we initialize particles based on prior distribution,here we use uniform distribution. And then we use SVGD to update the sampled particles. Run this file we can get a video to show the iteration process of sampled particles and the final iteration result.
# planning.py
This file include the context of motion planning using a graph. To initiate the planning context, you have to first define a manipulator. Run this script to see a planning request and a occupancy map. If there is a solution in the given roadmap, this solution will also be visualized.
