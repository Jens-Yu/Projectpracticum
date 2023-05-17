# Projectpracticum
# Reproduction of Stein Variational Probabilistic Roadmaps
## Overview
[Background](#Background)

[Process of the Project](#Process-of-the-Project)

-[1 Initialization of samples](#1-Initialization-of-samples)

-[2 Bayesian Occupancy Maps](#2-Bayesian-Occupancy-Maps)

-[3 Feasibility Distributions in Motion Planning](#3-Feasibility-Distributions-in-Motion-Planning)

-[4 Stein Variational Gradient Descent](#4-Stein-Variational-Gradient-Descent)

[Result](#Result)

## Background
Efficient and reliable generation of global path plans are necessary for safe execution and deployment of autonomous systems.In order to generate planning graphs which resolve the topology of a given environment, many sampling-based motion planners resort to coarse, but heuristically-driven strategies which often fail to generalize to new and varied surroundings.The paper "Stein Variational Probabilistic Roadmaps" proposes a method for Probabilistic Roadmaps which relies on particle-based Variational Inference to efficiently cover the posterior distribution over feasible regions in configuration space. Stein Variational Probabilistic Roadmap (SV-PRM) results in sample-efficient generation of planning-graphs and large improvements over traditional sampling approaches.Here is the compare of sv-prm and traditional prm. The left picture is the result of sv-prm, and the other one is traditional prm.
![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/3e9678ea-3bab-403b-a727-95338fe761f8)
![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/7bc62ac1-b956-450c-b54a-8f153a65fbf5)

## Process-of-the-Project
### 1-Initialization-of-samples

As the left figure shows , run the file manipulator.py, you can see the Cartesian space of a planner robot.The closest distances between manipulators and obstacles are shown in the figure.These distances will be used to calculate the posterior probability. And then we transfer the obstacles into configuration space of manipulator.

<img width="400" height="400" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/f54b8086-2822-4561-9047-b240f522c2c5">
<img width="400" height="400" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/ae8f7c5b-8c4e-4d4b-baa2-e9910786a40c">

Particles are initialized based on prior distribution,here we use uniform distribution. The black points in the figure below are the initial particles.
<img alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/4de0e820-a28e-4f20-8028-a022116b7680" width="400" height="400">

### 2-Bayesian-Occupancy-Maps

A Bayesian occupancy map is a probabilistic approach for modeling and updating a map of an environment.In a Bayesian occupancy map, each cell in a grid map represents the probability of whether that cell is occupied or unoccupied.Desire a sampling distribution having high probability in the safe set S, and low probability elsewhere. Represent the probability of a given point x∈𝑹^𝑑 being collision-free by the feasibility likelihood 𝑝(𝑧=1|𝐱;𝜃)with map parameter 𝜃∈𝑹^𝑚. The occupancy indicator variable 𝑧∈{0,1} labels a given location as being in-collision (z = 0) or collision-free (z =1).
Using Bayes’ Rule, we can obtain a posterior probability over collision-free space:
  𝑝(𝐱│z=1;θ)=ƞ𝑝(𝑧=1│𝐱;θ)𝑝(𝐱)
Where 𝑝(𝐱│z=1;θ)  is the probability of sampling a point in a non-occupied area, and 𝑝(𝑧=1│𝐱;θ)  is the probability of the sampled point not being in collision ,𝑝(𝐱) is a prior probability, and ƞ a normalizing factor.
![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/ad4263cc-321a-47e1-962f-867ab24ed581)

In this case we want to optimize the posterior probability. So based on the formula above we should calculate the likelihood first.

### 3-Feasibility-Distributions-in-Motion-Planning
Based on some reference papers we can get following funktions:

Log-likelihood: −𝑙𝑜𝑔𝑝(𝑧=1│𝐱;θ)=𝛼|(|𝐡(𝐱)|)|^2+𝑐𝑜𝑛𝑠𝑡

h(x) = [c(x; sj )]j=1:K

<img width="300" height="100" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/2c240e45-21d4-4a63-83f2-84ab26c21780">

x: trajectory state

sj:collection of body spheres, j = 1:k

𝑑(𝐱,𝑠𝑗 ):the distance from the surface of 𝑠𝑗 to the nearest obstacle

𝜖: penalizes a state if a Cartesian point on the robot is within an 𝜖 -distance from the surface of the nearest obstacle

From these formulas we can know that if we want to get the likelihood we should calculate the obstacle cost first. As we can see the cost is bigger if the manipulator is close to obstacles.

<img width="400" height="300" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/2cf54eee-2d4c-4817-a307-6bbe707d67c4">
<img width="300" height="200" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/5f7da2db-dce3-4dee-82ea-e818af5e6543">

These costs are combined across spheres to construct an obstacle-cost vector h(x),and define the total obstacle cost as the scaled inner-product. This picture below is the scaled inner-product of h(x) , the brighter area means the value is bigger.

![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/5d775f9a-d678-4dfa-a205-208ea92ce661)

Then we can use finite difference method to calculate gradient of log likelihood. And we can also get gradient of log posterior according to Bayes’ rule 


### 4-Stein-Variational-Gradient-Descent

Variational inference is a technique used in Bayesian inference to approximate the posterior distribution of a model, here q(x) is the approximate distribution, p(x|z) is the true posterior distribution. A proposal distribution q(x), belonging to a family Q, is chosen to minimize the KL-divergence with the target posterior distribution p(x|z) over latent variable x:

![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/17d53d11-1182-4aaa-9c86-a5002ca75a51)

Stein Variational Gradient Descent avoids the challenge of determining an appropriate Q by leveraging a non-parameteric, particle based representation of the posterior distribution.

A set of particles ![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/6ceffaf2-06b7-482f-bff6-0360469d79a8)

iteratively updated according to ![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/af454200-a90e-44bd-93e9-294e1e32cc28)

The function ∅^*(.) lies in the unit-ball of a reproducing kernel Hilbert space (RKHS). This RKHS is characterized by a positive-definite kernel k(.,.). The term ∅^*(.) represents the optimal perturbation or velocity field (i.e. gradient direction) which maximally decreases the KL-divergence:
![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/79979ea0-8a25-4d69-82ff-e53797589128)

where  q([ϵ∅]) indicates the particle distribution resulting from taking an update step. This has been shown to yield a closedform solution which can be interpreted as a functional gradient in RKHS, and can be approximated with the set of particles:
![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/4a45ced6-a952-43fd-b834-a58c2a15df9f)

And this function has two terms that control different aspects of the algorithm. The first term is essentially a scaled gradient of the log-likelihood over the posterior’s particle approximation. The second term is known as the repulsive force. Intuitively, it pushes particles apart when they get too close to each other and prevents them from collapsing into a single mode.

<img width="250" height="25" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/56347e0b-ddee-4893-8449-acd04db2be89"> RBF kernel, where h is the bandwidth parameter

Hessian matrix to be 𝐻(𝑥)=−∇_𝑥^2 𝑙𝑜𝑔𝑝(𝑥|𝑧)

The metric 
<img width="100" height="25" alt="image" src="https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/023832d9-1ce6-4f53-a2e3-6beddf83a5ff">

## Result

Run the file sv_prm.py, it will do all calculations above. And it will show the process of iterations, then get the final result.



https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/e582ed9c-c28a-419c-bb01-ebb4eb4cb9d2

![image](https://github.com/Jens-Yu/Projektpraktikum/assets/122354667/036808e2-cef4-468b-9d94-61b5ddaed8a1)








