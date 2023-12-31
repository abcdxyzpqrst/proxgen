[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10034285.svg)](https://doi.org/10.5281/zenodo.10034285)

# Adaptive Proximal Gradient Methods for Structured Neural Networks (NeurIPS 2021)
This repo contains the official implementations of the paper "Adaptive Proximal Gradient Methods for Structured Neural Networks" published in NeurIPS 2021.
+ Jihun Yun (KAIST), Aurélie C. Lozano (IBM T.J. Watson Research Center), and Eunho Yang (KAIST, AITRICS)

# Abstract
We consider the training of structured neural networks where the regularizer can be non-smooth and possibly non-convex. While popular machine learning libraries have resorted to stochastic (adaptive) subgradient approaches, the use of proximal gradient methods in the stochastic setting has been little explored and warrants further study, in particular regarding the incorporation of adaptivity. Towards this goal, we present a general framework of stochastic proximal gradient descent methods that allows for arbitrary positive preconditioners and lower semi-continuous regularizers. We derive two important instances of our framework: (i) the first proximal version of Adam, one of the most popular adaptive SGD algorithm, and (ii) a revised version of ProxQuant for quantization-specific regularizers, which improves upon the original approach by incorporating the effect of preconditioners in the proximal mapping computations. We provide convergence guarantees for our framework and show that adaptive gradient methods can have faster convergence in terms of constant than vanilla SGD for sparse data. Lastly, we demonstrate the superiority of stochastic proximal methods compared to subgradient-based approaches via extensive experiments. Interestingly, our results indicate that the benefit of proximal approaches over sub-gradient counterparts is more pronounced for non-convex regularizers than for convex ones.

# Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-00075, Artificial Intelligence Graduate School Program(KAIST))
