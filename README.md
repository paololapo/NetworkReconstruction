# Network Reconstruction
This repository contains two projects dealing with complex networks and link prediction:
* [**MCMC:**](https://github.com/paololapo/NetworkReconstruction/tree/main/MCMC) Bayesian framework to identify missing and spurious interactions and reconstruct a partially corrupted network.
* [**GNN:**](https://github.com/paololapo/NetworkReconstruction/tree/main/GNN)  deep-learning architecture to infer the full structure of a partially observed graph.

**Authors' contributions:** [Paolo Lapo Cerni](https://github.com/paololapo): MCMC, GNN; [Lorenzo Vigorelli](https://github.com/LorenzoVigorelli): GNN.

## Bayesian link reliability (MCMC)
Observed networks often come with missing and spurious interactions, i.e. they are corrupted. In this work, we reviewed a Bayesian framework to assess the *reliability* of a link. Such metric is defined as an ensemble average over network instances originating from the stochastic block model. This involves sampling over such space using the Metropolis algorithm (Monte Carlo Markov Chain).  

*Main reference:* Guimerà, R. and Sales-Pardo, M., 2009. Missing and spurious interactions and the reconstruction of complex networks. *Proceedings of the National Academy of Sciences, 106(52)*, pp.22073-22078.

## Graph Neural Networks (GNN)
In this work, we tackle the problem of reconstructing the full network starting from a subset of the edges and some node features. GNNs are an example of *geometric deep learning*, due to their necessity of being node equivariant. This project mainly focuses on graph convolutional autoencoder and the message-passing framework, tested going beyond the standard cases and applications.  

*Main reference:* Kipf, T.N. and Welling, M., 2016. Variational graph auto-encoders. *arXiv preprint arXiv:1611.07308*.
