import networkx as nx
import numpy as np
from scipy.special import binom
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import multiprocessing
from time import time
import pickle
from pprint import pprint
from datetime import datetime

from methods import *


def networkReconstruction(A_obs, n_samples, delay, transient, offset, verbose=False, A=None, full_output=False):
    """
    Reconstruct the network from the observed adjacency matrix A_obs
    """
    if verbose: 
        assert A is not None, "If verbose, the true adjacency matrix must be provided"

    # Define the hyperparameters
    hyper = (n_samples, delay, transient)

    # Initialize the current adjacency matrix
    A_cur = A_obs.copy()

    proposed = []
    accepted = []

    if verbose: print("Starting network reconstruction...")
    for j in range(5): # This is the maximum number of iterations
        if verbose: print("Starting iteration ", j, " on ", datetime.now().strftime("%A: %H:%M"))

        if j == 3: A_2 = A_cur.copy()

        # Compute the network reliability and the link reliability matrix
        R_N = getNetworkReliability(A_cur, A_obs, *generatePartitionsSet(A_cur, *hyper), offset=offset)
        if verbose: print("Network reliability: ", R_N)
        R_L = computeLinkReliabilityMatrix(A_cur, *generatePartitionsSet(A_cur, *hyper), offset=offset)
        
        # Sort the links (increasing R) and not links (decreasing R)
        sorted_links, sorted_not_links = sortLinkLists(*getLinkLists(A_cur), R_L)
        
        # Iterate to reconstruct the network
        num_iters = np.min([len(sorted_links), len(sorted_not_links)])
        miss_update = 0
        done_update = False
        for k in range(num_iters):
            # Choose a pair of link and not link in order
            link, not_link = sorted_links[k], sorted_not_links[k]

            # Define the proposed adjacency matrix
            A_temp = A_cur.copy()
            if verbose: missing_before, spurious_before = compareAdjacencyMatrices(A, A_temp)

            # Swap the links
            A_temp[link[0], link[1]] = 0
            A_temp[link[1], link[0]] = 0
            A_temp[not_link[0], not_link[1]] = 1
            A_temp[not_link[1], not_link[0]] = 1

            if verbose:
                missing_after, spurious_after = compareAdjacencyMatrices(A, A_temp)
                print("Proposed (missing, spurious): (", len(missing_before), ", ", len(spurious_before), ") -> (", len(missing_after), ", ", len(spurious_after), ")")
                proposed.append(2*len(missing_after))

            # Compute the network reliability with the new adjacency matrix
            R_N_temp = getNetworkReliability(A_temp, A_obs, *generatePartitionsSet(A_temp, *hyper), offset=offset)
            
            # If the network reliability increases, update the adjacency matrix
            if R_N_temp > R_N:
                if verbose: print("Network accepted \n")
                accepted.append(1)
                A_cur = A_temp
                R_N = R_N_temp
                miss_update = 0
                done_update = True
            else:
                miss_update += 1
                if verbose: print("Network rejected")
                accepted.append(0)
                if miss_update > 4:
                    break
                
        if not done_update:
            if verbose: print("No further update possible")
            break

    if j < 3: A_2 = None

    if full_output: return A_cur, A_2, proposed, accepted
    return A_cur


# Florentine Families Network
#G = nx.florentine_families_graph().to_undirected()
#print("Florentine Families Network")
#studyNetworkReconstructon(G, log_path="./logs/netRec_florentine_families.pkl",
#                         n_samples=1e4, 
#                         delay=50, 
#                         transient=100, 
#                         offset=120, 
#                         verbose=True)

# Karate Club Network
G = nx.karate_club_graph().to_undirected()
print("\n Karate Club Network")
studyNetworkReconstructon(G, log_path="./logs/netRec_karate_club.pkl",
                          n_samples=1e4, 
                          delay=400, 
                          transient=500, 
                          offset=600, 
                          verbose=True)

# Dolphins Network
G = nx.read_gml("dolphins.gml").to_undirected()
print("\n Dolphins Network")
studyNetworkReconstructon(G, log_path="./logs/netRec_dolphins.pkl", 
                          n_samples=5e3, 
                          delay=800, 
                          transient=900, 
                          offset=2000, 
                          verbose=True)


