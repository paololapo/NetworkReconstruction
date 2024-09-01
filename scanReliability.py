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

from methods import *


def scanWrapper(G, log_path, n_samples, delay, transient, offset):
    A = nx.adjacency_matrix(G).todense()
    A = (A > 0).astype(int)
    
    missing = scanMissing(A, n_samples, delay, transient, offset)
    spurious = scanSpurious(A, n_samples, delay, transient, offset)
    network = scanNetworkReliability(A, n_samples, delay, transient, offset)
    
    results = {"missing": missing, "spurious": spurious, "network": network}

    with open(log_path, "wb") as file:
        pickle.dump(results, file)


# Florentine Families Network
G = nx.florentine_families_graph().to_undirected()
print("Florentine Families Network")
scanWrapper(G, log_path="./logs/scan_florentine_families.pkl", 
            n_samples=1e4, 
            delay=100, 
            transient=100, 
            offset=120)

# Karate Club Network
G = nx.karate_club_graph().to_undirected()
print("\n Karate Club Network")
scanWrapper(G, log_path="./logs/scan_karate_club.pkl",
            n_samples=1e4, 
            delay=400,
            transient=500, 
            offset=600)

# Dolphins Network
G = nx.read_gml("dolphins.gml").to_undirected()
print("\n Dolphins Network")
scanWrapper(G, log_path="./logs/scan_dolphins.pkl",
            n_samples=5e3, 
            delay=800, 
            transient=900, 
            offset=2000)    




