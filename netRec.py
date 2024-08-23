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

# Florentine Families Network
G = nx.florentine_families_graph().to_undirected()
print("Florentine Families Network")
studyNetworkReconstructon(G, log_path="./logs/florentine_families.pkl",
                         n_samples=1e4, 
                         delay=50, 
                         transient=100, 
                         offset=120, 
                         verbose=True)

# Karate Club Network
G = nx.karate_club_graph().to_undirected()
print("\n Karate Club Network")
studyNetworkReconstructon(G, log_path="./logs/karate_club.pkl",
                          n_samples=1e4, 
                          delay=400, 
                          transient=500, 
                          offset=600, 
                          verbose=True)

# Dolphins Network
G = nx.read_gml("dolphins.gml").to_undirected()
print("\n Dolphins Network")
studyNetworkReconstructon(G, log_path="./logs/dolphins.pkl", 
                          n_samples=5e3, 
                          delay=800, 
                          transient=900, 
                          offset=2000, 
                          verbose=True)


