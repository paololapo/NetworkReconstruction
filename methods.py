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


# =====================================================================
#                About Networks and Adjacency matrices
# =====================================================================

def sillyAdjacencyMatrix(N):
    """
    Generate a random adjacency matrix
    """
    upper_triangular = np.triu(np.random.randint(0, 2, size=(N, N)), 1)
    A = mirrorUpperTri(upper_triangular)
    np.fill_diagonal(A, np.random.randint(0, 2, size=N))
    return A


def linksBetween(A, groups, g1, g2):
    """
    Compute the number of links between two groups
    """
    l = 0
    for i in groups[g1]:
        for j in groups[g2]:
            l += A[i, j]
    return l


def maxBetween(groups, g1, g2):
    """
    Compute the maximum number of links between two groups
    """
    return len(groups[g1]) * len(groups[g2])


def findGroup(groups, i):
    """
    Find the group of a given node
    """
    for g, group in enumerate(groups):
        if i in group:
            return g
    
    return None


def arrayGroups(groups):
    """
    Convert the groups data structure to an array of N elements
    """
    N = len(groups)
    belongs_to = np.full(N, -1, dtype=int)
    for i in range(N):
        belongs_to[i] = findGroup(groups, i)

    return belongs_to


def swap(groups, i, g_new):
    """
    Swap a node i from group to group g
    """
    # Copy the groups data structure
    new_groups = copy.deepcopy(groups)
    
    # Swap the node
    g_old = findGroup(groups, i)
    new_groups[g_old].remove(i)
    new_groups[g_new].append(i)

    return new_groups


def getLinkLists(A):
    """
    Get the list of links and not links from the adjacency matrix
    """
    N = A.shape[0]
    
    links = []
    not_links = []

    for i in range(N):
        for j in range(i, N):
            if A[i, j] == 1:
                links.append((i, j))
            else:
                not_links.append((i, j))

    return links, not_links


def compareAdjacencyMatrices(A, A_obs):
    """
    Compare the adjacency matrices returning the missing and spurious links
    """
    A_links, A_not_links = getLinkLists(A)
    A_obs_links, A_obs_not_links = getLinkLists(A_obs)

    missing = [link for link in A_links if link not in A_obs_links]
    spurious = [link for link in A_obs_links if link not in A_links]

    return missing, spurious


def mirrorUpperTri(A):
    """
    Obtain undirected network overwriting the lower triangular part of A
    """
    N = A.shape[0]
    A = A.copy()
    for i in range(N):
        for j in range(i, N):
            A[j, i] = A[i, j]
    return A


def corruptAdjacencyMatrix(A, p, mode):
    """
    Corrupts the adjacency matrix A by adding missing and/or spurious edges
    """
    assert mode in ['missing', 'spurious', 'both'], "Invalid mode"
    assert p >= 0 and p <= 1, "Invalid probability"

    links, not_links = getLinkLists(A)
    num_noisy_links = int(np.ceil(len(links) * p))
    
    A_corrupted = A.copy()

    if mode == 'missing' or mode == 'both':
        # Randomly select edges to remove and remove them
        edges_to_remove = np.random.choice(len(links), num_noisy_links, replace=False)
        for edge in edges_to_remove:
            i, j = links[edge]
            A_corrupted[i, j] = 0
            A_corrupted[j, i] = 0

    if mode == 'spurious' or mode == 'both':
        # Randomly select edges to add and add them
        edges_to_add = np.random.choice(len(not_links), num_noisy_links, replace=False)
        for edge in edges_to_add:
            i, j = not_links[edge]
            A_corrupted[i, j] = 1
            A_corrupted[j, i] = 1

    # Ensure A = A.T
    A_corrupted = mirrorUpperTri(A_corrupted)

    return A_corrupted




# =====================================================================
#                About the Hamiltonian and Sampling
# =====================================================================

def hamiltonian(A_obs, groups):
    """
    Compute the Hamiltonian of the system
    """
    N = A_obs.shape[0]

    def h(A_obs, groups, g1, g2):
        r = maxBetween(groups, g1, g2)
        l = linksBetween(A_obs, groups, g1, g2)
        return np.log(r+1) + np.log(binom(r, l)+1)
            
    H = 0
    for g2 in range(N):
        for g1 in range(g2+1):
            H += h(A_obs, groups, g1, g2)
    
    return H


def singleStep(groups, H, A):
    """
    Perform a single step of the Metropolis-Hastings algorithm
    """
    N = A.shape[0]
    
    # Randomly select a node and a group
    i = np.random.randint(0, N)
    g_prop = np.random.randint(0, N)

    # Move the node to another group
    groups_prop = swap(groups, i, g_prop)
    
    # Compute the Hamiltonian of the new configuration
    H_prop = hamiltonian(A, groups_prop)

    # Acceptance probability
    if H_prop <= H:
        groups = groups_prop
        H = H_prop
    else:
        r = np.random.rand()
        if r < np.exp(H - H_prop):
            groups = groups_prop
            H = H_prop

    return groups, H


def samplingBranch(A, groups, n_samples, delay, seed, return_H=False):
    """
    Sampling algorithm (at equilibrium) to parallelize the computation
    """
    np.random.seed(seed)
    

    partitions_set = []
    hamiltonians_list = []

    H = hamiltonian(A, groups)

    for k in range(n_samples):
        for _ in range(delay):
            groups, H = singleStep(groups, H, A)
        partitions_set.append(groups)
        hamiltonians_list.append(H)

    if return_H:
        return partitions_set, hamiltonians_list
    else:
        return partitions_set
    

def parallelPartitionsSet(A, groups_init, n_samples, delay, n_cores=-1, return_H=True):
    """
    Get the set of partitions after the transient (at equilibrium)
    """
    if n_cores == -1:
       n_cores = multiprocessing.cpu_count()

    n_samples = int(n_samples) # Cast n_samples
    n_samples_per_core = n_samples//n_cores
    # if (n_samples_per_core*n_cores < n_samples):
    #     print("Warning: n_samples is not a multiple of n_cores, computing ", n_samples_per_core*n_cores, " samples instead.")

    seeds = np.random.randint(0, 2**32-1, n_cores) # Random seeds to avoid the same partitions
    input = [(A, groups_init, n_samples_per_core, delay, seeds[i], return_H) for i in range(n_cores)]

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.starmap(samplingBranch, input)

    if return_H:
        partitions_set = [item for sublist in [results[i][0] for i in range(n_cores)] for item in sublist]
        hamiltonians_list = [item for sublist in [results[i][1] for i in range(n_cores)] for item in sublist]
        return partitions_set, hamiltonians_list
    else:
        partitions_set = [item for sublist in results for item in sublist]
        return partitions_set
    

def generatePartitionsSet(A, n_samples, delay, transient, n_cores=-1, return_H=True):
    """
    Generate the set of partitions from the adjacency matrix A
    """
    # Groups data structure: list of lists
    N = A.shape[0]
    groups = [[] for _ in range(N)]

    # Uniformly random initialization
    for i in range(N): 
        g = np.random.randint(0, N)
        groups[g].append(i)

    # Transient
    H = hamiltonian(A, groups)
    for _ in range(transient):
        groups, H = singleStep(groups, H, A)
    
    return parallelPartitionsSet(A, groups, n_samples, delay, n_cores, return_H)


def getTimeScales(A_obs, max_iter = 1e3, trials=5):
    """
    Compute the time scales: tau_transient, tau_nmi, mean_transient
    """
    max_iter = int(max_iter)
    N = A_obs.shape[0]
    
    def singleTrial(A_obs, max_iter):       
        # Random initialization
        groups = [[] for _ in range(N)]
        for i in range(N): 
            g = np.random.randint(0, N)
            groups[g].append(i)

        # Transient
        transient = np.zeros(max_iter+1)
        H = hamiltonian(A_obs, groups)
        transient[0] = H
        for k in range(max_iter):
            groups, H = singleStep(groups, H, A_obs)
            transient[k+1] = H

        # NMI at equilibrium
        belongs_to = arrayGroups(groups)
        nmi = np.zeros(max_iter)
        for k in range(max_iter):
            groups, H = singleStep(groups, H, A_obs)
            nmi[k] = normalized_mutual_info_score(belongs_to, arrayGroups(groups))


        # NMI time scale
        tau_nmi = np.where(nmi < np.mean(nmi))[0][0]

        # Transient time scale
        num_bins = len(transient) // tau_nmi
        binned_mean_transient = transient[:num_bins*tau_nmi].reshape(num_bins, tau_nmi).mean(axis=1)
        binned_time = np.arange(num_bins*tau_nmi).reshape(num_bins, tau_nmi).mean(axis=1)
        tau_transient = binned_time[np.where(binned_mean_transient < np.mean(transient))[0][0]] + tau_nmi/2
        
        return tau_transient, tau_nmi, np.mean(transient)
  

    # Multiple trials
    time_scales = np.zeros((trials, 3))
    for i in tqdm(range(trials)):
        time_scales[i] = singleTrial(A_obs, max_iter)

    return np.max(time_scales, axis=0)




# =====================================================================
#                About Reliability and Performance
# =====================================================================

def singleLinkReliability(A, groups, i, j, offset, H=None, return_H=False):
    """
    Reliability of a given link for a particular configuration
    """
    g_i = findGroup(groups, i)
    g_j = findGroup(groups, j)

    r = maxBetween(groups, g_i, g_j)
    l = linksBetween(A, groups, g_i, g_j)

    if H is None: H = hamiltonian(A, groups) # Possibly to save computation

    if return_H:
        return (l + 1)*np.exp(-np.float128(H - offset))/(r + 2), H
    else:
        return (l + 1)*np.exp(-np.float128(H - offset))/(r + 2)
    

def linkReliabilityEntry(A, partitions_set, i, j, hamiltionian_list, offset):
    """
    Helper function to parallelize the computation of the link reliability matrix 
    """
    l_r = 0
    for k in range(len(partitions_set)):
        l_r += singleLinkReliability(A, partitions_set[k], i, j, offset, hamiltionian_list[k])
    return (i, j), l_r


def computeLinkReliabilityMatrix(A, partitions_set, hamiltionian_list=None, offset=250, n_cores=-1):
    """
    Compute the link reliability matrix for a given configuration
    """
    if hamiltionian_list is not None: 
        assert len(partitions_set) == len(hamiltionian_list), "partitions_set and hamiltionian_list must have the same length"

    if n_cores == -1:
       n_cores = multiprocessing.cpu_count()

    N = A.shape[0]
    link_reliability = np.zeros((N, N))
        
    if hamiltionian_list is not None:        
        # Parallel computation
        input = [(A, partitions_set, i, j, hamiltionian_list, offset) for i in range(N) for j in range(i, N)]
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(linkReliabilityEntry, input)

        for (i, j), l_r in results:
            link_reliability[i, j] = l_r
        
        # # Local computation
        # for i in range(N):
        #     for j in range(i, N):
                
        #         for k in range(len(partitions_set)):
        #             link_reliability[i, j] += singleLinkReliability(A, partitions_set[k], i, j, hamiltionian_list[k])
    
    else:
        hamiltionian_list = []
        for i in range(N):
            for j in range(i, N):
                for k in range(len(partitions_set)):
                    l_r, H = singleLinkReliability(A, partitions_set[k], i, j, offset, return_H=True)
                    link_reliability[i, j] += l_r
                    hamiltionian_list.append(H)

    # Symmetrize the matrix
    link_reliability = link_reliability + link_reliability.T - np.diag(np.diag(link_reliability))

    # Compute the partition function
    Z = np.sum(np.exp(-np.array(hamiltionian_list, dtype=np.float128) + offset), dtype=np.float128)

    return link_reliability / Z


def sortLinkLists(links, not_links, link_reliability):
    """
    Sort the lists of links and not links according to their reliability
    """
    # Links: sort by increasing reliability
    links = sorted(links, key=lambda x: link_reliability[x[0], x[1]], reverse=False)
    # Not links: sort by decreasing reliability
    not_links = sorted(not_links, key=lambda x: link_reliability[x[0], x[1]], reverse=True)

    return links, not_links
   

def singleNetworkReliability(A, A_obs, groups, offset, H=None, return_H=False):
    """
    Reliability of the network for a particular configuration
    """
    N = A.shape[0]
    
    def exp_element(A, A_obs, groups, g1, g2):
        r = maxBetween(groups, g1, g2)
        l = linksBetween(A, groups, g1, g2)
        l_obs = linksBetween(A_obs, groups, g1, g2)

        t1 = (r+1)/(2*r+1)
        t2 = binom(r, l_obs)/binom(2*r, l+l_obs)

        return np.log(t1) + np.log(t2)

    if H is None: H = hamiltonian(A_obs, groups) # Using the observed network

    sum_h = 0
    for g2 in range(N):
        for g1 in range(g2+1):
            sum_h += exp_element(A, A_obs, groups, g1, g2)
          
    n_r = np.exp(np.float128(sum_h - H + offset))

    if return_H:
        return n_r, H
    else:
        return n_r
    

def getNetworkReliability(A, A_obs, partitions_set, hamiltionian_list=None, offset=250, n_cores=-1):
    """
    Compute the network reliability
    """
    if hamiltionian_list is not None: 
        assert len(partitions_set) == len(hamiltionian_list), "partitions_set and hamiltionian_list must have the same length"

    if n_cores == -1:
       n_cores = multiprocessing.cpu_count()

    network_reliability = 0

    if hamiltionian_list is not None:
        # Parallel computation
        input = [(A, A_obs, partitions_set[k], offset, hamiltionian_list[k]) for k in range(len(partitions_set))]

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(singleNetworkReliability, input)
        
        network_reliability = np.sum(results)

        # # Local computation
        # for k in range(len(partitions_set)):
        #     network_reliability += singleNetworkReliability(A, A_obs, partitions_set[k], hamiltionian_list[k])
    
    else:
        hamiltionian_list = []
        for k in range(len(partitions_set)):
            n_r, H = singleNetworkReliability(A, A_obs, partitions_set[k], offset=offset, return_H=True)
            network_reliability += n_r
            hamiltionian_list.append(H)

    Z = np.sum(np.exp(-np.array(hamiltionian_list, dtype=np.float128) + offset), dtype=np.float128)
    
    return network_reliability / Z


def scoreMissing(A, A_obs, n_samples=1e4, delay=20, transient=500, offset=250, n_cores=-1):
    """
    Compute the AUC score for the missing interactions
    """
    # Get the lists of links and not links
    A_links, A_not_links = getLinkLists(A)
    A_obs_links, A_obs_not_links = getLinkLists(A_obs)

    # Get false negative and true negative
    missing = [link for link in A_links if link not in A_obs_links] # False negatives
    true_negatives = [not_link for not_link in A_obs_not_links if not_link in A_not_links]
    assert len(missing) > 0, "No missing interactions"
    assert len(true_negatives) > 0, "No true negatives interactions"

    # Link reliability matrix
    partitions_set, hamiltionian_list = generatePartitionsSet(A_obs, n_samples=n_samples, delay=delay,
                                                              transient=transient, 
                                                              n_cores=n_cores,return_H=True)
    R_L = computeLinkReliabilityMatrix(A_obs, partitions_set, hamiltionian_list, offset=offset)

    # Pairwise comparison for AUC
    count = 0
    for fn in missing:
        for tn in true_negatives:
            if R_L[fn[0], fn[1]] > R_L[tn[0], tn[1]]:
                count += 1
            elif R_L[fn[0], fn[1]] == R_L[tn[0], tn[1]]:
                count += 0.5

    score = count / (len(missing) * len(true_negatives))
    
    return score


def scoreSpurious(A, A_obs, n_samples=1e4, delay=20, transient=500, offset=250, n_cores=-1):
    """
    Compute the AUC score for the spurious interactions
    """
    # Get the lists of links and not links
    A_links, A_not_links = getLinkLists(A)
    A_obs_links, A_obs_not_links = getLinkLists(A_obs)

    # Get false positive and true positive
    spurious = [link for link in A_obs_links if link not in A_links] # False positives
    true_positives = [link for link in A_links if link in A_obs_links]
    assert len(spurious) > 0, "No spurious interactions"
    assert len(true_positives) > 0, "No true positives interactions"

    # Link reliability matrix
    partitions_set, hamiltionian_list = generatePartitionsSet(A_obs, n_samples=n_samples, delay=delay,
                                                              transient=transient,
                                                              n_cores=n_cores, return_H=True)
    R_L = computeLinkReliabilityMatrix(A_obs, partitions_set, hamiltionian_list, offset=offset)

    # Pairwise comparison for AUC
    count = 0
    for fp in spurious:
        for tp in true_positives:
            if R_L[fp[0], fp[1]] < R_L[tp[0], tp[1]]:
                count += 1
            elif R_L[fp[0], fp[1]] == R_L[tp[0], tp[1]]:
                count += 0.5

    score = count / (len(spurious) * len(true_positives))
    
    return score


def scanMissing(A, n_samples, delay, transient, offset):
    """
    Scan a range of p and compute the AUC for missing interactions
    """
    hyper = (n_samples, delay, transient)

    p_vec = np.arange(0.05, 1, 0.05)
    score = np.zeros((len(p_vec)))

    print("Starting missing interaction scan")
    for i in tqdm(range(len(p_vec))):
        p = p_vec[i]
        # Corrupt the adjacency matrix
        A_obs = corruptAdjacencyMatrix(A, p=p, mode='missing')
        # Score
        auc = scoreMissing(A, A_obs, *hyper, offset=offset)
        score[0] = auc

    return score


def scanSpurious(A, n_samples, delay, transient, offset):
    """
    Scan a range of p and compute the AUC for spurious interactions
    """
    hyper = (n_samples, delay, transient)

    p_vec = np.arange(0.05, 1, 0.05)
    score = np.zeros((len(p_vec)))

    print("Starting missing interaction scan")
    for i in tqdm(range(len(p_vec))):
        p = p_vec[i]
        # Corrupt the adjacency matrix
        A_obs = corruptAdjacencyMatrix(A, p=p, mode='spurious')
        # Score
        auc = scoreSpurious(A, A_obs, *hyper, offset=offset)
        score[0] = auc

    return score


# =====================================================================
#                About Network Reconstruction
# =====================================================================

def networkReconstruction(A_obs, n_samples, delay, transient, offset, verbose=False, A=None):
    """
    Reconstruct the network from the observed adjacency matrix A_obs
    """
    if verbose: 
        assert A is not None, "If verbose, the true adjacency matrix must be provided"

    # Define the hyperparameters
    hyper = (n_samples, delay, transient)

    # Initialize the current adjacency matrix
    A_cur = A_obs.copy()

    if verbose: print("Starting network reconstruction...")
    for j in range(5): # This is the maximum number of iterations
        if verbose: print("Starting iteration ", j)

        # Compute the network reliability and the link reliability matrix
        R_N = getNetworkReliability(A_cur, A_obs, *generatePartitionsSet(A_obs, *hyper), offset=offset)
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

            # Compute the network reliability with the new adjacency matrix
            R_N_temp = getNetworkReliability(A_temp, A_obs, *generatePartitionsSet(A_obs, *hyper), offset=offset)
            
            # If the network reliability increases, update the adjacency matrix
            if R_N_temp > R_N:
                if verbose: print("Network accepted \n")
                A_cur = A_temp
                R_N = R_N_temp
                miss_update = 0
                done_update = True
            else:
                miss_update += 1
                if verbose: print("Network rejected")
                if miss_update > 4:
                    break
                
        if not done_update:
            if verbose: print("No further update possible")
            break
        
    return A_cur


def studyNetworkReconstructon(G, log_path, n_samples, delay, transient, offset, verbose=True):
    """
    Study the network reconstruction for different values of p
    """ 
    A = nx.adjacency_matrix(G).todense()
    A = (A > 0).astype(int)

    p_list = [0.3, 0.1, 0.2, 0.4, 0.5]
    
    results = {}
    for p in p_list:
        if verbose: print("Starting with p = ", p)

        # Generate the observed adjacency matrix
        A_obs = corruptAdjacencyMatrix(A, p=p, mode='both')
        errors_obs = 2*len(compareAdjacencyMatrices(A, A_obs))
        
        # Reconstruct the network
        start_time = time()
        A_recon = networkReconstruction(A_obs, n_samples, delay, transient, offset, verbose, A)
        elapsed_time = time() - start_time
        errors_recon = 2*len(compareAdjacencyMatrices(A, A_recon))

        # Results
        partial_results = {
            "A_obs": A_obs,
            "A_recon": A_recon,
            "errors_obs": errors_obs,
            "errors_recon": errors_recon,
            "elapsed_time": elapsed_time
        }
        results[p] = partial_results

    # Save the results as a pickle
    with open(log_path, "wb") as file:
        pickle.dump(results, file)

    return results


def scanNetworkReliability(A, n_samples, delay, transient, offset):
    """
    Scan a range of p and compute the score log(R_N_True/R_N_Obs)
    """
    hyper = (n_samples, delay, transient)

    p_vec = np.arange(0.05, 0.55, 0.05)
    score = np.zeros((len(p_vec)))

    print("Starting network reliability scan")
    for i in tqdm(range(len(p_vec))):
        p = p_vec[i]
        # Corrupt the adjacency matrix
        A_obs = corruptAdjacencyMatrix(A, p=p, mode='both')
        # Network reliability of the observed network
        R_N_obs = getNetworkReliability(A_obs, A_obs, *generatePartitionsSet(A_obs, *hyper), offset=offset)
        # Network reliability of the true network
        R_N = getNetworkReliability(A, A_obs, *generatePartitionsSet(A_obs, *hyper), offset=offset)
        # Compute the score
        score[i] = np.log10(R_N/R_N_obs)

    return score



# =====================================================================
#                About Visualization and Plotting
# =====================================================================

def plotGraphsComparison(A, A_obs, A_tilde):
    """
    Plot the comparison between the original, observed and reconstructed graphs
    """

    ##  Helper functions ##
    def degreeRelativeError(G, G_obs, cutoff=3):
        """
        Compute the relative error in the degree
        """
        assert cutoff > 0, "Invalid cutoff"

        deg = np.array(list(dict(G.degree()).values()))
        deg_obs = np.array(list(dict(G_obs.degree()).values()))

        deg_err_obs = (deg_obs-deg)/deg # Relative error in degree
        deg_err_obs[deg_err_obs > cutoff] = cutoff
        deg_err_obs[deg_err_obs < -cutoff] = -cutoff

        return deg_err_obs


    def diffGraph(A, A_obs):
        """
        Compute the graph with only missing and spurious interactions
        """
        # Create the graph and remove all the edges
        G_diff = nx.from_numpy_array(A)
        G_diff.remove_edges_from(list(G_diff.edges))

        # Add the missing and spurious edges
        missing, spurious = compareAdjacencyMatrices(A, A_obs)
        for m in missing: G_diff.add_edge(*m, color='red')
        for s in spurious: G_diff.add_edge(*s, color='blue')
        
        return G_diff
    
    
    def plotDiffGraph(A, A1, ax):
        """
        Plot the graph with only missing and spurious interactions
        """
        # Networks
        G = nx.from_numpy_array(A)
        G1 = nx.from_numpy_array(A1)
        bet1 = np.array(list(nx.betweenness_centrality(G1).values()))
        deg_err = degreeRelativeError(G, G1, cutoff=3)

        # Node cmap and colorbar
        norm = mcolors.Normalize(vmin=-3, vmax=3)
        cmap = plt.get_cmap('RdYlBu')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Degree relative error', orientation='horizontal', fraction=0.04, pad=0.1)

        # Plot
        G_diff = diffGraph(A, A1)
        node_colors = cmap(norm(deg_err))
        edge_colors = nx.get_edge_attributes(G_diff,'color').values()
        nx.draw(G_diff, ax=ax, pos=layout, with_labels=False,
                node_color=node_colors, node_size=(300+2000*bet1), cmap=cmap,
                edge_color=edge_colors, width=1)
        
        nx.draw(G1, ax=ax, pos=layout, with_labels=False,
                node_color='none', node_size=(300+2000*bet), edgecolors='black', linewidths=1,
                edge_color='grey', width=0)

        # Custom legend
        custom_lines = [plt.Line2D([0], [0], color='red', lw=2),
                        plt.Line2D([0], [0], color='blue', lw=2),
                        plt.Line2D([0], [0], color='#E8E9EB', lw=2)]
        ax.legend(custom_lines, ['Missing', 'Spurious', 'Correct not shown'], loc='lower center',
                    bbox_to_anchor=(0.5, -0.1), ncol=3)



    ## Networks ##
    G = nx.from_numpy_array(A)
    bet = np.array(list(nx.betweenness_centrality(G).values())) # Betweenness centrality
    

    ## Plots ##
    layout = nx.spring_layout(G)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    # Original network
    ax[0].set_title("Original network")
    nx.draw(G, ax=ax[0], pos=layout, with_labels=False, 
            node_color='skyblue', node_size=(300+2000*bet), edgecolors='black', linewidths=1,
            edge_color='gray')
    
    # Observed network
    ax[1].set_title("Observed network")
    plotDiffGraph(A, A_obs, ax[1])

    # Reconstructed network
    ax[2].set_title("Reconstructed network")
    plotDiffGraph(A, A_tilde, ax[2])

    fig.tight_layout()
    fig.patch.set_facecolor('#E8E9EB')

    return fig, ax
