import torch
from mdp import MDP

def calculate_C_matrix(TPM, gamma):
    """
    Calculate the controllability matrix C(s,s') for a given transition probability matrix (TPM) and discount factor (gamma).
    
    Element c_ij is sup_\pi M_{ij}^{(\pi)}

    We assume that this supremum is achieved for all "i" for any given "j"
    
    """
    C = torch.zeros((TPM.shape[0], TPM.shape[0]), dtype=torch.float32)

    # Calculate the controllability matrix C(s,s') for each state
    for i in range(TPM.shape[0]):
        rewards = torch.zeros(TPM.shape[0])
        rewards[i] = 1.0
        mdp = MDP(TPM, rewards, gamma=gamma)
        mdp.get_SR_M()
        M = mdp.M
        M *= (1 - gamma) # normalise the successor representation
        C[:, i] = M[:, i]

    return C




