from kuhn import KuhnState, Pass, Bet, cards, num_actions
import numpy as np
from itertools import permutations

class InformationSet:
    """
    has regret and strat for each information set
    b/c each info set can have 2 actions
    """

    def __init__(self):
        #regret for each action
        self.cumulative_regret = np.zeros(num_actions, dtype = np.float64)

        #sum of all strats (avg used for final output)
        self.strat_sum = np.zeros(num_actions, dtype = np.float64)

    def get_strategy(self):
        """
        want to match regret, strat is proportional to postive regret, 
        if all regret is negative, use uniform 
        """
        postive_regret = np.maximum(self.cumulative_regret, 0)
        total = postive_regret.sum()
        if total > 0:
            return postive_regret / total
        else:   
            return np.ones(num_actions) / num_actions
    
    def get_average_strat(self):
        #average strat across iterations

        total = self.strat_sum.sum()
        if total > 0:
            return self.strat_sum / total
        else:   
            return np.ones(num_actions) / num_actions
