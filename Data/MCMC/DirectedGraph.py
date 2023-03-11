import pandas as pd
import numpy as np
import random

class DirectedGraph:
    def __init__(self):
        self.p = 0 # Teleportation constant
        self.states = set()
        self.adjacency = dict()

    def is_connected(self):
        if len(self.states) == 0:
            return False
        reachable_states = set()
        new_states = {list(self.states)[0]}
        while len(new_states) != 0:
            new_states = {state for state in self.states if any([self.adjacency[state_][state] != 0 for state_ in new_states]) and state not in reachable_states}
            reachable_states = reachable_states.union(new_states)
        return reachable_states == self.states

    def transition(self, state):
        if state not in self.states:
            raise KeyError("Invalid State. Try add_state first.")
        if not any(self.adjacency[state].values()) and self.p == 0:
            raise ValueError("No transitions on this state and teleportation constant is 0. Change the teleportation constant or add more links.")
        sd = self.adjacency[state]
        n = len(sd)
        return random.choices(list(sd.keys()), weights = (1-self.p)*np.array(list(sd.values()))+self.p*np.ones(n)/n)[0]

    def get_df(self):
        return pd.DataFrame(self.adjacency).fillna(0)

    def transition_matrix(self):
        df = self.get_df()
        mat = np.array(df,dtype=float)
        mat = np.array([[mat[row][i]/sum(mat[row]) if sum(mat[row]) != 0 else float(i == row) for i in range(len(mat))] for row in range(len(mat))])
        mat *= 1-self.p
        mat += self.p*np.ones_like(mat)/len(mat)
        return mat

    def stationary(self):
        colnames = list(self.get_df().head())
        eigen = np.linalg.eig(self.transition_matrix().transpose())
        index = np.where(np.round(eigen[0],decimals = 6) == 1)[0]
        if index.size == 0:
            print(eigen)
            raise RuntimeError("There does not exist a stationary distribution.")
        eigenvec = eigen[1].transpose()[index[0]]
        eigenvec /= sum(eigenvec)
        return {colnames[k]:np.real(eigenvec[k]) for k in range(len(colnames))}

    def set_telep(self,p):
        if (p < 0) or (p > 1):
            raise ValueError("The teleportation constant must be between 0 and 1.")
        self.p = p
        return self

    def addState(self,key):
        self.states.add(key)
        self.adjacency[key] = {state : 0 for state in self.states}
        for state in self.states:
            self.adjacency[state][key] = 0
        return self

    def incLink(self, s1, s2, increment = 1, add = False):
        if add:
            for state in {s1,s2}.difference(self.states):
                self.addState(state)
        if (not add) and (not {s1,s2}.issubset(self.states)):
            raise KeyError("{diff} not instantiated as a state. Use DirectedGraph.addState(key) or set add = True.".format(diff = {s1,s2}.difference(self.states)))
        if (self.adjacency.get(s1)):
            if (self.adjacency.get(s1).get(s2) != None):
                self.adjacency[s1][s2] += increment
            else:
                self.adjacency[s1][s2] = increment
        else:
            self.adjacency[s1] = dict()
            self.adjacency[s1][s2] = increment
        return self
    
    def addLink(self, s1, s2, weight = 1, add = False):
        self.incLink(s1,s2,increment=0,add=add)
        self.incLink(s1,s2,increment=weight)
        return self

    def print(self):
        print(pd.DataFrame(self.adjacency).fillna(0).transpose())
