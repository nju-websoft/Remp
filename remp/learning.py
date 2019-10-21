import numpy as np
import networkx as nx
import pandas as pd


class RempMatcher(object):
    def __init__(self, dataset, edgelist, prior, tau=0.9):
        edgelist['source'] = edgelist['s1'] + '\t' + edgelist['s2']
        edgelist['target'] = edgelist['o1'] + '\t' + edgelist['o2']
        edgelist['label'] = 'r'
        edgelist = edgelist[(edgelist['forward'] >= tau) | (edgelist['backward'] >= tau)]
        w1 = edgelist[['source', 'target', 'forward']].rename(columns={'forward': 'weight'})
        w2 = edgelist[['source', 'target', 'backward']].rename(
            columns={'backward': 'weight', 'source': 'target', 'target': 'temp'}).rename(columns={'temp': 'source'})
        wg = pd.concat([w1, w2], sort=True).groupby(by=['source', 'target'])['weight'].max().reset_index()
        wg = wg.loc[wg['weight'] >= tau]
        wg['weight'] = -np.log(wg['weight'])
        er_graph = nx.from_pandas_edgelist(wg, edge_attr=['weight'], create_using=nx.DiGraph())
        inffered_set = nx.all_pairs_dijkstra_path_length(er_graph, cutoff=-np.log(tau), weight='weight')
        lsv = prior
        lsv['s'] = prior['s1'] + '\t' + prior['s2']
        self.prob = lsv[['s', 'p']].set_index('s')
        self.inffered_set = dict((k, set(v.keys())) for k, v in inffered_set)
        self.all_resolved = set()
        self.all_resolved_match = set()
        self.all_propagated_match = set()
        self.unresolved = set(er_graph.node)

    def next_questions(self, mu=10):
        problems = []
        new_unresolved = self.unresolved | set()
        total_best_value = []

        for j in range(0, mu):
            best_value = 0.0
            bset_problem = None
            for problem in new_unresolved:
                resolved = self.inffered_set[problem]# & unresolved
                if len(resolved) * self.prob.loc[problem].values[0] > best_value:
                    best_value = len(resolved) * self.prob.loc[problem].values[0]
                    best_problem = problem
            new_unresolved -= {best_problem}
            problems.append(best_problem)
        return problems
    
    def update_model(self, questions, probs):
        for i in range(0, len(questions)):
            problem = questions[i]
            answer = probs[i]
            if answer >= 0.8:
                self.all_propagated_match |= self.inffered_set[problem]
                self.all_resolved_match = self.all_propagated_match
                self.all_resolved |= self.inffered_set[problem]
                self.unresolved -= self.inffered_set[problem]
                for pp in self.unresolved:
                    self.inffered_set[pp] -= self.inffered_set[problem]
            elif answer <= 0.2:
                self.all_resolved.add(problem)
                self.unresolved.remove(problem)
                for pp in self.unresolved:
                    self.inffered_set[pp] -= {problem}
            else:
                self.prob.loc[problem, 'p'] = answer
