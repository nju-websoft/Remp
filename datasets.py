class PickleDataset(object):
    def __init__(self, name, base_dir):
        import pandas as pd
        super().__init__()
        self.label = pd.read_pickle(base_dir + '/label.pkl')
        self.relations_1 = pd.read_pickle(base_dir + '/relations_1.pkl')
        self.attributes_1 = pd.read_pickle(base_dir + '/attributes_1.pkl')
        self.relations_2 = pd.read_pickle(base_dir + '/relations_2.pkl')
        self.attributes_2 = pd.read_pickle(base_dir + '/attributes_2.pkl')
        self.link = pd.read_pickle(base_dir + '/link.pkl')