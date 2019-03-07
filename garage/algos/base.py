class Algorithm:
    pass


class RLAlgorithm(Algorithm):
    def train_once(self, itr, paths):
        raise NotImplementedError
