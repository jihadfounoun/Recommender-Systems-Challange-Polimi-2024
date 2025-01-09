import numpy as np
import scipy.sparse as sps

from src.Recommenders.BaseRecommender import BaseRecommender


from typing import List

class LinearWeightedRecommender(BaseRecommender):
    def __init__(self, URM_train, recommenders: List[BaseRecommender], weights: List[float]):
        super(LinearWeightedRecommender, self).__init__(URM_train)
        self.recommenders = recommenders
        self.weights = weights

    def _calculate_top_pop_items(self, cutoff):
        item_popularity = np.ediff1d(sps.csc_matrix(self.URM_train).indptr)
        self.filterTopPop = True
        self.filterTopPop_ItemsID = item_popularity.argsort()[-cutoff:]

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = np.zeros((len(user_id_array), self.n_items))
        for recommender, w in zip(self.recommenders, self.weights):
            scores += recommender._compute_item_score(user_id_array, items_to_compute) * w
        return scores