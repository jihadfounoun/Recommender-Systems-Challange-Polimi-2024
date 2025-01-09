import numpy as np
import scipy.sparse as sp


def split_URM_by_user(URM, train_percentage=0.8):
    URM = URM.tocsr()
    train_data = []
    test_data = []

    for user_id in range(URM.shape[0]):
        # Get indices of all items the user interacted with
        user_interactions = URM[user_id].indices
        if len(user_interactions) > 1:
            # Shuffle the items and split
            np.random.shuffle(user_interactions)
            split_index = int(len(user_interactions) * train_percentage)
            train_items = user_interactions[:split_index]
            test_items = user_interactions[split_index:]
        else:
            train_items = user_interactions
            test_items = []

        train_data.extend([(user_id, item) for item in train_items])
        test_data.extend([(user_id, item) for item in test_items])

    train_rows, train_cols = zip(*train_data)
    test_rows, test_cols = zip(*test_data)

    URM_train = sp.coo_matrix((np.ones(len(train_data)), (train_rows, train_cols)), shape=URM.shape)
    URM_test = sp.coo_matrix((np.ones(len(test_data)), (test_rows, test_cols)), shape=URM.shape)

    return URM_train, URM_test
