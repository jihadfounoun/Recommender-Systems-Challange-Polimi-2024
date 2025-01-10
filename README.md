# Recommender Systems Challenge 2024 (Politecnico di Milano)

### Ranking
- **4th place on the private leaderboard**
- **2nd place on the public leaderboard**

I participated in this challenge individually, competing against 64 teams.

---

### Repository Overview
This repository contains my solution for the Kaggle competition on book recommendation systems, part of the Recommender Systems Challenge 2024. The challenge involved designing a system to recommend books based on user-item interactions and item content features.

#### Repository Structure

- **`challenge_notebooks/`** Contains Jupyter notebooks for hyperparameter tuning of individual recommenders and hybrid models.

- **`data/`** Includes the URM, ICM, and processed datasets for training.

- **`models/`** Trained model files.

- **`hypertuning_results/`** Results of hyperparameter tuning, stored in databases for analysis.

- **`src/`** Source code, including utilities and core implementations from the [Recommender Systems course repo](https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi).

- **`submissions/`** Contains Kaggle submission files.

- **`final_hybrid_submission.ipynb`**
Jupyter notebook containing the final version of the hybrid recommendation system.

---

### Competition Overview
The competition provided the following data:
- **ICM (Item Content Matrix):** 37,000 items with multiple features.
- **URM (User Rating Matrix):** 35,700 users and 1.9 million interactions.

Participants were tasked with building a high-performing recommendation system to deliver the top 10 recommendations for each user in the test set, evaluated using the **Mean Average Precision at 10** (MAP@10) metric. For more details, visit the [Kaggle competition page](https://www.kaggle.com/t/3f68d668ae1048969441243e7f3a7644).

---

### Approach

I trained a variety of models and explored different combinations to construct the hybrid recommender. Each model was optimized using the Optuna library, leveraging diverse samplers such as CMA-ES and TPE. Kaggle was utilized as a platform for hyperparameter tuning. After identifying the most effective model combination, I fine-tuned their weights to maximize MAP@10 performance.



---
### Final Hybrid

The final recommendation system is an ensemble of the following models:
- SLIM Elastic Net
- RP3beta
- RP3beta using a  stacked version of the urm
- MultVAE
- PureSVD & Scaled PureSVD

These models were combined using a linear weighted strategy, with weights optimized to achieve the best validation performance.

---

This project was developed based on the [Recommender Systems course repo](https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi).

