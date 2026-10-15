# Steam Game Success Prediction

Can you predict whether players will like a game **before anyone reviews it**? This project trains classifiers to predict whether a Steam game will be well received, using only information available at launch: its genres, tags, feature specs, title, and release year.

The success label comes from player reviews, but **review data is deliberately kept out of the model's inputs**. That avoids target leakage and makes the task realistic: a studio or storefront would want this kind of signal before reviews exist.

*Built for DSC 256 in the UC San Diego M.S. in Data Science program.*

## Results

| Model | Split | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| Majority-class baseline | Validation | 0.555 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression (TF-IDF + metadata) | Validation | **0.743** | 0.695 | **0.754** | **0.723** | **0.817** |
| Random forest (tuned) | Validation | 0.734 | **0.728** | 0.641 | 0.682 | 0.791 |
| Gradient boosting (tuned) | Validation | 0.715 | 0.689 | 0.655 | 0.671 | 0.805 |
| **Gradient boosting (final, refit on train + val)** | **Test** | **0.725** | **0.710** | **0.650** | **0.679** | **0.803** |

**Takeaways**

- Metadata alone carries a real signal: every model clearly beats the majority-class baseline, reaching about 0.80 ROC AUC.
- A simple logistic regression on TF-IDF features matched or beat the tuned tree ensembles on validation. With sparse, high-dimensional text features, a well-regularized linear model is a strong choice.
- Recall for successful games stays around 0.65 for the tree models, so they miss about a third of well-received titles. Error analysis suggests many misses are games with only a handful of reviews, where the label itself is noisy.

## Data

Public Steam datasets curated by Julian McAuley's group at UC San Diego:

- **`steam_games.json`**: metadata for ~32,000 games (title, genres, tags, specs, release date, price)
- **`australian_user_reviews.json`**: ~59,000 user reviews, each with a recommend / not-recommend flag

After joining the two tables, the modeling set contains **3,195 games** (1,425 labeled successful, 1,770 not).

> The raw data files are not included in this repository because of their size.

## Approach

**1. Defining "success."** The dataset has no success label, so one is built from review history. A game counts as successful if it has **at least 2 reviews** and **at least 75% of them recommend it**. This treats success as player satisfaction rather than sales.

**2. Feature engineering (launch-time information only)**

- Release year, parsed from inconsistent date strings
- Number of genres and number of tags
- A combined text field of genres, tags, specs, and title, vectorized with **TF-IDF** (unigrams and bigrams, 5,000 features)

**3. Preprocessing.** A scikit-learn `ColumnTransformer` pipeline handles median imputation and scaling for numeric features, plus TF-IDF for text, so the exact same transformations apply at train and test time.

**4. Modeling and evaluation**

- Stratified 80 / 10 / 10 train / validation / test split
- Baselines: majority class, then class-balanced logistic regression
- Tree ensembles: random forest and gradient boosting, each tuned with 3-fold `GridSearchCV` optimizing F1
- The final model is refit on train + validation and evaluated **once** on the held-out test set
- Error analysis inspects false positives and false negatives on the test set

## Tech Stack

Python · pandas · NumPy · scikit-learn (`Pipeline`, `ColumnTransformer`, `TfidfVectorizer`, `GridSearchCV`) · Matplotlib · Jupyter

## Repository Contents

| File | Description |
|---|---|
| `steam_success_prediction.ipynb` | Full pipeline: loading, EDA, feature engineering, modeling, evaluation, error analysis |
| `workbook.html` | Exported HTML version of the notebook, viewable without running anything |
| `requirements.txt` | Python dependencies |

## Limitations and Next Steps

- **Noisy labels for low-review games.** Half the games have 2 or fewer reviews, so a single review can flip the label. Raising the minimum review count would give cleaner labels at the cost of a smaller dataset.
- **Model selection.** Logistic regression had the strongest validation scores. Evaluating it on the test set alongside the ensembles would make the final comparison complete.
- **Richer features.** Developer and publisher history, price, and learned text embeddings could all add signal.
- **Regional sample.** The reviews come from Australian users, so results may not generalize to every market.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook steam_success_prediction.ipynb
```

Download `steam_games.json` and `australian_user_reviews.json` from [Julian McAuley's recommender systems datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) then unzip them (`gunzip *.json.gz`) into the project folder, or update `GAMES_PATH` and `REVIEWS_PATH` at the top of the notebook.

## Acknowledgments

Data from the Steam datasets curated by Julian McAuley and collaborators at UC San Diego, used in work including Kang & McAuley (ICDM 2018), Wan & McAuley (RecSys 2018), and Pathak, Gupta & McAuley (SIGIR 2017).
