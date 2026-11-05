import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from src.models.baseline import build_model

def train_baseline(train):
    texts = train["text"]
    labels = train["model"]
    groups = train["prompt_id"]

    folds = GroupKFold(n_splits=5)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(folds.split(texts, labels, groups), start=1):
        print(f"\nTraining fold {fold}/5...")

        model = build_model()
        model.fit(texts.iloc[train_idx], labels.iloc[train_idx])

        predictions = model.predict(texts.iloc[val_idx])
        score = balanced_accuracy_score(labels.iloc[val_idx], predictions)
        scores.append(score)
        print(f"Balanced Accuracy: {score:.4f}")

    print(f"\n Mean score: {np.mean(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")

    return scores