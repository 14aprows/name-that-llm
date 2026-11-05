from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

def build_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            lowercase=False,
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            lowercase=False,
        )),
    ])

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
        )),
    ])