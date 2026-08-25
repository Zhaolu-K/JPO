import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
import joblib
from typing import List, Dict
import jieba
import json
from multiprocessing import Pool

def cut_text(text):
    return ' '.join(jieba.lcut(text))

class LegalNaiveBayes:
    """
    Legal Naive Bayes Model - Used to compute P(Article|Fact)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_features: int = 20000,
        ngram_max: int = 3,
        min_df: int = 2,
        max_df: float = 0.8
    ):
        """
        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha

        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_df=max_df
        )

        self.classifier = ComplementNB(alpha=alpha)

        # Save vocabulary
        self.vocabulary_ = None

        # Save conditional probability P(word|article) - Used for R_consistency
        self.feature_log_prob_ = None  # shape: [n_classes, n_features]
        self.class_log_prior_ = None  # shape: [n_classes]
        self.classes_ = None  # List of article numbers

    def train(self, facts: List[str], labels: List[str]):
        """
        Train the Naive Bayes model

        Args:
            facts: List of case fact texts
            labels: Corresponding list of article numbers
        """
        print(f"Training data scale: {len(facts)} cases, {len(set(labels))} article categories")

        # 1. Feature extraction
        with Pool(processes=8) as pool:
            facts = pool.map(cut_text, facts)

        X = self.vectorizer.fit_transform(facts)
        self.vocabulary_ = self.vectorizer.get_feature_names_out()

        print(f"Feature dimension: {X.shape[1]} feature words")

        # 2. Train classifier
        self.classifier.fit(X, labels)

        # 3. Save model parameters for R_consistency calculation
        self.feature_log_prob_ = self.classifier.feature_log_prob_  # Conditional probability log
        self.class_log_prior_ = self.classifier.class_log_prior_  # Class prior
        self.classes_ = self.classifier.classes_  # Class labels

        # 4. Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nClass distribution statistics:")
        for cls, count in zip(unique[:10], counts[:10]):  # Only show first 10
            print(f"  Article {cls}: {count} cases ({count / len(facts) * 100:.2f}%)")

    def _predict_proba(self, fact: str) -> Dict[str, float]:
        """
        Predict the probability distribution of articles for a single case

        Returns:
            Dictionary: {article number: probability}
        """
        fact = ' '.join(jieba.lcut(fact))
        X = self.vectorizer.transform([fact])
        proba = self.classifier.predict_proba(X)[0]

        return {cls: prob for cls, prob in zip(self.classes_, proba)}

    def compute_P_A_given_F(self, fact: str, articles: List[str]) -> float:
        """
        Compute P(A|F) - Used for R_consistency

        Args:
            fact: Case fact text
            articles: List of articles predicted by the model

        Returns:
            P(A|F) probability value
        """
        # Get probabilities for all articles
        proba_dict = self._predict_proba(fact)

        # Calculate the average probability of the predicted article set
        total_prob = 0.0
        cnt = 0
        for article in articles:
            if article in proba_dict:
                total_prob += proba_dict[article]
                cnt += 1

        # Normalize
        return total_prob / cnt if cnt > 0 else 0.0

    def save_model(self, path: str):
        """Save model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'feature_log_prob_': self.feature_log_prob_,
            'class_log_prior_': self.class_log_prior_,
            'classes_': self.classes_,
            'vocabulary_': self.vocabulary_
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model"""
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.feature_log_prob_ = model_data['feature_log_prob_']
        self.class_log_prior_ = model_data['class_log_prior_']
        self.classes_ = model_data['classes_']
        self.vocabulary_ = model_data['vocabulary_']
        print(f"Model loaded from {path}")

if __name__ == '__main__':
    legal_naive_bayes = LegalNaiveBayes()

    facts = []
    labels = []

    source_data = []
    with open(f'../data/jpo_dataset/train_rl.json', 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data = json.loads(line)
                source_data.append(data)

    print(len(source_data))

    for item in source_data:
        fact = item['fact']
        articles = item['meta']['relevant_articles']
        for article in articles:
            facts.append(fact)
            labels.append(article)

    legal_naive_bayes.train(facts, labels)

    legal_naive_bayes.save_model(f'legal_naive_bayes.joblib')