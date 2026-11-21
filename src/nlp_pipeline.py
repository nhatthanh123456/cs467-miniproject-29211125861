"""
NLP Pipeline Module for Study Resource and Exam Prep Material Recommender
Student: Chu Van Nhat Thanh (ID: 29211125861)
Course: CS467 - Advanced Artificial Intelligence
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Domain-tailored stop words to preserve technical academic terms while stripping generic words
DOMAIN_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "covers", "covering", "included", "including", "summary", "summarizing",
    "use", "using", "used", "need", "needs", "step", "steps", "quick"
})


def clean_text(text: str) -> str:
    """
    Cleans raw text through lowercasing, punctuation removal, stop-word removal,
    and whitespace normalization.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Replace semicolons and slashes with spaces to preserve concept boundaries
    text = re.sub(r'[;/\\_-]', ' ', text)
    
    # 3. Strip non-alphanumeric characters (keep basic letters, digits, spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. Tokenize by whitespace
    tokens = text.split()
    
    # 5. Remove stop words & single character non-digit noise
    filtered_tokens = [
        tok for tok in tokens 
        if tok not in DOMAIN_STOP_WORDS and (len(tok) > 1 or tok.isdigit())
    ]
    
    # 6. Normalize whitespace
    return " ".join(filtered_tokens)


class StudyResourceNLPPipeline:
    """
    NLP Pipeline for processing study resource text summaries and concept tags,
    computing unigram + bigram TF-IDF representations, and extracting explanatory concepts.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 1):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True
        )
        self.fitted = False
        self.items_df: pd.DataFrame = None
        self.corpus: List[str] = []
        self.tfidf_matrix: np.ndarray = None
        self.feature_names: np.ndarray = None

    def compose_item_document(self, row: Union[pd.Series, Dict[str, Any]]) -> str:
        """
        Combines resource title, subject, resource type, summary text,
        and concept tags into a single text representation with weighted emphasis.
        """
        title = str(row.get("resource_title", ""))
        subject = str(row.get("subject", ""))
        rtype = str(row.get("resource_type", ""))
        summary = str(row.get("summary_text", ""))
        tags = str(row.get("concept_tags", "")).replace(";", " ")
        
        # Combine with repetition for high-signal fields (tags & title)
        combined = f"{title} {title} {subject} {rtype} {summary} {tags} {tags}"
        return clean_text(combined)

    def fit(self, items_df: pd.DataFrame) -> "StudyResourceNLPPipeline":
        """
        Fits the TF-IDF vectorizer on the items dataset.
        """
        self.items_df = items_df.copy().reset_index(drop=True)
        self.corpus = [
            self.compose_item_document(row) for _, row in self.items_df.iterrows()
        ]
        
        # Fit vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.fitted = True
        return self

    def transform_query(self, query: str) -> np.ndarray:
        """
        Helper function that transforms any new free-text query into the
        same vocabulary and TF-IDF space.
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transforming queries.")
        
        cleaned = clean_text(query)
        # Fallback if cleaning stripped everything
        if not cleaned:
            cleaned = query.lower()
        return self.vectorizer.transform([cleaned])

    def get_item_tfidf_vector(self, item_id: str) -> np.ndarray:
        """
        Retrieves the precomputed TF-IDF vector for a specific item_id.
        """
        if not self.fitted:
            raise ValueError("Pipeline is not fitted.")
        matches = self.items_df.index[self.items_df["item_id"] == item_id].tolist()
        if not matches:
            raise KeyError(f"Item '{item_id}' not found in corpus.")
        idx = matches[0]
        return self.tfidf_matrix[idx]

    def extract_matching_concepts(
        self, query_or_profile_vector: np.ndarray, item_index: int, top_k: int = 5
    ) -> List[str]:
        """
        Extracts matching study concepts/keywords between a query vector
        and a specific item to justify why the guide was recommended.
        """
        if not self.fitted:
            raise ValueError("Pipeline is not fitted.")
        
        # Convert query vector to dense 1D array
        if hasattr(query_or_profile_vector, "toarray"):
            q_vec = query_or_profile_vector.toarray().flatten()
        else:
            q_vec = np.asarray(query_or_profile_vector).flatten()
            
        item_vec = self.tfidf_matrix[item_index].toarray().flatten()
        
        # Element-wise product identifies shared, high-importance terms
        interaction = q_vec * item_vec
        nz_indices = np.where(interaction > 0)[0]
        
        if len(nz_indices) == 0:
            # Fallback to top terms of the item itself
            item_nz = np.where(item_vec > 0)[0]
            if len(item_nz) == 0:
                return []
            sorted_item_terms = item_nz[np.argsort(-item_vec[item_nz])][:top_k]
            return [self.feature_names[i] for i in sorted_item_terms]
        
        sorted_indices = nz_indices[np.argsort(-interaction[nz_indices])][:top_k]
        return [self.feature_names[i] for i in sorted_indices]

    def get_top_keywords_for_item(self, item_index: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns top TF-IDF terms with scores for a given item index.
        """
        item_vec = self.tfidf_matrix[item_index].toarray().flatten()
        nz = np.where(item_vec > 0)[0]
        if len(nz) == 0:
            return []
        sorted_nz = nz[np.argsort(-item_vec[nz])][:top_k]
        return [(self.feature_names[i], round(float(item_vec[i]), 4)) for i in sorted_nz]
