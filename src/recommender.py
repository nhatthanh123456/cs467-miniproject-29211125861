"""
Recommendation Engine Module
Student: Chu Van Nhat Thanh (ID: 29211125861)
Course: CS467 - Advanced Artificial Intelligence

This module implements:
1. Popularity Baseline Recommender (with min rating threshold and user filtering)
2. NLP Content-Based Recommender (Query-based and Item-to-Item / User-Profile)
3. Item-Based Collaborative Filtering (IBCF)
4. Hybrid Recommender (combining Collaborative Filtering and NLP Concept Matching)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.nlp_pipeline import StudyResourceNLPPipeline


class PopularityBaselineRecommender:
    """
    Ranks items by popularity and average rating with a minimum rating threshold,
    filtering out items already rated by the target user.
    """

    def __init__(self, min_ratings_threshold: int = 2):
        self.min_ratings_threshold = min_ratings_threshold
        self.items_df: pd.DataFrame = None
        self.ratings_df: pd.DataFrame = None
        self.item_stats: pd.DataFrame = None
        self.global_mean: float = 0.0

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "PopularityBaselineRecommender":
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.global_mean = float(self.ratings_df["rating"].mean())

        # Aggregate count and mean per item
        stats = (
            self.ratings_df.groupby("item_id")["rating"]
            .agg(count="count", avg_rating="mean")
            .reset_index()
        )

        # Merge with all items (items with 0 ratings get count=0, avg=0)
        merged = pd.merge(self.items_df[["item_id", "resource_title", "subject", "resource_type"]],
                          stats, on="item_id", how="left").fillna({"count": 0, "avg_rating": 0.0})

        # Weighted Popularity Score (Bayesian Weighted Average / IMDB formula):
        # W = (v / (v + m)) * R + (m / (v + m)) * C
        v = merged["count"]
        m = self.min_ratings_threshold
        R = merged["avg_rating"]
        C = self.global_mean

        merged["popularity_score"] = (v / (v + m)) * R + (m / (v + m)) * C
        self.item_stats = merged.sort_values(by="popularity_score", ascending=False).reset_index(drop=True)
        return self

    def recommend_for_user(
        self, user_id: str, top_k: int = 5, exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recommends top_k most popular items, filtering out items the user has already rated.
        """
        rated_items = set()
        if exclude_rated and user_id in self.ratings_df["user_id"].values:
            rated_items = set(
                self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"].tolist()
            )

        results = []
        for _, row in self.item_stats.iterrows():
            if row["item_id"] in rated_items:
                continue
            if row["count"] < self.min_ratings_threshold:
                continue
            results.append({
                "item_id": row["item_id"],
                "resource_title": row["resource_title"],
                "subject": row["subject"],
                "resource_type": row["resource_type"],
                "score": round(float(row["popularity_score"]), 4),
                "avg_rating": round(float(row["avg_rating"]), 2),
                "rating_count": int(row["count"]),
                "explanation": f"Popular in community (avg {row['avg_rating']:.2f}★ from {int(row['count'])} students)"
            })
            if len(results) >= top_k:
                break
        return results


class NLPContentBasedRecommender:
    """
    Recommends study materials based on cosine similarity over TF-IDF vectors
    for:
      (a) Free-text queries
      (b) Item-to-item similarity
      (c) User profile vectors (weighted by past ratings)
    """

    def __init__(self, nlp_pipeline: StudyResourceNLPPipeline):
        self.nlp = nlp_pipeline
        self.items_df: pd.DataFrame = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.similarity_matrix: np.ndarray = None

    def fit(self, items_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> "NLPContentBasedRecommender":
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy() if ratings_df is not None else None
        
        # Ensure NLP pipeline is fitted
        if not self.nlp.fitted:
            self.nlp.fit(self.items_df)
        
        # Precompute item-to-item cosine similarity
        self.similarity_matrix = cosine_similarity(self.nlp.tfidf_matrix, self.nlp.tfidf_matrix)
        return self

    def recommend_by_query(
        self, query: str, top_k: int = 5, exclude_item_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommends top_k resources matching a free-text search query.
        Returns similarity scores and top matching concepts.
        """
        query_vec = self.nlp.transform_query(query)
        sim_scores = cosine_similarity(query_vec, self.nlp.tfidf_matrix).flatten()
        exclude = set(exclude_item_ids or [])

        # Sort descending
        ranked_indices = np.argsort(-sim_scores)
        results = []
        for idx in ranked_indices:
            item_id = self.items_df.iloc[idx]["item_id"]
            if item_id in exclude:
                continue
            
            matching_concepts = self.nlp.extract_matching_concepts(query_vec, idx, top_k=4)
            results.append({
                "item_id": item_id,
                "resource_title": self.items_df.iloc[idx]["resource_title"],
                "subject": self.items_df.iloc[idx]["subject"],
                "resource_type": self.items_df.iloc[idx]["resource_type"],
                "score": round(float(sim_scores[idx]), 4),
                "matching_concepts": matching_concepts,
                "explanation": f"Matched concepts: {', '.join(matching_concepts) if matching_concepts else 'general context'}"
            })
            if len(results) >= top_k:
                break
        return results

    def recommend_by_item(
        self, item_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Given a study guide item_id, suggests top_k follow-up related materials.
        """
        matches = self.items_df.index[self.items_df["item_id"] == item_id].tolist()
        if not matches:
            raise KeyError(f"Item '{item_id}' not found.")
        target_idx = matches[0]
        
        sim_scores = self.similarity_matrix[target_idx]
        target_vec = self.nlp.tfidf_matrix[target_idx]
        
        ranked_indices = np.argsort(-sim_scores)
        results = []
        for idx in ranked_indices:
            if idx == target_idx:
                continue  # skip self
            
            cand_id = self.items_df.iloc[idx]["item_id"]
            matching_concepts = self.nlp.extract_matching_concepts(target_vec, idx, top_k=4)
            results.append({
                "item_id": cand_id,
                "resource_title": self.items_df.iloc[idx]["resource_title"],
                "subject": self.items_df.iloc[idx]["subject"],
                "resource_type": self.items_df.iloc[idx]["resource_type"],
                "score": round(float(sim_scores[idx]), 4),
                "matching_concepts": matching_concepts,
                "explanation": f"Related by concepts: {', '.join(matching_concepts)}"
            })
            if len(results) >= top_k:
                break
        return results

    def recommend_for_user(
        self, user_id: str, top_k: int = 5, exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Builds a user profile vector as a rating-weighted combination of item TF-IDF vectors,
        then finds the most similar unrated materials.
        """
        if self.ratings_df is None:
            raise ValueError("ratings_df required for user profile recommendations.")

        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return []

        rated_ids = set(user_ratings["item_id"].tolist())
        
        # Build user profile vector
        user_profile_vec = np.zeros((1, self.nlp.tfidf_matrix.shape[1]))
        total_weight = 0.0

        for _, r in user_ratings.iterrows():
            weight = float(r["rating"]) - 2.5  # Positive for 3,4,5; negative for 1,2
            if weight <= 0:
                continue
            try:
                item_vec = self.nlp.get_item_tfidf_vector(r["item_id"]).toarray()
                user_profile_vec += weight * item_vec
                total_weight += weight
            except KeyError:
                continue

        if total_weight > 0:
            user_profile_vec /= total_weight
        else:
            # Fallback uniform average
            for _, r in user_ratings.iterrows():
                try:
                    user_profile_vec += self.nlp.get_item_tfidf_vector(r["item_id"]).toarray()
                except KeyError:
                    continue

        sim_scores = cosine_similarity(user_profile_vec, self.nlp.tfidf_matrix).flatten()
        ranked_indices = np.argsort(-sim_scores)

        results = []
        for idx in ranked_indices:
            item_id = self.items_df.iloc[idx]["item_id"]
            if exclude_rated and item_id in rated_ids:
                continue
            
            matching_concepts = self.nlp.extract_matching_concepts(user_profile_vec, idx, top_k=4)
            results.append({
                "item_id": item_id,
                "resource_title": self.items_df.iloc[idx]["resource_title"],
                "subject": self.items_df.iloc[idx]["subject"],
                "resource_type": self.items_df.iloc[idx]["resource_type"],
                "score": round(float(sim_scores[idx]), 4),
                "matching_concepts": matching_concepts,
                "explanation": f"Matches user study interests ({', '.join(matching_concepts)})"
            })
            if len(results) >= top_k:
                break
        return results


class ItemBasedCollaborativeFiltering:
    """
    Item-Based Collaborative Filtering using Pearson / Adjusted Cosine correlation
    between item rating vectors across students.
    """

    def __init__(self, min_common_ratings: int = 1):
        self.min_common_ratings = min_common_ratings
        self.ratings_df: pd.DataFrame = None
        self.user_item_matrix: pd.DataFrame = None
        self.item_sim_df: pd.DataFrame = None
        self.user_means: pd.Series = None

    def fit(self, ratings_df: pd.DataFrame) -> "ItemBasedCollaborativeFiltering":
        self.ratings_df = ratings_df.copy()
        
        # Create user x item pivot table
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        )
        
        # User mean ratings for mean-centering
        self.user_means = self.user_item_matrix.mean(axis=1)
        centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0)

        # Compute pairwise item similarity (adjusted cosine)
        items = list(self.user_item_matrix.columns)
        n_items = len(items)
        sim_matrix = np.zeros((n_items, n_items))

        for i in range(n_items):
            sim_matrix[i, i] = 1.0
            col_i = centered_matrix[items[i]]
            for j in range(i + 1, n_items):
                col_j = centered_matrix[items[j]]
                # Common rated users
                valid_mask = col_i.notna() & col_j.notna()
                if valid_mask.sum() >= self.min_common_ratings:
                    vec_i = col_i[valid_mask].values
                    vec_j = col_j[valid_mask].values
                    denom = (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                    if denom > 1e-8:
                        sim = float(np.dot(vec_i, vec_j) / denom)
                    else:
                        sim = 0.0
                else:
                    sim = 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        self.item_sim_df = pd.DataFrame(sim_matrix, index=items, columns=items)
        return self

    def predict_rating(self, user_id: str, item_id: str) -> Optional[float]:
        """
        Predicts the rating of user_id on item_id using rated neighbor items.
        """
        if user_id not in self.user_item_matrix.index or item_id not in self.item_sim_df.columns:
            return None

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        if user_ratings.empty:
            return None

        sim_series = self.item_sim_df.loc[item_id, user_ratings.index]
        positive_sims = sim_series[sim_series > 0]
        
        if positive_sims.empty:
            return float(self.user_means[user_id])

        weighted_sum = (positive_sims * user_ratings[positive_sims.index]).sum()
        sim_sum = positive_sims.sum()
        pred = weighted_sum / sim_sum
        return float(np.clip(pred, 1.0, 5.0))


class HybridRecommender:
    """
    Project-Specific Extension:
    Blends Item-Based Collaborative Filtering usefulness predictions
    with NLP Content-Based concept matching:
        Score = alpha * Normalized_CF_Score + (1 - alpha) * NLP_Score

    Seamlessly handles Cold-Start items (zero ratings) by falling back to NLP.
    """

    def __init__(
        self,
        nlp_recommender: NLPContentBasedRecommender,
        cf_recommender: ItemBasedCollaborativeFiltering,
        alpha: float = 0.5
    ):
        self.nlp_rec = nlp_recommender
        self.cf_rec = cf_recommender
        self.alpha = alpha
        self.items_df: pd.DataFrame = None
        self.ratings_df: pd.DataFrame = None

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "HybridRecommender":
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()
        
        self.nlp_rec.fit(self.items_df, self.ratings_df)
        self.cf_rec.fit(self.ratings_df)
        return self

    def recommend_for_user(
        self, user_id: str, top_k: int = 5, exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Produces top_k hybrid recommendations combining CF score and NLP score.
        """
        rated_items = set()
        if exclude_rated and user_id in self.ratings_df["user_id"].values:
            rated_items = set(
                self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"].tolist()
            )

        # Get NLP content profile scores for all candidate items
        nlp_recs = self.nlp_rec.recommend_for_user(user_id, top_k=len(self.items_df), exclude_rated=False)
        nlp_score_map = {r["item_id"]: r["score"] for r in nlp_recs}
        nlp_concept_map = {r["item_id"]: r["matching_concepts"] for r in nlp_recs}

        hybrid_candidates = []
        for _, row in self.items_df.iterrows():
            item_id = row["item_id"]
            if exclude_rated and item_id in rated_items:
                continue

            nlp_score = nlp_score_map.get(item_id, 0.0)
            pred_rating = self.cf_rec.predict_rating(user_id, item_id)

            if pred_rating is not None:
                # Normalize predicted rating from [1, 5] to [0, 1]
                cf_score = (pred_rating - 1.0) / 4.0
                final_score = self.alpha * cf_score + (1.0 - self.alpha) * nlp_score
                mode = "CF + NLP Hybrid"
            else:
                # Cold-start fallback
                final_score = nlp_score
                mode = "NLP Content (Cold-Start fallback)"

            concepts = nlp_concept_map.get(item_id, [])
            hybrid_candidates.append({
                "item_id": item_id,
                "resource_title": row["resource_title"],
                "subject": row["subject"],
                "resource_type": row["resource_type"],
                "score": round(float(final_score), 4),
                "cf_predicted_rating": round(pred_rating, 2) if pred_rating else None,
                "nlp_content_score": round(float(nlp_score), 4),
                "matching_concepts": concepts,
                "recommendation_mode": mode,
                "explanation": f"[{mode}] Matching concepts: {', '.join(concepts[:3]) if concepts else 'N/A'}"
            })

        hybrid_candidates.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_candidates[:top_k]
