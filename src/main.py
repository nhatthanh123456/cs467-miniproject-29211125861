"""
Main Entrypoint and CLI Demo for Study Resource Recommender
Student: Chu Van Nhat Thanh (ID: 29211125861)
Course: CS467 - Advanced Artificial Intelligence
Project: Study Resource and Exam Prep Material Recommender (MP-C15)
"""

import os
import sys

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd

from src.nlp_pipeline import StudyResourceNLPPipeline
from src.recommender import (
    PopularityBaselineRecommender,
    NLPContentBasedRecommender,
    ItemBasedCollaborativeFiltering,
    HybridRecommender
)


def print_banner():
    banner = """
================================================================================
     STUDY RESOURCE & EXAM PREP MATERIAL RECOMMENDER SYSTEM (MP-C15)
     Student: Chu Van Nhat Thanh | ID: 29211125861 | Class: CS467C
     Duy Tan University - CS467 Advanced Artificial Intelligence
================================================================================
"""
    print(banner)


def display_recommendations(recs, title="RECOMMENDED STUDY MATERIALS"):
    print(f"\n--- {title} ---")
    if not recs:
        print("  No recommendations found.")
        return

    for rank, r in enumerate(recs, 1):
        score_info = f"Score: {r['score']:.4f}"
        if "cf_predicted_rating" in r and r["cf_predicted_rating"] is not None:
            score_info += f" | CF Pred: {r['cf_predicted_rating']}★"
        
        print(f"[{rank}] {r['resource_title']} ({r['item_id']})")
        print(f"    Subject: {r['subject']} | Type: {r['resource_type']} | {score_info}")
        if "matching_concepts" in r and r["matching_concepts"]:
            print(f"    Concepts: {', '.join(r['matching_concepts'])}")
        print(f"    Rationale: {r.get('explanation', 'N/A')}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Study Resource & Exam Prep Material Recommender (MP-C15)"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "query", "user", "related", "popular"],
        default="demo",
        help="Recommendation mode to run (default: demo showing test cases)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="practice problems on probability, bayes theorem, and distributions",
        help="Free-text query for NLP recommendation",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="STU001",
        help="Student user ID for profile / hybrid recommendation (e.g. STU001 to STU015)",
    )
    parser.add_argument(
        "--item",
        type=str,
        default="RES001",
        help="Seed study resource item_id for related-material recommendation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of recommendations to return (default: 5)",
    )

    args = parser.parse_args()
    print_banner()

    print("[*] Loading datasets from data/...")
    items_df = pd.read_csv("data/items.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    print(f"    Loaded {len(items_df)} study resources and {len(ratings_df)} student ratings.\n")

    print("[*] Initializing NLP Pipeline & Recommender Engines...")
    nlp_pipe = StudyResourceNLPPipeline().fit(items_df)
    nlp_rec = NLPContentBasedRecommender(nlp_pipe).fit(items_df, ratings_df)
    ibcf = ItemBasedCollaborativeFiltering().fit(ratings_df)
    hybrid = HybridRecommender(nlp_rec, ibcf, alpha=0.5).fit(items_df, ratings_df)
    pop = PopularityBaselineRecommender().fit(items_df, ratings_df)
    print("    Engines successfully trained and ready.\n")

    if args.mode == "query":
        print(f"[*] Executing Free-Text Query: '{args.query}'")
        recs = nlp_rec.recommend_by_query(args.query, top_k=args.top_k)
        display_recommendations(recs, f"Top-{args.top_k} Results for Query: '{args.query}'")

    elif args.mode == "user":
        print(f"[*] Generating Hybrid Recommendations for Student: {args.user}")
        recs = hybrid.recommend_for_user(args.user, top_k=args.top_k, exclude_rated=True)
        display_recommendations(recs, f"Top-{args.top_k} Hybrid Recommendations for {args.user}")

    elif args.mode == "related":
        print(f"[*] Finding Materials Related to: {args.item}")
        recs = nlp_rec.recommend_by_item(args.item, top_k=args.top_k)
        display_recommendations(recs, f"Top-{args.top_k} Materials Related to {args.item}")

    elif args.mode == "popular":
        print(f"[*] Generating Popularity Baseline Recommendations for: {args.user}")
        recs = pop.recommend_for_user(args.user, top_k=args.top_k)
        display_recommendations(recs, f"Top-{args.top_k} Popular Resources for {args.user}")

    else:
        # Full Demo Mode: Demonstrating all 3 required test cases cleanly
        print(">>> DEMO MODE: Demonstrating the 3 Mandatory Course Test Cases <<<\n")

        # Test Case 1
        print("=" * 80)
        print("TEST CASE 1: Weak-Topic Query Recommendation")
        print(f"Query: '{args.query}'")
        recs1 = nlp_rec.recommend_by_query(args.query, top_k=5)
        display_recommendations(recs1, "Test Case 1: Top-5 Guides for Weak-Topic Query")

        # Test Case 2
        print("=" * 80)
        print(f"TEST CASE 2: Related-Material Test (Seed: {args.item})")
        recs2 = nlp_rec.recommend_by_item(args.item, top_k=5)
        display_recommendations(recs2, f"Test Case 2: Top-5 Follow-up Materials for {args.item}")

        # Test Case 3
        print("=" * 80)
        print("TEST CASE 3: Cold-Start Test (Newly Uploaded Cheatsheet with 0 Ratings)")
        new_resource = {
            "item_id": "RES041_NEW",
            "resource_title": "Continuous & Discrete Fourier Transform Exam Cheatsheet",
            "subject": "Applied Mathematics",
            "resource_type": "Cheatsheet",
            "summary_text": "Comprehensive exam formula sheet covering continuous Fourier transform, FFT algorithms, frequency domain filtering, and Nyquist rate sampling.",
            "concept_tags": "fourier transform; fft; signal processing; frequency filtering; sampling theorem"
        }
        print(f"New Item: [{new_resource['item_id']}] {new_resource['resource_title']} (Ratings: 0)")
        ext_items = pd.concat([items_df, pd.DataFrame([new_resource])], ignore_index=True)
        ext_nlp = StudyResourceNLPPipeline().fit(ext_items)
        ext_rec = NLPContentBasedRecommender(ext_nlp).fit(ext_items, ratings_df)
        recs3 = ext_rec.recommend_by_query("formula sheet for fourier transform frequency filtering", top_k=3)
        display_recommendations(recs3, "Test Case 3: Retrieval of 0-Rating Resource for Domain Query")

        print("=" * 80)
        print("Demo completed successfully! You can test interactive modes using:")
        print("  python src/main.py --mode query --query \"dynamic programming knapsack\"")
        print("  python src/main.py --mode user --user STU002")
        print("  python src/main.py --mode related --item RES017")


if __name__ == "__main__":
    main()
