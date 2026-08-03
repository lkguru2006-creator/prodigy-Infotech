"""
Synthetic data generator for the Mall Customers dataset.

Mirrors the exact schema of the real Kaggle dataset
(vjchoudhary7/customer-segmentation-tutorial-in-python):
    CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

This exists purely so the pipeline is runnable end-to-end without network
access to Kaggle. Generated data is built around 5 latent customer
archetypes (blobs in Age / Income / Spending Score space) so that a
5-cluster K-means run on synthetic data produces sensible, well-separated
clusters -- a reasonable stand-in for the real data's known structure.

To use real data: download Mall_Customers.csv from the Kaggle link and
place it at data/raw/Mall_Customers.csv. The pipeline will use it directly
and this generator will never be invoked.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.exceptions import DataValidationError


def generate_synthetic_customers(
    n_samples: int = 200,
    n_blobs: int = 5,
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic customer dataset matching the Mall Customers schema.

    Archetypes loosely modeled on the well-known structure of the real
    dataset: budget shoppers, high-income low spenders, average customers,
    high-income high spenders, and young high spenders on modest income.

    Args:
        n_samples: total number of synthetic customers to generate.
        n_blobs: number of latent customer archetypes (should match the
            downstream n_clusters for a clean demonstration).
        random_state: seed for reproducibility.
        logger: optional logger for structured progress messages.

    Returns:
        DataFrame with columns: CustomerID, Gender, Age,
        Annual Income (k$), Spending Score (1-100)

    Raises:
        DataValidationError: if generation parameters are invalid.
    """
    if n_samples < n_blobs:
        raise DataValidationError(
            f"n_samples ({n_samples}) must be >= n_blobs ({n_blobs})."
        )

    rng = np.random.default_rng(random_state)

    # Archetype centers: (age_mean, age_std, income_mean, income_std, spend_mean, spend_std)
    archetypes = [
        (45, 10, 25, 8, 20, 12),   # budget-conscious older shoppers
        (32, 8, 85, 12, 18, 10),   # high income, low spend (savers)
        (40, 12, 55, 10, 50, 12),  # average income, average spend
        (35, 7, 90, 10, 82, 10),   # high income, high spend (premium)
        (24, 5, 40, 8, 78, 10),    # young, modest income, high spend
    ]

    if n_blobs != len(archetypes):
        # Cycle through archetypes if a different blob count is requested
        archetypes = [archetypes[i % len(archetypes)] for i in range(n_blobs)]

    base_count = n_samples // n_blobs
    remainder = n_samples % n_blobs
    counts = [base_count + (1 if i < remainder else 0) for i in range(n_blobs)]

    ages, incomes, spends, genders = [], [], [], []

    for (age_mu, age_sd, inc_mu, inc_sd, sp_mu, sp_sd), count in zip(archetypes, counts):
        ages.append(rng.normal(age_mu, age_sd, count))
        incomes.append(rng.normal(inc_mu, inc_sd, count))
        spends.append(rng.normal(sp_mu, sp_sd, count))
        genders.extend(rng.choice(["Male", "Female"], size=count).tolist())

    ages = np.concatenate(ages)
    incomes = np.concatenate(incomes)
    spends = np.concatenate(spends)

    # Clip to realistic bounds (positional args -- keyword `.clip(min=...)`
    # is deprecated/unsupported on newer numpy versions)
    ages = np.clip(ages, 18, 70).round().astype(int)
    incomes = np.clip(incomes, 15, 140).round().astype(int)
    spends = np.clip(spends, 1, 100).round().astype(int)

    # Shuffle so archetypes aren't grouped in CustomerID order
    shuffle_idx = rng.permutation(n_samples)

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_samples + 1),
            "Gender": np.array(genders)[shuffle_idx],
            "Age": ages[shuffle_idx],
            "Annual Income (k$)": incomes[shuffle_idx],
            "Spending Score (1-100)": spends[shuffle_idx],
        }
    )

    if logger:
        logger.info(
            "Generated synthetic dataset: %d customers across %d archetypes (seed=%d)",
            n_samples,
            n_blobs,
            random_state,
        )

    return df
