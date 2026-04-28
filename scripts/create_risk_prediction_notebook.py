#!/usr/bin/env python3
"""Create the malaria AI/ML risk prediction notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Malaria_AIML_Risk_Model.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # AI/ML Malaria Risk Prediction Model

        This notebook builds a supervised machine learning model using the malaria dataset prepared from the PDFs.

        ## Main Goal

        Predict whether a state/UT will be **high malaria risk in the next year** using only information available before that year.

        ## What This Model Can Do

        - Learn from past malaria burden and case-rate patterns.
        - Compare multiple AI/ML models against a baseline model.
        - Evaluate predictions on later years that were not used for training.
        - Estimate next-year risk probability for each state/UT.
        - Highlight the project regions: Odisha, Mizoram, and Tripura.

        ## Important Limitation

        This is a risk prediction model, not a causal model. It does not directly know rainfall, temperature, mosquito density, interventions, sanitation, or migration. It predicts from historical malaria patterns already present in the dataset.
        """
    ),
    md(
        """
        ## 0. Setup

        Run the install cell only if the notebook kernel is missing packages.
        """
    ),
    code(
        """
        # Uncomment and run this cell if packages are missing.
        # %pip install pandas numpy matplotlib scikit-learn
        """
    ),
    md(
        """
        **Why this matters:** The model needs common data-science libraries for loading CSV files, plotting trends, training classifiers, and calculating evaluation metrics. Keeping the install command separate avoids reinstalling packages every time the notebook runs.
        """
    ),
    code(
        """
        from pathlib import Path
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from sklearn.base import clone
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            RocCurveDisplay,
            average_precision_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        warnings.filterwarnings("ignore")
        plt.style.use("seaborn-v0_8-whitegrid")

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        elif not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent

        ANALYSIS_READY_DIR = PROJECT_ROOT / "data" / "analysis_ready"
        STATE_YEAR_CSV = ANALYSIS_READY_DIR / "state_year_malaria_clean.csv"
        AIML_FEATURES_CSV = ANALYSIS_READY_DIR / "aiml_state_year_features.csv"

        SELECTED_REGIONS = ["Odisha", "Mizoram", "Tripura"]
        RANDOM_STATE = 42

        HIGHLIGHT_COLORS = {
            "Odisha": "#1f77b4",
            "Mizoram": "#ff7f0e",
            "Tripura": "#2ca02c",
        }

        def fmt_probability(value):
            return f"{value:.1%}"

        def risk_band(probability):
            if probability >= 0.70:
                return "High"
            if probability >= 0.40:
                return "Moderate"
            return "Low"
        """
    ),
    md(
        """
        **Why this matters:** This cell defines the project paths, selected regions, plotting style, and helper functions in one place. That keeps every later table and graph consistent and makes the notebook easier to rerun on another computer.
        """
    ),
    md(
        """
        ## 1. Load the Analysis-Ready Data

        We use two prepared CSV files:

        - `state_year_malaria_clean.csv`: state-level yearly malaria burden.
        - `aiml_state_year_features.csv`: lagged features and next-year outcomes for machine learning.
        """
    ),
    code(
        """
        state_year = pd.read_csv(STATE_YEAR_CSV)
        model_df = pd.read_csv(AIML_FEATURES_CSV)

        state_numeric_cols = [
            "year",
            "total_cases",
            "total_deaths",
            "district_count",
            "population_2011",
            "cases_per_100k",
            "deaths_per_100k",
        ]
        for col in state_numeric_cols:
            state_year[col] = pd.to_numeric(state_year[col], errors="coerce")

        model_numeric_cols = [col for col in model_df.columns if col not in {"state", "selected_region"}]
        for col in model_numeric_cols:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        display(state_year.head())
        display(model_df.head())

        print("State-year rows:", len(state_year))
        print("Feature rows:", len(model_df))
        print("States/UTs:", state_year["state"].nunique())
        print("State-year range:", int(state_year["year"].min()), "-", int(state_year["year"].max()))
        print("Feature-year range:", int(model_df["year"].min()), "-", int(model_df["year"].max()))
        """
    ),
    md(
        """
        **Why this matters:** The model depends on clean numeric columns. This cell confirms that the files load correctly and shows the exact number of rows, states, and years available before we train anything.
        """
    ),
    md(
        """
        ## 2. Inspect the Project Regions

        The AI/ML model is trained on all available states/UTs, but Odisha, Mizoram, and Tripura are kept visible because they are the main project regions.
        """
    ),
    code(
        """
        selected = state_year[state_year["state"].isin(SELECTED_REGIONS)].copy()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for region in SELECTED_REGIONS:
            region_df = selected[selected["state"] == region]
            axes[0].plot(
                region_df["year"],
                region_df["total_cases"],
                marker="o",
                linewidth=2,
                label=region,
                color=HIGHLIGHT_COLORS[region],
            )
            axes[1].plot(
                region_df["year"],
                region_df["cases_per_100k"],
                marker="o",
                linewidth=2,
                label=region,
                color=HIGHLIGHT_COLORS[region],
            )

        axes[0].set_title("Annual malaria cases")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Cases")
        axes[0].legend()

        axes[1].set_title("Annual malaria case rate")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Cases per 100,000")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        display(selected.sort_values(["state", "year"]).tail(12))
        """
    ),
    md(
        """
        **Why this matters:** These plots show whether the selected regions have similar or contrasting epidemic histories. That matters because an ML model trained across all states must still make sensible predictions for the specific regions used in the report.
        """
    ),
    md(
        """
        ## 3. Define the Prediction Target

        We predict:

        > Will this state/UT be high malaria risk next year?

        A state/UT is labelled **high risk** if its next-year malaria case rate is in the top 25% of the training-period observations.
        """
    ),
    code(
        """
        feature_cols = [
            "year",
            "population_2011",
            "district_count",
            "total_cases_lag1",
            "total_cases_lag2",
            "total_cases_lag3",
            "total_deaths_lag1",
            "cases_per_100k_lag1",
            "cases_per_100k_lag2",
            "cases_per_100k_lag3",
            "deaths_per_100k_lag1",
            "cases_rate_roll3",
            "cases_yoy_change",
            "rate_yoy_change",
        ]

        model_df = model_df.dropna(subset=feature_cols + ["next_year_rate"]).copy()

        train_period_for_threshold = model_df[model_df["year"] <= 2016]
        risk_threshold = train_period_for_threshold["next_year_rate"].quantile(0.75)

        model_df["risk_target"] = (model_df["next_year_rate"] >= risk_threshold).astype(int)
        model_df["prediction_year"] = model_df["year"] + 1

        print("High-risk threshold:", round(risk_threshold, 2), "cases per 100,000")
        print("High-risk share:", round(model_df["risk_target"].mean(), 3))

        display(
            model_df[
                [
                    "state",
                    "year",
                    "prediction_year",
                    "cases_per_100k",
                    "next_year_rate",
                    "risk_target",
                ]
            ].head(10)
        )
        """
    ),
    md(
        """
        **Why this matters:** The model needs a clear outcome variable. Using a training-period threshold avoids defining “high risk” with information from the future test period, which would make the evaluation too optimistic.
        """
    ),
    md(
        """
        ## 4. Chronological Train, Validation, and Test Split

        We split by time:

        - Training: feature years up to 2016
        - Validation: feature years 2017 to 2019
        - Test: feature years 2020 onward

        Each feature year predicts the following year.
        """
    ),
    code(
        """
        train = model_df[model_df["year"] <= 2016].copy()
        validation = model_df[(model_df["year"] >= 2017) & (model_df["year"] <= 2019)].copy()
        test = model_df[model_df["year"] >= 2020].copy()

        X_train = train[feature_cols]
        y_train = train["risk_target"]

        X_val = validation[feature_cols]
        y_val = validation["risk_target"]

        X_test = test[feature_cols]
        y_test = test["risk_target"]

        split_summary = pd.DataFrame(
            [
                {"split": "train", "rows": len(train), "year_min": train["year"].min(), "year_max": train["year"].max(), "high_risk_rate": y_train.mean()},
                {"split": "validation", "rows": len(validation), "year_min": validation["year"].min(), "year_max": validation["year"].max(), "high_risk_rate": y_val.mean()},
                {"split": "test", "rows": len(test), "year_min": test["year"].min(), "year_max": test["year"].max(), "high_risk_rate": y_test.mean()},
            ]
        )

        display(split_summary)
        """
    ),
    md(
        """
        **Why this matters:** Epidemic prediction is a forecasting task, so time order matters. A chronological split tests whether the model can generalize to later years instead of memorizing random rows from the same time period.
        """
    ),
    md(
        """
        ## 5. Train Candidate AI/ML Models

        We compare simple and stronger classifiers:

        - Dummy baseline
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        """
    ),
    code(
        """
        candidate_models = {
            "Dummy baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                min_samples_leaf=3,
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        }

        validation_results = []
        fitted_candidates = {}

        for name, model in candidate_models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            fitted_candidates[name] = fitted

            val_pred = fitted.predict(X_val)
            val_prob = fitted.predict_proba(X_val)[:, 1] if hasattr(fitted, "predict_proba") else val_pred

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_pred, average="binary", zero_division=0
            )

            validation_results.append(
                {
                    "model": name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
                    "roc_auc": roc_auc_score(y_val, val_prob) if len(set(y_val)) > 1 else np.nan,
                    "average_precision": average_precision_score(y_val, val_prob) if len(set(y_val)) > 1 else np.nan,
                }
            )

        validation_results_df = (
            pd.DataFrame(validation_results)
            .sort_values(["f1", "average_precision", "balanced_accuracy"], ascending=False)
            .reset_index(drop=True)
        )

        display(validation_results_df)
        """
    ),
    md(
        """
        **Why this matters:** The dummy model shows what happens if we do almost no learning. The real ML models must beat that baseline; otherwise the features are not adding useful predictive signal.
        """
    ),
    md(
        """
        ## 6. Select the Best Model and Evaluate on the Test Period

        The best model is selected on validation performance, then retrained on training plus validation data and evaluated once on the held-out test period.
        """
    ),
    code(
        """
        best_model_name = validation_results_df.iloc[0]["model"]
        best_model_template = candidate_models[best_model_name]

        train_val = pd.concat([train, validation], ignore_index=True)
        X_train_val = train_val[feature_cols]
        y_train_val = train_val["risk_target"]

        best_model = clone(best_model_template)
        best_model.fit(X_train_val, y_train_val)

        test_pred = best_model.predict(X_test)
        test_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else test_pred

        print("Selected model:", best_model_name)
        print()
        print(classification_report(y_test, test_pred, target_names=["not high risk", "high risk"], zero_division=0))

        test_metrics = {
            "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
            "roc_auc": roc_auc_score(y_test, test_prob) if len(set(y_test)) > 1 else np.nan,
            "average_precision": average_precision_score(y_test, test_prob) if len(set(y_test)) > 1 else np.nan,
        }
        display(pd.DataFrame([test_metrics]))

        ConfusionMatrixDisplay(
            confusion_matrix(y_test, test_pred),
            display_labels=["not high risk", "high risk"],
        ).plot(cmap="Blues")
        plt.title(f"Test Confusion Matrix: {best_model_name}")
        plt.show()

        if len(set(y_test)) > 1:
            RocCurveDisplay.from_predictions(y_test, test_prob)
            plt.title(f"Test ROC Curve: {best_model_name}")
            plt.show()
        """
    ),
    md(
        """
        **Why this matters:** The test period is the honest check. It tells us how well the chosen model performs on later years that were not used for model selection.
        """
    ),
    md(
        """
        ## 7. Understand Which Features Drive Prediction

        Feature importance helps explain what the risk model is using.
        """
    ),
    code(
        """
        if best_model_name in {"Random Forest", "Gradient Boosting"}:
            importance_values = best_model.feature_importances_
        elif best_model_name == "Logistic Regression":
            importance_values = np.abs(best_model.named_steps["model"].coef_[0])
        else:
            importance_values = np.zeros(len(feature_cols))

        importance_df = (
            pd.DataFrame({"feature": feature_cols, "importance": importance_values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        display(importance_df)

        fig, ax = plt.subplots(figsize=(9, 5))
        top_importance = importance_df.head(10).sort_values("importance")
        ax.barh(top_importance["feature"], top_importance["importance"], color="#4c78a8")
        ax.set_title(f"Top feature importances: {best_model_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        **Why this matters:** A prediction alone is not enough for a report. Feature importance shows whether the model is mainly using recent case rates, older lags, deaths, population, or trend changes.
        """
    ),
    md(
        """
        ## 8. Historical Risk Predictions for Selected Regions

        These are test-period predictions for Odisha, Mizoram, and Tripura.
        """
    ),
    code(
        """
        historical_predictions = test.copy()
        historical_predictions["predicted_high_risk"] = test_pred
        historical_predictions["predicted_high_risk_probability"] = test_prob
        historical_predictions["risk_band"] = historical_predictions["predicted_high_risk_probability"].apply(risk_band)

        selected_historical_predictions = historical_predictions[
            historical_predictions["state"].isin(SELECTED_REGIONS)
        ][
            [
                "state",
                "year",
                "prediction_year",
                "cases_per_100k",
                "next_year_rate",
                "risk_target",
                "predicted_high_risk",
                "predicted_high_risk_probability",
                "risk_band",
            ]
        ].sort_values(["state", "year"])

        display(selected_historical_predictions)

        fig, ax = plt.subplots(figsize=(10, 5))
        for region in SELECTED_REGIONS:
            region_df = selected_historical_predictions[selected_historical_predictions["state"] == region]
            ax.plot(
                region_df["prediction_year"],
                region_df["predicted_high_risk_probability"],
                marker="o",
                linewidth=2,
                label=region,
                color=HIGHLIGHT_COLORS[region],
            )

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.50 decision line")
        ax.set_title("Predicted high-risk probability for selected regions")
        ax.set_xlabel("Prediction year")
        ax.set_ylabel("Predicted probability")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        **Why this matters:** This connects the ML model back to the chosen project regions. It shows not only whether the model predicts high risk, but also how confident the model is over the held-out years.
        """
    ),
    md(
        """
        ## 9. Build Latest-Year Features for 2025 Risk Scoring

        The labelled AI/ML file ends at feature year 2023 because it needs a known next year. To score 2025 risk, we create the same lagged features from the latest available state-year data.
        """
    ),
    code(
        """
        def build_latest_feature_rows(state_year_df):
            rows = []
            for state, group in state_year_df.dropna(subset=["population_2011"]).groupby("state"):
                group = group.sort_values("year").reset_index(drop=True)
                if len(group) < 4:
                    continue

                current = group.iloc[-1]
                lag1 = group.iloc[-2]
                lag2 = group.iloc[-3]
                lag3 = group.iloc[-4]

                current_cases = current["total_cases"]
                previous_cases = lag1["total_cases"]
                current_rate = current["cases_per_100k"]
                previous_rate = lag1["cases_per_100k"]

                rows.append(
                    {
                        "state": state,
                        "year": current["year"],
                        "prediction_year": current["year"] + 1,
                        "population_2011": current["population_2011"],
                        "district_count": current["district_count"],
                        "total_cases": current["total_cases"],
                        "cases_per_100k": current["cases_per_100k"],
                        "total_cases_lag1": lag1["total_cases"],
                        "total_cases_lag2": lag2["total_cases"],
                        "total_cases_lag3": lag3["total_cases"],
                        "total_deaths_lag1": lag1["total_deaths"],
                        "cases_per_100k_lag1": lag1["cases_per_100k"],
                        "cases_per_100k_lag2": lag2["cases_per_100k"],
                        "cases_per_100k_lag3": lag3["cases_per_100k"],
                        "deaths_per_100k_lag1": lag1["deaths_per_100k"],
                        "cases_rate_roll3": np.nanmean(
                            [
                                lag1["cases_per_100k"],
                                lag2["cases_per_100k"],
                                lag3["cases_per_100k"],
                            ]
                        ),
                        "cases_yoy_change": (current_cases - previous_cases) / previous_cases if previous_cases > 0 else np.nan,
                        "rate_yoy_change": (current_rate - previous_rate) / previous_rate if previous_rate > 0 else np.nan,
                    }
                )
            return pd.DataFrame(rows)

        latest_features = build_latest_feature_rows(state_year)
        latest_features = latest_features.dropna(subset=feature_cols).copy()

        display(latest_features[["state", "year", "prediction_year", "total_cases", "cases_per_100k"]].head())
        print("Latest feature year:", int(latest_features["year"].max()))
        print("Prediction year:", int(latest_features["prediction_year"].max()))
        print("Rows available for scoring:", len(latest_features))
        """
    ),
    md(
        """
        **Why this matters:** This turns the trained model into an actual risk prediction tool. Instead of only evaluating old predictions, we use the most recent available malaria history to estimate next-year risk.
        """
    ),
    md(
        """
        ## 10. Predict 2025 High-Risk Probability

        The final model is trained on all labelled historical rows, then used to score the latest available feature rows.
        """
    ),
    code(
        """
        final_model = clone(best_model_template)
        final_model.fit(model_df[feature_cols], model_df["risk_target"])

        latest_features["predicted_high_risk"] = final_model.predict(latest_features[feature_cols])
        latest_features["predicted_high_risk_probability"] = (
            final_model.predict_proba(latest_features[feature_cols])[:, 1]
            if hasattr(final_model, "predict_proba")
            else latest_features["predicted_high_risk"]
        )
        latest_features["risk_band"] = latest_features["predicted_high_risk_probability"].apply(risk_band)

        latest_risk_table = latest_features[
            [
                "state",
                "year",
                "prediction_year",
                "total_cases",
                "cases_per_100k",
                "predicted_high_risk",
                "predicted_high_risk_probability",
                "risk_band",
            ]
        ].sort_values("predicted_high_risk_probability", ascending=False)

        print("Top predicted high-risk states/UTs:")
        display(latest_risk_table.head(20))

        print("Selected project regions:")
        display(latest_risk_table[latest_risk_table["state"].isin(SELECTED_REGIONS)].sort_values("state"))
        """
    ),
    md(
        """
        **Why this matters:** This is the main risk prediction output. The probability column is more informative than only a yes/no label because it lets us compare states by relative predicted risk.
        """
    ),
    md(
        """
        ## 11. Visualize the Highest Predicted Risks
        """
    ),
    code(
        """
        top_latest = latest_risk_table.head(15).copy()
        colors = [HIGHLIGHT_COLORS.get(state, "#6b7280") for state in top_latest["state"]]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(
            top_latest["state"][::-1],
            top_latest["predicted_high_risk_probability"][::-1],
            color=colors[::-1],
        )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title("Top predicted malaria high-risk probabilities")
        ax.set_xlabel("Predicted probability of high risk")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        **Why this matters:** The bar chart makes the risk ranking easy to communicate in the final report. It also shows whether the selected regions appear among the highest predicted risks nationally.
        """
    ),
    md(
        """
        ## 12. Interpretation for the Report

        Use these points when writing the AI/ML section:

        - The target is next-year high malaria risk, defined by a training-period case-rate threshold.
        - The model uses lagged malaria indicators, rolling averages, and recent trend changes.
        - The chronological split makes the evaluation closer to real forecasting.
        - The final risk table gives predicted probability, not just a yes/no class.
        - The model can identify risk patterns, but it cannot prove causes because environmental and socioeconomic variables are not included.

        ## Biological Assumptions Not Included

        This AI/ML model does not explicitly model mosquito dynamics, rainfall, temperature, immunity, interventions, or human movement. Those factors may explain why risk differs between regions, but they are not present in the current CSV files.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
