#!/usr/bin/env python3
"""Create the assignment notebook for malaria geospatial + modelling work."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Malaria_Geospatial_Modeling_Assignment.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


cells = [
    md(
        """
        # Malaria Geospatial Analysis and Regional Epidemic Modelling

        This notebook follows the assignment requirements:

        1. Select at least three comparable or contrasting regions and define the spatial scale.
        2. Perform mandatory geospatial analysis before modelling.
        3. Fit two models to the same regional data:
           - Model A: mechanistic SIR model
           - Model B: machine learning/time-series ARIMA model

        The malaria data were extracted from the supplied PDF files into CSV format.
        """
    ),
    md(
        """
        ## 0. Setup

        Run the install cell once if these packages are not already available in your notebook environment.
        """
    ),
    code(
        """
        # Uncomment and run this cell if packages are missing.
        # %pip install pandas numpy matplotlib plotly scipy statsmodels scikit-learn
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.express as px

        from scipy.integrate import solve_ivp
        from scipy.optimize import minimize
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
        from statsmodels.tsa.arima.model import ARIMA

        warnings.filterwarnings("ignore")

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        elif not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent
        DATA_DIR = PROJECT_ROOT / "data"
        PROCESSED_DIR = DATA_DIR / "processed"
        RAW_DIR = DATA_DIR / "raw"

        DISTRICT_CSV = PROCESSED_DIR / "district_malaria_2000_2024_from_pdf.csv"
        POPULATION_CSV = RAW_DIR / "state_population_2011.csv"
        GEOJSON_PATH = RAW_DIR / "india_states.geojson"

        SELECTED_REGIONS = ["Odisha", "Mizoram", "Tripura"]
        SPATIAL_SCALE = "state-level"
        """
    ),
    md(
        """
        ## 1. Select Regions and Spatial Scale

        **Spatial scale:** state-level analysis within India.

        **Selected regions:** Odisha, Mizoram, and Tripura.

        **Justification:**

        - These states are comparable because they all show substantial malaria burden in the provided dataset.
        - They are contrasting because they differ sharply in population size, geography, and epidemic intensity.
        - Odisha has high raw case counts and a large population.
        - Mizoram has a much smaller population but high malaria intensity, making per-capita rates especially important.
        - Tripura provides another northeastern comparison with strong malaria burden but a different trend profile.

        This makes the three-region comparison useful for showing why raw case counts and population-adjusted rates can lead to different interpretations.
        """
    ),
    code(
        """
        district = pd.read_csv(DISTRICT_CSV)
        population = pd.read_csv(POPULATION_CSV)

        district["year"] = district["year"].astype(int)
        district["total_cases"] = pd.to_numeric(district["total_cases"], errors="coerce")
        district["total_deaths"] = pd.to_numeric(district["total_deaths"], errors="coerce")

        state_year = (
            district
            .groupby(["year", "state"], as_index=False)
            .agg(total_cases=("total_cases", "sum"), total_deaths=("total_deaths", "sum"))
            .merge(population[["state", "population_2011"]], on="state", how="left")
        )
        state_year["cases_per_100k"] = state_year["total_cases"] / state_year["population_2011"] * 100_000

        selected = state_year[state_year["state"].isin(SELECTED_REGIONS)].copy()

        print("Selected regions:", SELECTED_REGIONS)
        print("Spatial scale:", SPATIAL_SCALE)
        display(selected.head())
        """
    ),
    md(
        """
        ## 2. Mandatory Geospatial Data Analysis

        This section comes before modelling, as required.

        It includes:

        - Choropleth map of cumulative cases
        - Choropleth map of per-capita case rates
        - Temporal snapshots for early, peak, and late epidemic periods
        - Interpretation of spatial heterogeneity and implications for modelling

        The downloaded GeoJSON is district-level. For state-level choropleths, each district polygon is colored using its state's value. This preserves real geography while keeping the modelling scale at state level.
        """
    ),
    code(
        """
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            india_geojson = json.load(f)

        STATE_NAME_FIX = {
            "Andaman and Nicobar Islands": "Andaman And Nicobar Islands",
            "Dadra and Nagar Haveli and Daman and Diu": "The Dadra And Nagar Haveli And Daman And Diu",
            "Jammu and Kashmir": "Jammu And Kashmir",
        }

        geo_rows = []
        for feature in india_geojson["features"]:
            props = feature["properties"]
            map_state = props["st_nm"]
            data_state = STATE_NAME_FIX.get(map_state, map_state)
            geo_rows.append({
                "dt_code": str(props["dt_code"]),
                "map_state": map_state,
                "state": data_state,
                "map_district": props["district"],
            })

        geo_index = pd.DataFrame(geo_rows)
        geo_index.head()
        """
    ),
    code(
        """
        cumulative = (
            state_year
            .groupby("state", as_index=False)
            .agg(
                cumulative_cases=("total_cases", "sum"),
                population_2011=("population_2011", "first"),
            )
        )
        cumulative["cumulative_cases_per_100k"] = (
            cumulative["cumulative_cases"] / cumulative["population_2011"] * 100_000
        )

        map_cumulative = geo_index.merge(cumulative, on="state", how="left")

        def plot_choropleth(map_df, value_col, title, color_scale="YlOrRd"):
            fig = px.choropleth(
                map_df,
                geojson=india_geojson,
                locations="dt_code",
                featureidkey="properties.dt_code",
                color=value_col,
                hover_name="map_district",
                hover_data={
                    "map_state": True,
                    value_col: ":,.2f",
                    "dt_code": False,
                },
                color_continuous_scale=color_scale,
                title=title,
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            fig.show()

        plot_choropleth(
            map_cumulative,
            "cumulative_cases",
            "Cumulative Malaria Cases by State, 2000-2024",
        )
        """
    ),
    code(
        """
        plot_choropleth(
            map_cumulative,
            "cumulative_cases_per_100k",
            "Cumulative Malaria Case Rate per 100,000 Population, 2000-2024",
            color_scale="Viridis",
        )
        """
    ),
    code(
        """
        national_by_year = (
            state_year
            .groupby("year", as_index=False)
            .agg(total_cases=("total_cases", "sum"))
            .sort_values("year")
        )

        early_year = int(national_by_year["year"].min())
        peak_year = int(national_by_year.loc[national_by_year["total_cases"].idxmax(), "year"])
        late_year = int(national_by_year["year"].max())

        print("Temporal snapshots:")
        print("Early year:", early_year)
        print("Peak year by national total cases:", peak_year)
        print("Late year:", late_year)

        snapshot_years = [early_year, peak_year, late_year]
        snapshot_data = state_year[state_year["year"].isin(snapshot_years)].copy()
        """
    ),
    code(
        """
        for year in snapshot_years:
            year_metrics = state_year[state_year["year"] == year][
                ["state", "total_cases", "cases_per_100k"]
            ]
            map_year = geo_index.merge(year_metrics, on="state", how="left")
            plot_choropleth(
                map_year,
                "cases_per_100k",
                f"Temporal Snapshot: Malaria Case Rate per 100,000 in {year}",
                color_scale="Plasma",
            )
        """
    ),
    code(
        """
        comparison = (
            cumulative[cumulative["state"].isin(SELECTED_REGIONS)]
            .sort_values("cumulative_cases_per_100k", ascending=False)
        )

        display(comparison)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        comparison.sort_values("cumulative_cases").plot(
            x="state", y="cumulative_cases", kind="barh", ax=axes[0], legend=False
        )
        axes[0].set_title("Cumulative cases")
        axes[0].set_xlabel("Cases")

        comparison.sort_values("cumulative_cases_per_100k").plot(
            x="state", y="cumulative_cases_per_100k", kind="barh", ax=axes[1], legend=False, color="darkorange"
        )
        axes[1].set_title("Cumulative cases per 100,000")
        axes[1].set_xlabel("Cases per 100,000")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### Geospatial Interpretation

        **Do all regions show similar epidemic intensity?**

        No. Odisha, Mizoram, and Tripura all show meaningful malaria burden, but the intensity differs strongly. Odisha is expected to dominate raw cumulative counts because it has a much larger population. Mizoram and Tripura can appear more intense after adjusting for population.

        **Are raw counts misleading compared to per-capita rates?**

        Yes. Raw counts emphasize larger states, while per-capita rates reveal where the disease burden is intense relative to population size. This is especially important for smaller states like Mizoram and Tripura.

        **What spatial patterns might affect model behavior?**

        Spatial heterogeneity means one shared model would be inappropriate. High-burden regions can have different transmission patterns, surveillance intensity, population denominators, ecological conditions, and health-system response. Therefore, the mechanistic and ML models are fit separately for each selected region.
        """
    ),
    md(
        """
        ## 3. Model A: Mechanistic Model

        ### Model Choice

        We use a simple SIR model because the available malaria data contain annual case counts but not exposed, recovered, relapse, vector abundance, immunity, or mosquito infection data.

        Malaria is vector-borne, so a full biological model would ideally use a Ross-Macdonald or vector-host framework. Here, SIR is used as a simplified mechanistic approximation to compare region-specific transmission behavior.

        ### Compartments

        - **S(t):** susceptible population
        - **I(t):** infected population
        - **R(t):** removed/recovered population
        - **C(t):** cumulative infections generated by the model

        ### Parameters

        - **beta:** transmission rate
        - **gamma:** recovery/removal rate
        - **R0 = beta / gamma:** basic reproduction number proxy

        Parameters are estimated separately for each region by minimizing the squared error between observed annual malaria cases and model-generated annual infections.
        """
    ),
    code(
        """
        def sir_with_cumulative(t, y, beta, gamma, population):
            S, I, R, C = y
            new_infections = beta * S * I / population
            dS = -new_infections
            dI = new_infections - gamma * I
            dR = gamma * I
            dC = new_infections
            return [dS, dI, dR, dC]

        def simulate_sir_annual(beta, gamma, population, initial_infected, n_years):
            initial_infected = max(1.0, min(initial_infected, population * 0.1))
            y0 = [population - initial_infected, initial_infected, 0.0, 0.0]
            t_eval = np.arange(0, n_years + 1)
            sol = solve_ivp(
                sir_with_cumulative,
                [0, n_years],
                y0,
                t_eval=t_eval,
                args=(beta, gamma, population),
                method="RK45",
            )
            cumulative = sol.y[3]
            annual_cases = np.diff(cumulative)
            return np.maximum(annual_cases, 0)

        def fit_sir_for_region(region_df):
            region_df = region_df.sort_values("year")
            observed = region_df["total_cases"].to_numpy(dtype=float)
            population = float(region_df["population_2011"].iloc[0])
            n_years = len(observed)

            def objective(params):
                beta, gamma, initial_infected = params
                pred = simulate_sir_annual(beta, gamma, population, initial_infected, n_years)
                return np.mean((np.log1p(observed) - np.log1p(pred)) ** 2)

            initial_guess = [0.8, 0.5, max(1.0, observed[0])]
            bounds = [(1e-4, 5.0), (1e-4, 5.0), (1.0, max(10.0, population * 0.01))]
            result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

            beta, gamma, initial_infected = result.x
            fitted = simulate_sir_annual(beta, gamma, population, initial_infected, n_years)
            return {
                "beta": beta,
                "gamma": gamma,
                "R0_proxy": beta / gamma,
                "initial_infected": initial_infected,
                "sir_mae": mean_absolute_error(observed, fitted),
                "sir_rmse": mean_squared_error(observed, fitted, squared=False),
                "fitted": fitted,
                "success": result.success,
            }

        sir_results = {}
        sir_summary_rows = []

        for region in SELECTED_REGIONS:
            region_df = selected[selected["state"] == region].sort_values("year")
            fit = fit_sir_for_region(region_df)
            sir_results[region] = fit
            sir_summary_rows.append({
                "region": region,
                "beta": fit["beta"],
                "gamma": fit["gamma"],
                "R0_proxy": fit["R0_proxy"],
                "initial_infected": fit["initial_infected"],
                "MAE": fit["sir_mae"],
                "RMSE": fit["sir_rmse"],
                "optimizer_success": fit["success"],
            })

        sir_summary = pd.DataFrame(sir_summary_rows)
        display(sir_summary)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(len(SELECTED_REGIONS), 1, figsize=(10, 9), sharex=True)

        for ax, region in zip(axes, SELECTED_REGIONS):
            region_df = selected[selected["state"] == region].sort_values("year")
            ax.plot(region_df["year"], region_df["total_cases"], marker="o", label="Observed cases")
            ax.plot(region_df["year"], sir_results[region]["fitted"], marker="x", label="SIR fitted annual infections")
            ax.set_title(f"SIR fit: {region}")
            ax.set_ylabel("Cases")
            ax.legend()

        axes[-1].set_xlabel("Year")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 4. Model B: Machine Learning / Time-Series Model

        We use ARIMA as the ML/time-series model. ARIMA treats each regional epidemic curve as a forecasting problem.

        ### What biological assumptions are not included?

        ARIMA does not explicitly include:

        - susceptible or recovered compartments
        - mosquito/vector dynamics
        - transmission mechanisms
        - immunity or relapse
        - rainfall, temperature, or intervention effects

        It only learns statistical structure from past case counts. This makes it useful for forecasting baselines, but less biologically interpretable than mechanistic models.
        """
    ),
    code(
        """
        def fit_arima_for_region(region_df, forecast_steps=4):
            region_df = region_df.sort_values("year")
            y = region_df["total_cases"].astype(float).to_numpy()
            years = region_df["year"].to_numpy()

            train_y = y[:-forecast_steps]
            test_y = y[-forecast_steps:]
            test_years = years[-forecast_steps:]

            candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (1, 0, 0)]
            best_model = None
            best_order = None
            best_aic = np.inf

            for order in candidate_orders:
                try:
                    model = ARIMA(train_y, order=order).fit()
                    if model.aic < best_aic:
                        best_model = model
                        best_order = order
                        best_aic = model.aic
                except Exception:
                    continue

            if best_model is None:
                forecast = np.repeat(train_y[-1], forecast_steps)
                best_order = "naive_fallback"
                best_aic = np.nan
            else:
                forecast = np.asarray(best_model.forecast(steps=forecast_steps))
                forecast = np.maximum(forecast, 0)

            return {
                "order": best_order,
                "aic": best_aic,
                "test_years": test_years,
                "test_y": test_y,
                "forecast": forecast,
                "mae": mean_absolute_error(test_y, forecast),
                "rmse": mean_squared_error(test_y, forecast, squared=False),
            }

        arima_results = {}
        arima_summary_rows = []

        for region in SELECTED_REGIONS:
            region_df = selected[selected["state"] == region].sort_values("year")
            fit = fit_arima_for_region(region_df)
            arima_results[region] = fit
            arima_summary_rows.append({
                "region": region,
                "ARIMA_order": fit["order"],
                "AIC": fit["aic"],
                "MAE": fit["mae"],
                "RMSE": fit["rmse"],
            })

        arima_summary = pd.DataFrame(arima_summary_rows)
        display(arima_summary)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(len(SELECTED_REGIONS), 1, figsize=(10, 9), sharex=True)

        for ax, region in zip(axes, SELECTED_REGIONS):
            region_df = selected[selected["state"] == region].sort_values("year")
            fit = arima_results[region]
            ax.plot(region_df["year"], region_df["total_cases"], marker="o", label="Observed")
            ax.plot(fit["test_years"], fit["forecast"], marker="x", label=f"ARIMA forecast {fit['order']}")
            ax.axvline(fit["test_years"][0], color="gray", linestyle="--", alpha=0.6)
            ax.set_title(f"ARIMA holdout forecast: {region}")
            ax.set_ylabel("Cases")
            ax.legend()

        axes[-1].set_xlabel("Year")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 5. Additional AI/ML Model: Random Forest Risk Classifier

        The assignment requires one ML/time-series model, which ARIMA already satisfies. However, to include a clearer AI/ML component, we also fit a supervised machine learning model.

        ### ML Objective

        Predict whether a state will be **high malaria risk in the next year**.

        ### Target Variable

        A state-year is labelled high risk if the **next year's cases per 100,000 population** is in the top 25% of all state-year rates.

        ### Features

        The model uses lagged epidemiological features:

        - previous year's total cases
        - previous two years' case rates
        - rolling 3-year average case rate
        - population denominator
        - calendar year

        This model is not mechanistic. It does not model biological compartments or mosquito transmission. It learns statistical patterns from historical surveillance data.
        """
    ),
    code(
        """
        ml_df = state_year.sort_values(["state", "year"]).copy()

        for col in ["total_cases", "cases_per_100k"]:
            ml_df[f"{col}_lag1"] = ml_df.groupby("state")[col].shift(1)
            ml_df[f"{col}_lag2"] = ml_df.groupby("state")[col].shift(2)

        ml_df["rate_roll3"] = (
            ml_df
            .groupby("state")["cases_per_100k"]
            .transform(lambda series: series.shift(1).rolling(3).mean())
        )

        ml_df["next_year_rate"] = ml_df.groupby("state")["cases_per_100k"].shift(-1)
        risk_threshold = ml_df["next_year_rate"].quantile(0.75)
        ml_df["high_risk_next_year"] = (ml_df["next_year_rate"] >= risk_threshold).astype(int)

        feature_cols = [
            "year",
            "population_2011",
            "total_cases_lag1",
            "total_cases_lag2",
            "cases_per_100k_lag1",
            "cases_per_100k_lag2",
            "rate_roll3",
        ]

        ml_model_df = ml_df.dropna(subset=feature_cols + ["next_year_rate", "high_risk_next_year"]).copy()

        train = ml_model_df[ml_model_df["year"] <= 2019]
        test = ml_model_df[ml_model_df["year"] > 2019]

        X_train = train[feature_cols]
        y_train = train["high_risk_next_year"]
        X_test = test[feature_cols]
        y_test = test["high_risk_next_year"]

        rf_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=3,
        )
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]

        print("High-risk threshold, next-year cases per 100k:", round(risk_threshold, 2))
        print(classification_report(y_test, y_pred, target_names=["not high risk", "high risk"]))

        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=["not high risk", "high risk"],
        ).plot(cmap="Blues")
        plt.title("Random Forest High-Risk Classification")
        plt.show()
        """
    ),
    code(
        """
        importances = (
            pd.DataFrame({
                "feature": feature_cols,
                "importance": rf_model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
        )

        display(importances)

        selected_ml_predictions = test[test["state"].isin(SELECTED_REGIONS)].copy()
        selected_ml_predictions["predicted_high_risk"] = y_pred[test["state"].isin(SELECTED_REGIONS)]
        selected_ml_predictions["predicted_high_risk_probability"] = y_prob[test["state"].isin(SELECTED_REGIONS)]

        display(
            selected_ml_predictions[
                [
                    "state",
                    "year",
                    "next_year_rate",
                    "high_risk_next_year",
                    "predicted_high_risk",
                    "predicted_high_risk_probability",
                ]
            ].sort_values(["state", "year"])
        )
        """
    ),
    md(
        """
        ## 6. Compare Model Outputs

        Use this section to compare the models region-by-region.

        - The SIR model gives interpretable parameters such as beta, gamma, and an R0 proxy.
        - ARIMA gives a data-driven forecasting baseline.
        - Random Forest gives a supervised AI/ML risk classification model.
        - Differences between regions should be interpreted together with the geospatial findings, especially the difference between raw counts and per-capita rates.
        """
    ),
    code(
        """
        model_comparison = (
            sir_summary[["region", "beta", "gamma", "R0_proxy", "MAE", "RMSE"]]
            .rename(columns={"MAE": "SIR_MAE", "RMSE": "SIR_RMSE"})
            .merge(
                arima_summary[["region", "ARIMA_order", "MAE", "RMSE"]]
                .rename(columns={"MAE": "ARIMA_MAE", "RMSE": "ARIMA_RMSE"}),
                on="region",
            )
        )
        display(model_comparison)
        """
    ),
    md(
        """
        ## 7. Final Interpretation Template

        Write your final interpretation using the outputs above:

        1. **Spatial scale and regions:** State-level comparison of Odisha, Mizoram, and Tripura.
        2. **Geospatial finding:** Raw cases and per-capita rates show different spatial patterns.
        3. **Spatial heterogeneity:** Regions differ enough that separate regional models are justified.
        4. **Mechanistic model:** Compare beta, gamma, and R0 proxy across regions.
        5. **ML/time-series model:** Compare ARIMA forecast errors across regions.
        6. **AI/ML risk model:** Interpret Random Forest high-risk predictions and feature importances.
        7. **Limitation:** SIR, ARIMA, and Random Forest do not fully represent malaria vector ecology; stronger future models should include rainfall, temperature, mosquito dynamics, and intervention coverage.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
