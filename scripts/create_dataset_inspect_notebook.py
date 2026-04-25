#!/usr/bin/env python3
"""Create a notebook for inspecting all malaria project CSV files."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Malaria_Dataset_Inspect.ipynb"


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


datasets = [
    {
        "title": "Processed District Malaria Data",
        "var": "processed_district",
        "path": "data/processed/district_malaria_2000_2024_from_pdf.csv",
        "about": """
        This CSV is the district-level malaria dataset extracted from the long district-wise PDF.
        It contains annual malaria cases and deaths by district and state from 2000 to 2024.
        This is close to the original extracted table and is useful for checking source values.
        """,
    },
    {
        "title": "Processed State Malaria Situation Data",
        "var": "processed_state_situation",
        "path": "data/processed/state_malaria_situation_2021_2025_from_pdf.csv",
        "about": """
        This CSV is the state/UT-level malaria situation table extracted from the PDF.
        It contains tested, positive, Pf, and deaths for 2021 to 2025.
        It is useful for recent state-level trends and surveillance indicators.
        """,
    },
    {
        "title": "Processed State Epidemiological Data",
        "var": "processed_state_epi",
        "path": "data/processed/state_epidemiological_2024_2025_from_pdf.csv",
        "about": """
        This CSV is the state/UT-level epidemiological report for 2024 and 2025.
        It includes indicators such as Pf percentage, TPR, TFR, deaths, imported cases,
        indigenous cases, and category where available.
        """,
    },
    {
        "title": "Analysis-Ready District Malaria Data",
        "var": "clean_district",
        "path": "data/analysis_ready/district_malaria_clean.csv",
        "about": """
        This is the cleaned district-level dataset used for analysis.
        It keeps district cases and deaths, and adds state population plus per-capita rates.
        Use this when district-level analysis needs normalized case or death rates.
        """,
    },
    {
        "title": "Analysis-Ready State-Year Malaria Data",
        "var": "clean_state_year",
        "path": "data/analysis_ready/state_year_malaria_clean.csv",
        "about": """
        This is the main state-year dataset for modelling.
        It aggregates district values to state-year level and includes population,
        cases per 100,000, deaths per 100,000, district count, and selected-region labels.
        This is used by the geospatial, SIR, and ARIMA notebook.
        """,
    },
    {
        "title": "Selected Regions State-Year Data",
        "var": "selected_regions",
        "path": "data/analysis_ready/selected_regions_state_year.csv",
        "about": """
        This CSV contains only the three selected project regions:
        Odisha, Mizoram, and Tripura.
        It is a smaller state-year file for focused comparison, geospatial interpretation,
        SIR modelling, and ARIMA forecasting.
        """,
    },
    {
        "title": "AI/ML State-Year Feature Data",
        "var": "aiml_features",
        "path": "data/analysis_ready/aiml_state_year_features.csv",
        "about": """
        This is the machine-learning-ready dataset.
        It contains lag features, rolling averages, year-over-year changes, next-year cases,
        next-year case rate, and the high-risk classification target.
        This is used by the standalone AI/ML risk model notebook.
        """,
    },
    {
        "title": "State Population Data",
        "var": "population",
        "path": "data/raw/state_population_2011.csv",
        "about": """
        This CSV contains state/UT population denominators from Census 2011.
        It is used to calculate per-capita malaria rates such as cases per 100,000 population.
        """,
    },
]


cells = [
    md(
        """
        # Malaria Dataset Inspect

        This notebook loads each CSV used in the malaria project, displays the first and last rows,
        and explains what each dataset is for.
        """
    ),
    code(
        """
        from pathlib import Path
        import pandas as pd

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        elif not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent

        pd.set_option("display.max_columns", 100)
        """
    ),
]

for dataset in datasets:
    cells.extend(
        [
            md(f"## {dataset['title']}"),
            code(
                f"""
                {dataset['var']} = pd.read_csv(PROJECT_ROOT / "{dataset['path']}")
                print("Shape:", {dataset['var']}.shape)
                display({dataset['var']}.head())
                display({dataset['var']}.tail())
                """
            ),
            md(dataset["about"]),
        ]
    )


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
