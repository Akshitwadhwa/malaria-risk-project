# Project Plan: Malaria Risk Modelling

## Notebook Review

### `notebooks/TS_N_Final.ipynb`

Useful for the ML/time-series part of the malaria project.

It covers:

- Time-series visualization
- Trend estimation
- Seasonality checks
- Moving averages
- STL decomposition
- Forecast baselines: mean, random walk, seasonal naive, drift
- Forecast evaluation using MAE and RMSE
- AR, MA, ARMA, ARIMA, SARIMA, and ETS models

Current notebook data dependency:

- Expects `state_long.csv`

How to adapt:

- Replace COVID daily death data with malaria yearly state or district data.
- Use `data/processed/state_malaria_situation_2021_2025_from_pdf.csv` for state-level forecasting.
- Use `data/processed/district_malaria_2000_2024_from_pdf.csv` for longer district-level trend analysis.

### `notebooks/Week2_Epidemic_Modelling.ipynb`

Useful for the written/theory section.

It covers:

- What epidemic modelling is
- Mechanistic vs statistical models
- SIR and SEIR concepts
- Epidemiology terms

How to adapt:

- Use this content to explain why malaria can be studied with both ML and mechanistic models.

### `notebooks/Week_3_SIR_SEIR.ipynb`

Useful for the mechanistic modelling section.

It covers:

- SIR equations
- SEIR equations
- ODE simulation using `scipy.integrate.odeint`
- Public health interpretation of model parameters

How to adapt:

- Use it as a baseline mechanistic model.
- For malaria, discuss limitations because malaria is vector-borne, so a full model should include mosquito/vector dynamics.
- A simple SIR/SEIR model can still be used as an educational comparison.

### `notebooks/Week_4_ILI_PE.ipynb`

Useful for parameter estimation.

It covers:

- Fitting SIR model parameters to observed disease data
- SSE/grid search
- Numerical optimization with `scipy.optimize.minimize`
- Likelihood-based parameter estimation

Current notebook data dependency:

- Expects `ILINet.csv`

How to adapt:

- Replace ILI weekly counts with malaria case counts.
- Fit parameters to one high-burden state or district, such as Odisha, Mizoram, Tripura, Jharkhand, or Chhattisgarh.

## Recommended Project Direction

Best project structure:

1. Clean and validate PDF-extracted malaria data.
2. Perform exploratory data analysis by state, district, and year.
3. Build a district-level risk classification model.
4. Build a simple time-series forecasting baseline.
5. Add a mechanistic SIR/SEIR demonstration and explain how a vector-borne malaria model would extend it.

## Main Assignment Notebook

Created:

`notebooks/Malaria_Geospatial_Modeling_Assignment.ipynb`

Sections:

1. Region selection and spatial scale
2. Mandatory geospatial analysis with choropleth maps
3. Temporal spatial snapshots
4. Spatial heterogeneity interpretation
5. Mechanistic SIR model fitted separately for Odisha, Mizoram, and Tripura
6. ARIMA time-series model fitted separately for the same regions
7. Supervised AI/ML Random Forest model for next-year high-risk classification
8. Model comparison and interpretation template

## Standalone AI/ML Notebook

Created:

`notebooks/Malaria_AIML_Risk_Model.ipynb`

Purpose:

- Keep the AI/ML model separate from the geospatial/mechanistic assignment notebook.
- Train supervised models for next-year high malaria risk.
- Compare Dummy baseline, Logistic Regression, Random Forest, and Gradient Boosting.
- Interpret results using feature importance and selected-region predictions.
