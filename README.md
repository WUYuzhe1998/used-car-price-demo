# Vehicle Fair Market Price Range Demo

Streamlit demo for an already trained tabular MLP quantile regression model. The app predicts an advertised vehicle market listing price range and then applies rule-based post-processing for deal verdicts.

The web app does not retrain the model and does not require the original training CSV for inference.

## Required Files

The deployment must include these trained artifacts under `artifacts/`:

- `artifacts/best_model.pt`
- `artifacts/scaler.pkl`
- `artifacts/vocabularies.json`
- `artifacts/feature_config.json`
- `artifacts/metrics.json` optional, used only for displaying model metrics

The code files required by the app are:

- `app.py`
- `predict.py`
- `model.py`
- `preprocess.py`
- `requirements.txt`

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run one command-line prediction:

```bash
python predict.py \
  --model-dir artifacts \
  --listing-price 18000 \
  --input-json '{"make":"Volkswagen","model":"Golf","registration_year":2020,"mileage_km":72000,"power_kw":110,"fuel_category":"Gasoline","transmission":"Manual","body_type":"Compact","had_accident":false,"nr_prev_owners":1,"has_full_service_history":true,"seller_type":"Dealer","seller_is_dealer":true,"ratings_average":4.5,"ratings_count":20,"equipment_comfort_count":8,"equipment_entertainment_count":4,"equipment_extra_count":3,"equipment_safety_count":7}'
```

## Deploy on Streamlit Community Cloud

1. Push this project directory to a GitHub repository.
2. Make sure the required `artifacts/` files are committed or otherwise available in the repository.
3. In Streamlit Community Cloud, create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

5. Streamlit Cloud installs packages from `requirements.txt` and runs the app on CPU.

The trained artifact files are small in this project, so they can be committed directly. If future model files exceed GitHub limits, store them with Git LFS or download them during deployment startup.

## Model Logic

The trained MLP predicts three log-scale quantiles of advertised listing price:

- `P10`: low market listing price
- `P50`: median market listing price
- `P90`: high market listing price

The model output is converted back to EUR with `expm1`. The architecture enforces ordered outputs, so `P10 <= P50 <= P90`.

The prediction target during training was `price` because supervised learning needs the observed listing price as the label. During inference, the user listing price is not a model input. It is only compared against the predicted range after the model has produced `P10/P50/P90`.

Using `P10/P50/P90` is more useful than a single price because car listings have natural uncertainty. The range supports fair-market interpretation: low, typical, and high advertised prices for similar vehicle attributes.

## Verdict Logic

After prediction, the app calculates the price position:

- `listing_price < P10`: Below Market
- `P10 <= listing_price <= P90`: Fair Range
- `listing_price > P90`: Above Market
- Missing listing price: Unknown

Risk is rule-based because the dataset does not contain a true fraud or post-purchase failure label. Risk increases for accident history, no full service history, many previous owners, private seller, low seller rating, very low or missing rating count, and missing important fields.

Final verdict:

- Below Market + Low Risk or Medium Risk: Good Deal
- Below Market + High Risk: Suspiciously Low
- Fair Range: Fair Price
- Above Market: Overpriced
- Missing listing price: Price Range Only

The model predicts advertised market listing price range, not final transaction price.
