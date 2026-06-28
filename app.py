from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from predict import DEFAULT_MODEL_DIR, load_artifacts, predict_from_artifacts


st.set_page_config(
    page_title="Vehicle Price Range Demo",
    layout="wide",
)


@st.cache_resource
def cached_artifacts(model_dir: str) -> dict[str, Any]:
    return load_artifacts(Path(model_dir), device="cpu")


def bool_select(label: str, value: str = "Unknown") -> bool | None:
    selected = st.selectbox(label, ["Unknown", "Yes", "No"], index=["Unknown", "Yes", "No"].index(value))
    if selected == "Yes":
        return True
    if selected == "No":
        return False
    return None


def format_eur(value: float | None) -> str:
    if value is None:
        return "Not provided"
    return f"EUR {value:,.0f}"


def format_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "Unknown"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def verdict_color(verdict: str) -> str:
    if verdict == "Good Deal":
        return "normal"
    if verdict in {"Suspiciously Low", "Overpriced"}:
        return "inverse"
    return "off"


def normalize_seller_type(value: str) -> str:
    if value == "Private seller":
        return "PrivateSeller"
    return value


model_dir = Path(DEFAULT_MODEL_DIR)

st.title("Vehicle Fair Market Price Range")

try:
    artifacts = cached_artifacts(str(model_dir))
except Exception as exc:
    st.error("Model artifacts could not be loaded.")
    st.code(str(exc))
    st.stop()

metrics = artifacts.get("metrics") or {}
test_metrics = metrics.get("test", {})

with st.sidebar:
    st.header("Model")
    st.write("MLP quantile regression")
    st.write(f"Checkpoint: `{artifacts['checkpoint_path'].name}`")
    if test_metrics:
        st.metric("Test P50 MAE", format_eur(test_metrics.get("p50_mae_eur")))
        st.metric("P10-P90 Coverage", f"{test_metrics.get('coverage_inside_p10_p90_percent', 0):.1f}%")

with st.form("vehicle_form"):
    st.subheader("Listing")
    price_col, make_col, model_col = st.columns([1, 1, 1])
    with price_col:
        has_listing_price = st.checkbox("Listing price available", value=True)
        listing_price = None
        if has_listing_price:
            listing_price = st.number_input(
                "Listing price (EUR)",
                min_value=0.0,
                value=22000.0,
                step=500.0,
            )
    with make_col:
        make = st.text_input("Make", value="Volkswagen")
    with model_col:
        model = st.text_input("Model", value="Golf")

    st.subheader("Vehicle")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        registration_year = st.number_input(
            "Registration year",
            min_value=1950,
            max_value=2035,
            value=2020,
            step=1,
        )
        mileage_km = st.number_input("Mileage (km)", min_value=0, value=72000, step=1000)
    with v2:
        power_kw = st.number_input("Power (kW)", min_value=0, value=110, step=5)
        fuel_category = st.selectbox(
            "Fuel type",
            ["Unknown", "Gasoline", "Diesel", "Hybrid", "Electric", "LPG", "CNG"],
            index=1,
        )
    with v3:
        transmission = st.selectbox(
            "Transmission",
            ["Unknown", "Manual", "Automatic", "Semi-automatic"],
            index=1,
        )
        body_type = st.selectbox(
            "Body type",
            [
                "Unknown",
                "Compact",
                "Sedan",
                "Station wagon",
                "SUV/Off-Road/Pick-up",
                "Coupe",
                "Convertible",
                "Van",
            ],
            index=1,
        )
    with v4:
        had_accident = bool_select("Accident history", value="No")
        has_full_service_history = bool_select("Full service history", value="Yes")

    st.subheader("Seller")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        nr_prev_owners = st.number_input("Previous owners", min_value=0, value=1, step=1)
        seller_type = st.selectbox("Seller type", ["Unknown", "Dealer", "Private seller"], index=1)
    with s2:
        seller_is_dealer = bool_select("Is the seller a dealer?", value="Yes")
        country_code = st.text_input("Country code", value="DE")
    with s3:
        ratings_average = st.number_input(
            "Seller rating",
            min_value=0.0,
            max_value=5.0,
            value=4.5,
            step=0.1,
        )
        ratings_count = st.number_input("Rating count", min_value=0, value=20, step=1)
    with s4:
        offer_type = st.selectbox("Offer type", ["Unknown", "Used", "New", "Demonstration"], index=1)
        vehicle_type = st.selectbox("Vehicle type", ["Unknown", "Car", "Classic", "Pre-registered"], index=1)

    st.subheader("Equipment")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        equipment_comfort_count = st.number_input("Comfort options", min_value=0, value=8, step=1)
    with e2:
        equipment_entertainment_count = st.number_input("Entertainment options", min_value=0, value=4, step=1)
    with e3:
        equipment_extra_count = st.number_input("Extra options", min_value=0, value=3, step=1)
    with e4:
        equipment_safety_count = st.number_input("Safety options", min_value=0, value=7, step=1)

    submitted = st.form_submit_button("Predict price range", use_container_width=True)

if submitted:
    vehicle = {
        "make": make,
        "model": model,
        "registration_year": int(registration_year),
        "mileage_km": float(mileage_km),
        "power_kw": float(power_kw),
        "fuel_category": fuel_category,
        "transmission": transmission,
        "body_type": body_type,
        "had_accident": had_accident,
        "nr_prev_owners": int(nr_prev_owners),
        "has_full_service_history": has_full_service_history,
        "seller_type": normalize_seller_type(seller_type),
        "seller_is_dealer": seller_is_dealer,
        "ratings_average": float(ratings_average),
        "ratings_count": int(ratings_count),
        "country_code": country_code,
        "offer_type": offer_type,
        "vehicle_type": vehicle_type,
        "is_used": offer_type != "New",
        "is_new": offer_type == "New",
        "is_preregistered": vehicle_type == "Pre-registered",
        "equipment_comfort_count": int(equipment_comfort_count),
        "equipment_entertainment_count": int(equipment_entertainment_count),
        "equipment_extra_count": int(equipment_extra_count),
        "equipment_safety_count": int(equipment_safety_count),
    }

    result = predict_from_artifacts(
        artifacts,
        vehicle,
        listing_price=listing_price if has_listing_price else None,
    )

    st.divider()
    st.caption(result["car_summary"])
    st.metric(
        "Final verdict",
        result["final_verdict"],
        delta=result["price_position"],
        delta_color=verdict_color(result["final_verdict"]),
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("P10 low market price", format_eur(result["p10"]))
    r2.metric("P50 median market price", format_eur(result["p50"]))
    r3.metric("P90 high market price", format_eur(result["p90"]))
    r4.metric("Listing price", format_eur(result["listing_price"]))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Risk")
        st.metric("Risk level", f"{result['risk_level']} ({result['risk_score']})")
        for factor in result["risk_factors"]:
            st.write(f"- {factor}")
    with c2:
        st.subheader("Derived features")
        derived = result["engineered_features"]
        st.write(f"- Vehicle age: {format_number(derived.get('age_years'))} years")
        st.write(f"- Mileage per year: {format_number(derived.get('mileage_per_year'), 0)} km/year")
        st.write(f"- Zero mileage flag: {format_number(derived.get('is_zero_mileage'), 0)}")
        st.write(f"- Low mileage flag: {format_number(derived.get('is_low_mileage'), 0)}")
        st.write(f"- Nearly-new flag: {format_number(derived.get('is_nearly_new'), 0)}")
        st.write(f"- Total equipment count: {format_number(derived.get('total_equipment_count'), 0)}")

    st.info(
        "The MLP predicts advertised market listing price range from vehicle attributes. "
        "The user listing price is not used as a model input; it is compared with P10/P50/P90 after prediction."
    )
