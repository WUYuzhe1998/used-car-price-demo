from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from model import OrderedQuantileMLP
from preprocess import PriceRangePreprocessor, build_feature_frame


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "artifacts"

MISSING_STRINGS = {"", "unknown", "nan", "none", "null", "n/a", "na", "<na>"}
TRUE_STRINGS = {"true", "1", "yes", "y", "t", "ja", "oui"}
FALSE_STRINGS = {"false", "0", "no", "n", "f", "nein", "nee"}
MODEL_INPUT_EXCLUDE = {
    "listing_price",
    "user_listing_price",
    "price",
    "price_net",
    "price_vat_rate",
    "price_currency",
    "price_negotiable",
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in MISSING_STRINGS


def to_bool(value: Any) -> bool | None:
    if is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return None


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_input_json(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(value)
    path = Path(value)
    if path.exists():
        return load_json_file(path)
    return json.loads(value)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _model_checkpoint_path(model_dir: Path) -> Path:
    for name in ["best_model.pt", "model.pt"]:
        path = model_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing best_model.pt or model.pt in {model_dir}")


def validate_required_artifacts(model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
    model_dir = Path(model_dir)
    required = [
        _model_checkpoint_path(model_dir),
        model_dir / "scaler.pkl",
        model_dir / "vocabularies.json",
        model_dir / "feature_config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required model artifacts: " + ", ".join(missing))


def load_artifacts(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str = "cpu") -> dict[str, Any]:
    model_dir = Path(model_dir)
    validate_required_artifacts(model_dir)

    torch_device = torch.device(device)
    checkpoint_path = _model_checkpoint_path(model_dir)
    checkpoint = _torch_load(checkpoint_path, torch_device)
    if checkpoint.get("deal_verdict_classifier_trained") is True:
        raise ValueError("This app expects a price range model, not a verdict classifier")

    model_config = dict(checkpoint["model_config"])
    model_config.pop("model_class", None)
    model = OrderedQuantileMLP(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()

    metrics_path = model_dir / "metrics.json"
    metrics = load_json_file(metrics_path) if metrics_path.exists() else None

    return {
        "model": model,
        "preprocessor": PriceRangePreprocessor.load(model_dir),
        "device": torch_device,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "metrics": metrics,
    }


def sanitize_vehicle_input(vehicle: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in vehicle.items()
        if key not in MODEL_INPUT_EXCLUDE and not key.lower().startswith("price_")
    }

    if "mileage" in cleaned and "mileage_km" not in cleaned:
        cleaned["mileage_km"] = cleaned.pop("mileage")
    if "power" in cleaned and "power_kw" not in cleaned:
        cleaned["power_kw"] = cleaned.pop("power")

    return cleaned


def price_position(listing_price: float | None, p10: float, p90: float) -> str:
    if listing_price is None:
        return "Unknown"
    if listing_price < p10:
        return "Below Market"
    if listing_price <= p90:
        return "Fair Range"
    return "Above Market"


def risk_level(score: int) -> str:
    if score <= 1:
        return "Low Risk"
    if score <= 3:
        return "Medium Risk"
    return "High Risk"


def is_severely_below_market(listing_price: float | None, p10: float) -> bool:
    if listing_price is None or p10 <= 0:
        return False
    return listing_price < p10 * 0.70


def add_price_anomaly_risk(
    risk: dict[str, Any],
    listing_price: float | None,
    p10: float,
) -> dict[str, Any]:
    adjusted = {
        "risk_score": int(risk["risk_score"]),
        "risk_level": risk["risk_level"],
        "risk_factors": list(risk["risk_factors"]),
    }
    if not is_severely_below_market(listing_price, p10):
        return adjusted

    if adjusted["risk_factors"] == ["No major risk signals from available fields"]:
        adjusted["risk_factors"] = []

    ratio = listing_price / p10 if listing_price is not None and p10 > 0 else 0.0
    adjusted["risk_score"] += 4
    adjusted["risk_level"] = risk_level(adjusted["risk_score"])
    adjusted["risk_factors"].append(
        f"Listing price is severely below the predicted P10 market price "
        f"({ratio:.0%} of P10)"
    )
    return adjusted


def compute_risk_score(vehicle: dict[str, Any]) -> dict[str, Any]:
    score = 0
    factors: list[str] = []

    if to_bool(vehicle.get("had_accident")) is True:
        score += 2
        factors.append("Accident history is reported")

    service_history = to_bool(vehicle.get("has_full_service_history"))
    if service_history is not True:
        score += 2
        factors.append("Full service history is missing or not confirmed")

    previous_owners = to_float(vehicle.get("nr_prev_owners"))
    if previous_owners is not None and previous_owners >= 3:
        score += 1
        factors.append("Previous owners is high")

    seller_type = str(vehicle.get("seller_type", "")).strip().lower()
    seller_is_dealer = to_bool(vehicle.get("seller_is_dealer"))
    if "private" in seller_type or seller_is_dealer is False:
        score += 1
        factors.append("Seller is private or not marked as a dealer")

    ratings_average = to_float(vehicle.get("ratings_average"))
    if ratings_average is not None and ratings_average < 3.5:
        score += 1
        factors.append("Seller rating average is low")

    ratings_count = to_float(vehicle.get("ratings_count"))
    if ratings_count is None or ratings_count < 3:
        score += 1
        factors.append("Seller rating count is very low or missing")

    important_fields = [
        "make",
        "model",
        "registration_year",
        "registration_date",
        "mileage_km",
        "power_kw",
        "fuel_category",
        "transmission",
        "body_type",
        "nr_prev_owners",
        "has_full_service_history",
        "had_accident",
        "seller_type",
        "seller_is_dealer",
        "ratings_average",
        "ratings_count",
    ]
    registration_present = not (
        is_missing(vehicle.get("registration_year"))
        and is_missing(vehicle.get("registration_date"))
    )
    missing_fields = []
    for field in important_fields:
        if field in {"registration_year", "registration_date"}:
            continue
        if is_missing(vehicle.get(field)):
            missing_fields.append(field)
    if not registration_present:
        missing_fields.append("registration_year_or_date")

    if len(missing_fields) >= 4:
        score += 2
        factors.append("Many important fields are missing: " + ", ".join(missing_fields))
    elif missing_fields:
        score += 1
        factors.append("Some important fields are missing: " + ", ".join(missing_fields))

    if not factors:
        factors.append("No major risk signals from available fields")

    score = int(score)
    return {
        "risk_score": score,
        "risk_level": risk_level(score),
        "risk_factors": factors,
    }


def final_verdict(position: str, level: str) -> str:
    if position == "Below Market" and level in {"Low Risk", "Medium Risk"}:
        return "Good Deal"
    if position == "Below Market" and level == "High Risk":
        return "Suspiciously Low"
    if position == "Fair Range":
        return "Fair Price"
    if position == "Above Market":
        return "Overpriced"
    return "Price Range Only"


def car_summary(vehicle: dict[str, Any]) -> str:
    make = str(vehicle.get("make") or "Unknown make")
    model = str(vehicle.get("model") or "Unknown model")
    year = vehicle.get("registration_year")
    if is_missing(year) and not is_missing(vehicle.get("registration_date")):
        year = str(vehicle["registration_date"])[:4]
    mileage = to_float(vehicle.get("mileage_km"))

    parts = [f"{make} {model}".strip()]
    if not is_missing(year):
        parts.append(str(int(float(year))))
    if mileage is not None:
        parts.append(f"{mileage:,.0f} km")
    return " · ".join(parts)


def engineered_preview(artifacts: dict[str, Any], vehicle: dict[str, Any]) -> dict[str, Any]:
    preprocessor: PriceRangePreprocessor = artifacts["preprocessor"]
    features = build_feature_frame(pd.DataFrame([vehicle]), preprocessor._reference_date())
    row = features.frame.iloc[0].to_dict()
    keys = [
        "age_years",
        "mileage_per_year",
        "is_zero_mileage",
        "is_low_mileage",
        "is_nearly_new",
        "total_equipment_count",
    ]
    return {key: row.get(key) for key in keys if key in row}


@torch.no_grad()
def predict_from_artifacts(
    artifacts: dict[str, Any],
    vehicle_dict: dict[str, Any],
    listing_price: float | None = None,
) -> dict[str, Any]:
    model_input = sanitize_vehicle_input(vehicle_dict)
    x_num, x_cat = artifacts["preprocessor"].transform(pd.DataFrame([model_input]))

    device = artifacts["device"]
    x_num_tensor = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_tensor = torch.tensor(x_cat, dtype=torch.long, device=device)
    pred_log = artifacts["model"](x_num_tensor, x_cat_tensor).cpu().numpy()[0]
    p10, p50, p90 = [float(max(value, 0.0)) for value in np.expm1(pred_log)]

    listing_value = None if listing_price is None else float(listing_price)
    position = price_position(listing_value, p10, p90)
    risk = add_price_anomaly_risk(
        compute_risk_score(model_input),
        listing_price=listing_value,
        p10=p10,
    )

    return {
        "car_summary": car_summary(model_input),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "listing_price": listing_value,
        "price_position": position,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "risk_factors": risk["risk_factors"],
        "final_verdict": final_verdict(position, risk["risk_level"]),
        "engineered_features": engineered_preview(artifacts, model_input),
        "model_input": model_input,
    }


def predict_price_range(
    vehicle_dict: dict[str, Any],
    listing_price: float | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    device: str = "cpu",
) -> dict[str, Any]:
    artifacts = load_artifacts(model_dir=model_dir, device=device)
    return predict_from_artifacts(artifacts, vehicle_dict, listing_price=listing_price)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict vehicle market listing price range")
    parser.add_argument("--input-json", required=True, help="JSON string or JSON file path")
    parser.add_argument("--listing-price", type=float, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_price_range(
        load_input_json(args.input_json),
        listing_price=args.listing_price,
        model_dir=args.model_dir,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
