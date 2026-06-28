from __future__ import annotations

import ast
import json
import pickle
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PRICE_COLUMN = "price"

UNKNOWN_TOKEN = "Unknown"
OTHER_TOKEN = "Other"

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "body_type",
    "fuel_category",
    "transmission",
    "country_code",
    "seller_type",
    "offer_type",
    "vehicle_type",
    "listing_status",
]

BOOLEAN_FEATURES = [
    "is_used",
    "is_new",
    "is_preregistered",
    "seller_is_dealer",
    "had_accident",
    "has_full_service_history",
    "is_rental",
    "non_smoking",
]

BASE_NUMERIC_FEATURES = [
    "mileage_km",
    "power_kw",
    "nr_prev_owners",
    "ratings_average",
    "ratings_count",
    "ratings_recommend_percentage",
]

IMPORTANT_NUMERIC_FIELDS = [
    "mileage_km",
    "power_kw",
    "nr_prev_owners",
    "ratings_average",
    "ratings_count",
    "ratings_recommend_percentage",
]

EQUIPMENT_COLUMNS = [
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
]

EXCLUDED_INPUT_COLUMNS = {
    "price",
    "price_net",
    "price_vat_rate",
    "price_currency",
    "price_negotiable",
    "price_tax_deductible",
    "vin",
    "id",
    "seller_company_name",
    "seller_name",
    "description",
    "street",
    "zip",
    "latitude",
    "longitude",
    "url",
    "listing_url",
    "ad_url",
}

MISSING_STRINGS = {"", "unknown", "nan", "none", "null", "n/a", "na", "<na>"}
TRUE_STRINGS = {"true", "1", "yes", "y", "t", "ja", "oui"}
FALSE_STRINGS = {"false", "0", "no", "n", "f", "nein", "nee"}


@dataclass
class FeatureFrame:
    frame: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]


def find_csv_in_zip(zip_path: str | Path) -> str:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Data archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]

    if not csv_names:
        raise FileNotFoundError(f"No CSV file found inside {zip_path}")

    preferred = [
        name for name in csv_names if Path(name).name == "autoscout24_dataset_20251108.csv"
    ]
    return preferred[0] if preferred else csv_names[0]


def extract_csv_from_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    csv_name = find_csv_in_zip(zip_path)
    output_path = extract_dir / Path(csv_name).name
    if not output_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(csv_name) as source, output_path.open("wb") as target:
                target.write(source.read())
    return output_path


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string")
    text = text.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    text = text.str.replace(r"[^\d,.\-]", "", regex=True)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def normalize_category_series(series: pd.Series) -> pd.Series:
    out = series.astype("string").fillna(UNKNOWN_TOKEN).str.strip()
    out = out.mask(out.str.lower().isin(MISSING_STRINGS), UNKNOWN_TOKEN)
    return out.astype("object")


def normalize_bool_value(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return 1.0
        if value == 0:
            return 0.0
    text = str(value).strip().lower()
    if text in TRUE_STRINGS:
        return 1.0
    if text in FALSE_STRINGS:
        return 0.0
    if text in MISSING_STRINGS:
        return np.nan
    return np.nan


def normalize_bool_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_bool_value).astype("float32")


def count_equipment_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return 0

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return len(parsed)
        if isinstance(parsed, (list, tuple, set)):
            return len(parsed)
    except (ValueError, SyntaxError):
        pass

    lower_text = text.lower()
    if "<li>" in lower_text:
        return lower_text.count("<li>")
    for separator in [";", "|", "\n", ","]:
        if separator in text:
            return len([part for part in text.split(separator) if part.strip()])
    return 1


def infer_reference_date(df: pd.DataFrame, source_name: str | None = None) -> date:
    for column in [
        "dataset_date",
        "snapshot_date",
        "scrape_date",
        "scraped_at",
        "listing_date",
        "created_at",
        "first_seen",
        "date",
    ]:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                return parsed.max().date()

    if source_name:
        match = re.search(r"(20\d{2})(\d{2})(\d{2})", source_name)
        if match:
            year, month, day = map(int, match.groups())
            return date(year, month, day)

    return date.today()


def clean_training_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COLUMN not in raw_df.columns:
        raise ValueError(f"Missing supervised target column: {PRICE_COLUMN}")

    df = raw_df.copy()
    df[PRICE_COLUMN] = clean_numeric_series(df[PRICE_COLUMN])
    df = df.dropna(subset=[PRICE_COLUMN])
    df = df[df[PRICE_COLUMN] >= 0]
    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Too few rows with a valid price: {len(df)}")
    return df


def _first_existing(columns: Iterable[str], df: pd.DataFrame) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None


def _registration_age_years(df: pd.DataFrame, reference_date: date) -> pd.Series:
    age = pd.Series(np.nan, index=df.index, dtype="float32")

    if "registration_date" in df.columns:
        registration_date = pd.to_datetime(df["registration_date"], errors="coerce")
        days = (pd.Timestamp(reference_date) - registration_date).dt.days
        age = days / 365.25

    year_source = _first_existing(["registration_year", "production_year"], df)
    if year_source is not None:
        year = clean_numeric_series(df[year_source])
        year_age = reference_date.year - year
        age = age.fillna(year_age)

    return age.clip(lower=0.25, upper=80)


def _listing_status(df: pd.DataFrame) -> pd.Series:
    status = pd.Series(UNKNOWN_TOKEN, index=df.index, dtype="object")
    if "is_used" in df.columns:
        status = status.mask(normalize_bool_series(df["is_used"]) == 1.0, "Used")
    if "is_new" in df.columns:
        status = status.mask(normalize_bool_series(df["is_new"]) == 1.0, "New")
    if "is_preregistered" in df.columns:
        status = status.mask(
            normalize_bool_series(df["is_preregistered"]) == 1.0,
            "Pre-registered",
        )
    return status


def build_feature_frame(raw_df: pd.DataFrame, reference_date: date | None = None) -> FeatureFrame:
    reference_date = reference_date or date.today()
    df = raw_df.copy()

    df["age_years"] = _registration_age_years(df, reference_date)

    if "mileage_km" in df.columns:
        mileage = clean_numeric_series(df["mileage_km"])
    else:
        mileage = pd.Series(np.nan, index=df.index)

    df["mileage_per_year"] = mileage / df["age_years"].clip(lower=0.25)
    df["log_mileage_km"] = np.log1p(mileage.clip(lower=0))
    df["log_mileage_per_year"] = np.log1p(df["mileage_per_year"].clip(lower=0))
    df["is_zero_mileage"] = (mileage == 0).astype("float32")
    df["is_low_mileage"] = (mileage <= 100).astype("float32")
    df["is_nearly_new"] = ((df["age_years"] <= 1) & (mileage <= 1000)).astype("float32")

    for column in EQUIPMENT_COLUMNS:
        count_column = f"{column}_count"
        if column in df.columns:
            df[count_column] = df[column].map(count_equipment_items).astype("float32")
        elif count_column in df.columns:
            df[count_column] = clean_numeric_series(df[count_column]).astype("float32")
        else:
            df[count_column] = np.nan
    df["total_equipment_count"] = df[
        [f"{column}_count" for column in EQUIPMENT_COLUMNS]
    ].sum(axis=1, min_count=1)

    df["listing_status"] = _listing_status(df)

    feature_df = pd.DataFrame(index=df.index)
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    numeric_candidates = [
        *BASE_NUMERIC_FEATURES,
        "age_years",
        "mileage_per_year",
        "log_mileage_km",
        "log_mileage_per_year",
        "is_zero_mileage",
        "is_low_mileage",
        "is_nearly_new",
        "equipment_comfort_count",
        "equipment_entertainment_count",
        "equipment_extra_count",
        "equipment_safety_count",
        "total_equipment_count",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            feature_df[column] = clean_numeric_series(df[column])
            numeric_columns.append(column)

    for column in BOOLEAN_FEATURES:
        missing_column = f"{column}_missing"
        if column in df.columns:
            values = normalize_bool_series(df[column])
            feature_df[missing_column] = values.isna().astype("float32")
            feature_df[column] = values.fillna(0.0).astype("float32")
        else:
            feature_df[missing_column] = 1.0
            feature_df[column] = 0.0
        numeric_columns.extend([column, missing_column])

    for column in IMPORTANT_NUMERIC_FIELDS:
        missing_column = f"{column}_missing"
        if column in df.columns:
            feature_df[missing_column] = df[column].isna().astype("float32")
        else:
            feature_df[missing_column] = 1.0
        numeric_columns.append(missing_column)

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            feature_df[column] = normalize_category_series(df[column])
            categorical_columns.append(column)

    numeric_columns = list(dict.fromkeys(numeric_columns))
    categorical_columns = list(dict.fromkeys(categorical_columns))

    leaked = (set(numeric_columns) | set(categorical_columns)) & EXCLUDED_INPUT_COLUMNS
    if leaked:
        raise ValueError(f"Leaking excluded input columns into features: {sorted(leaked)}")

    return FeatureFrame(feature_df, numeric_columns, categorical_columns)


@dataclass
class PriceRangePreprocessor:
    max_categories: int = 1000
    min_category_frequency: int = 2
    reference_date: str | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)
    cat_cardinalities: list[int] = field(default_factory=list)
    scaler: StandardScaler | None = None

    def fit(self, raw_df: pd.DataFrame) -> "PriceRangePreprocessor":
        ref_date = self._reference_date()
        features = build_feature_frame(raw_df, ref_date)
        self.numeric_columns = features.numeric_columns
        self.categorical_columns = features.categorical_columns

        numeric_df = self._numeric_dataframe(features.frame)
        if self.numeric_columns:
            medians = numeric_df.median(axis=0, skipna=True).fillna(0.0)
            self.numeric_medians = {
                column: float(medians[column]) for column in self.numeric_columns
            }
            filled = numeric_df.fillna(self.numeric_medians)
            self.scaler = StandardScaler()
            self.scaler.fit(filled.to_numpy(dtype=np.float32))
        else:
            self.numeric_medians = {}
            self.scaler = None

        self.vocabularies = {}
        self.cat_cardinalities = []
        for column in self.categorical_columns:
            series = self._category_series(features.frame, column)
            counts = series.value_counts(dropna=False)
            kept_values = []
            for value, count in counts.items():
                if value in {UNKNOWN_TOKEN, OTHER_TOKEN}:
                    continue
                if count < self.min_category_frequency:
                    continue
                kept_values.append(str(value))

            limit = max(self.max_categories - 2, 0)
            values = [UNKNOWN_TOKEN, OTHER_TOKEN, *kept_values[:limit]]
            mapping = {value: idx for idx, value in enumerate(values)}
            self.vocabularies[column] = mapping
            self.cat_cardinalities.append(len(mapping))

        return self

    def transform(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = build_feature_frame(raw_df, self._reference_date())
        return self.transform_feature_frame(features.frame)

    def fit_transform(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(raw_df)
        return self.transform(raw_df)

    def transform_feature_frame(self, feature_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_rows = len(feature_frame)

        if self.numeric_columns:
            numeric_df = self._numeric_dataframe(feature_frame)
            numeric_df = numeric_df.fillna(self.numeric_medians)
            numeric = numeric_df.to_numpy(dtype=np.float32)
            if self.scaler is not None:
                numeric = self.scaler.transform(numeric).astype(np.float32)
        else:
            numeric = np.zeros((n_rows, 0), dtype=np.float32)

        if self.categorical_columns:
            categorical = np.zeros((n_rows, len(self.categorical_columns)), dtype=np.int64)
            for idx, column in enumerate(self.categorical_columns):
                series = self._category_series(feature_frame, column)
                mapping = self.vocabularies[column]
                categorical[:, idx] = (
                    series.map(lambda value: mapping.get(str(value), mapping[OTHER_TOKEN]))
                    .fillna(mapping[UNKNOWN_TOKEN])
                    .astype(np.int64)
                    .to_numpy()
                )
        else:
            categorical = np.zeros((n_rows, 0), dtype=np.int64)

        return numeric, categorical

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "scaler.pkl").open("wb") as f:
            pickle.dump(self.scaler, f)
        with (output_dir / "preprocessor.pkl").open("wb") as f:
            pickle.dump(self, f)
        with (output_dir / "vocabularies.json").open("w", encoding="utf-8") as f:
            json.dump(self.vocabularies, f, indent=2, ensure_ascii=False)
        with (output_dir / "feature_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.to_config(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, model_dir: str | Path) -> "PriceRangePreprocessor":
        model_dir = Path(model_dir)
        preprocessor_path = model_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            with preprocessor_path.open("rb") as f:
                return pickle.load(f)

        with (model_dir / "feature_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        with (model_dir / "vocabularies.json").open("r", encoding="utf-8") as f:
            vocabularies = json.load(f)
        with (model_dir / "scaler.pkl").open("rb") as f:
            scaler = pickle.load(f)

        preprocessor = cls(
            max_categories=config["max_categories"],
            min_category_frequency=config["min_category_frequency"],
            reference_date=config["reference_date"],
        )
        preprocessor.numeric_columns = config["numeric_columns"]
        preprocessor.categorical_columns = config["categorical_columns"]
        preprocessor.numeric_medians = {
            key: float(value) for key, value in config["numeric_medians"].items()
        }
        preprocessor.vocabularies = {
            key: {str(token): int(index) for token, index in mapping.items()}
            for key, mapping in vocabularies.items()
        }
        preprocessor.cat_cardinalities = [
            len(preprocessor.vocabularies[column])
            for column in preprocessor.categorical_columns
        ]
        preprocessor.scaler = scaler
        return preprocessor

    def to_config(self) -> dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "categorical_cardinalities": self.cat_cardinalities,
            "numeric_medians": self.numeric_medians,
            "max_categories": self.max_categories,
            "min_category_frequency": self.min_category_frequency,
            "reference_date": self.reference_date,
            "unknown_token": UNKNOWN_TOKEN,
            "other_token": OTHER_TOKEN,
            "target_column": PRICE_COLUMN,
            "target_transform": "log1p",
            "excluded_input_columns": sorted(EXCLUDED_INPUT_COLUMNS),
        }

    def _reference_date(self) -> date:
        if self.reference_date is None:
            return date.today()
        return date.fromisoformat(self.reference_date)

    def _numeric_dataframe(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for column in self.numeric_columns:
            if column in feature_frame.columns:
                data[column] = clean_numeric_series(feature_frame[column])
            else:
                data[column] = pd.Series(np.nan, index=feature_frame.index)
        numeric_df = pd.DataFrame(data, index=feature_frame.index)
        for column in numeric_df.columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")
        return numeric_df.astype("float32")

    @staticmethod
    def _category_series(feature_frame: pd.DataFrame, column: str) -> pd.Series:
        if column in feature_frame.columns:
            return normalize_category_series(feature_frame[column])
        return pd.Series(UNKNOWN_TOKEN, index=feature_frame.index, dtype="object")
