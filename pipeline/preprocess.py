import polars as pl

from .data import PreprocessedData, RawData
from .schema.preprocessed import PrepModelsSchema, PrepPoliciesSchema


def preprocess(raw: RawData) -> PreprocessedData:
    return PreprocessedData(
        policies=raw.policies.pipe(preprocess_policies),
        models=raw.models.pipe(preprocess_models),
    )


def preprocess_policies[T: (pl.DataFrame, pl.LazyFrame)](policies: T) -> T:
    """Transform the raw policies for optimal representation."""
    df = policies.with_columns(
        pl.col("model").cast(pl.Categorical),
        pl.col("area_cluster").cast(pl.Categorical),
        # Float columns often do not need full 64-bit precision
        # This depends on the domain we are working on
        pl.col("policy_tenure").cast(pl.Float32),
        pl.col("age_of_car").cast(pl.Float32),
        pl.col("age_of_policyholder").cast(pl.Float32),
        pl.col("population_density").cast(pl.Float32),
        policy_id=pl.col("policy_id").str.strip_prefix("policy").cast(pl.UInt64),
    )
    return PrepPoliciesSchema.validate(df, cast=True)


def preprocess_models[T: (pl.DataFrame, pl.LazyFrame)](models: T) -> T:
    """Transform the raw models for optimal representation."""

    # Since 'model' is the primary key in our prep schema, we must ensure it is unique.
    df = models.unique(subset=["model"])

    df = df.with_columns(
        pl.col("^is_.*$").eq("Yes"),
    )

    torque_parts = pl.col("max_torque").str.split("@")
    df = df.with_columns(
        max_torque_nm=torque_parts.list[0].str.strip_suffix("Nm").cast(pl.Float32),
        max_torque_rpm=torque_parts.list[1].str.strip_suffix("rpm").cast(pl.UInt16),
    )

    power_parts = pl.col("max_power").str.split("@")
    df = df.with_columns(
        max_power_bhp=power_parts.list[0].str.strip_suffix("bhp").cast(pl.Float32),
        max_power_rpm=power_parts.list[1].str.strip_suffix("rpm").cast(pl.UInt16),
    )

    # We saw width, height, length are either around 150-180 (cm) or 1500-1800 (mm)
    # A simple heuristic: if > 500 it is mm, else cm.
    def to_meters(col_name: str):
        return (
            pl.when(pl.col(col_name) > 500)
            .then(pl.col(col_name) / 1000)
            .otherwise(pl.col(col_name) / 100)
            .cast(pl.Float32)
        )

    df = df.with_columns(
        pl.col("steering_type").cast(pl.Enum(["Manual", "Power", "Electric"])),
        pl.col("fuel_type").cast(pl.Enum(["CNG", "Petrol", "Diesel"])),
        pl.col("rear_brakes_type").cast(pl.Enum(["Disc", "Drum"])),
        # For other categoricals, we may not be sure yet that we have seen all values
        # so we do not want to commit to an Enum, yet
        pl.col("engine_type").cast(pl.Categorical),
        pl.col("model").cast(pl.Categorical),
        pl.col("segment").cast(pl.Categorical),
        width=to_meters("width"),
        height=to_meters("height"),
        length=to_meters("length"),
        displacement=pl.col("displacement").cast(pl.UInt16),
        cylinder=pl.col("cylinder").cast(pl.UInt8),
        gross_weight=pl.col("gross_weight").cast(pl.UInt16),
        gear_box=pl.col("gear_box").cast(pl.UInt8),
        airbags=pl.col("airbags").cast(pl.UInt8),
    )

    return PrepModelsSchema.validate(df, cast=True)
