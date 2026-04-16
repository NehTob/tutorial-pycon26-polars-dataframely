import polars as pl

from ._internal import Report
from .data import PreprocessedData
from .schema.report import (
    AverageCarVolumeSchema,
    PopularModelsSchema,
    SafestModelsSchema,
)


def build_report(prep: PreprocessedData) -> Report:
    return Report(
        popularity=find_three_most_popular_make_and_models(prep.models, prep.policies),
        safety=find_safest_models(prep.models),
        volume=find_average_car_volume_by_age(prep.models, prep.policies),
    )


def find_three_most_popular_make_and_models[T: (pl.DataFrame, pl.LazyFrame)](
    models: T, policies: T
) -> T:
    """Among all policies, compute the three make/model combinations that appears most often.

    Returns:
        A dataframe with three rows and three columns (make, model, count).
    """
    res = (
        policies.group_by("model")
        .len(name="count")
        .join(models.select("model", "make"), on="model")
        .sort("count", descending=True)
        .head(3)
        .select("make", "model", "count")
    )
    return PopularModelsSchema.validate(res, cast=True)


def find_safest_models[T: (pl.DataFrame, pl.LazyFrame)](models: T) -> T:
    """Among all models, find the safest ones as measured by the number of safety features.

    Returns:
        A data frame with five rows and three columns (model, segment, safety_score).
    """
    res = (
        models.with_columns(
            safety_score=pl.sum_horizontal(pl.col("^is_.*$")) + pl.col("airbags")
        )
        .sort("safety_score", descending=True)
        .head(5)
        .select("model", "segment", "safety_score")
    )
    return SafestModelsSchema.validate(res, cast=True)


def find_average_car_volume_by_age[T: (pl.DataFrame, pl.LazyFrame)](
    models: T, policies: T
) -> T:
    """Among all policies, find the mean physical car volume in 10-year blocks of car age.

    This method should compute the volume of a car if interpreted as cuboid (i.e. box-shaped).
    Blocks should be 0-10 years, 10-20 years, etc.

    Returns:
        A data frame with three columns (age block, mean volume in cubic meters,
        relative change of mean volume relative to the previous age block in percent).
    """

    # Volume is already in meters because of preprocessing!
    df_volume = models.select("model", "length", "width", "height").with_columns(
        volume=pl.col("length") * pl.col("width") * pl.col("height")
    )

    res = (
        policies.join(df_volume, on="model")
        .with_columns(
            age_block=pl.col("age_of_car").cut(
                breaks=[10, 20, 30, 40, 50],
                labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50+"],
            )
        )
        .group_by("age_block")
        .agg(mean_volume=pl.col("volume").mean())
        .sort("age_block")
        .with_columns(
            relative_change_pct=pl.col("mean_volume").pct_change() * 100,
        )
    )

    return AverageCarVolumeSchema.validate(res, cast=True)
