from pipeline.schema.preprocessed import PrepModelsSchema, PrepPoliciesSchema
from pipeline.schema.report import AverageCarVolumeSchema
from pipeline.report import find_average_car_volume_by_age
import polars as pl
from polars.testing import assert_frame_equal


def test_find_average_car_volume_by_age():
    # PrepModelsSchema expects dimensions in meters (1-10)
    models = PrepModelsSchema.sample(
        overrides=[
            {"model": "M1", "height": 1.5, "width": 2.0, "length": 2.5},
            {"model": "M2", "height": 2.0, "width": 2.0, "length": 2.0},
        ]
    )
    
    policies = PrepPoliciesSchema.sample(
        overrides=[
            {"model": "M1", "age_of_car": 4.5},
            {"model": "M2", "age_of_car": 14.5},
        ]
    )

    volume_m1 = 1.5 * 2.0 * 2.5 # 7.5
    volume_m2 = 2.0 * 2.0 * 2.0 # 8.0
    change = 100 * (volume_m2 / volume_m1 - 1)
    
    expected = AverageCarVolumeSchema.validate(
        pl.DataFrame(
            [
                {"age_block": "0-10", "mean_volume": volume_m1, "relative_change_pct": None},
                {"age_block": "10-20", "mean_volume": volume_m2, "relative_change_pct": change},
            ]
        ),
        cast=True,
    )

    df = find_average_car_volume_by_age(models, policies)

    assert_frame_equal(expected, df)
