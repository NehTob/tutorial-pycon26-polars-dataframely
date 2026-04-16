import dataframely as dy


class PopularModelsSchema(dy.Schema):
    make = dy.UInt64()
    model = dy.Categorical()
    count = dy.UInt32()


class SafestModelsSchema(dy.Schema):
    model = dy.Categorical()
    segment = dy.Categorical()
    safety_score = dy.UInt8()


class AverageCarVolumeSchema(dy.Schema):
    age_block = dy.Categorical()
    mean_volume = dy.Float32()
    relative_change_pct = dy.Float32(nullable=True)
