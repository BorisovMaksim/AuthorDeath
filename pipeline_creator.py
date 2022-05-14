from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression



class PipelineCreator:
    def __init__(self, numeric_impute_strategy, categorical_impute_strategy,
                 numerical_features, categorical_features):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_numeric_impute_strategy = categorical_impute_strategy
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def create(self):
        numeric_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy=self.numeric_impute_strategy)),
            ('scale', StandardScaler())
        ])
        categorical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy=self.categorical_numeric_impute_strategy)),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        full_processor = ColumnTransformer(transformers=[
            ('number', numeric_pipeline, self.numerical_features),
            ('category', categorical_pipeline, self.categorical_features)
        ])
        pipeline = Pipeline(steps=[('preprocessor', full_processor)])
        return pipeline
