import pandas as pd
import numpy as np
import statsmodels.api as sm
from pipeline_creator import PipelineCreator


def save_data(df, name):
    pd.to_pickle(df, f"/home/maksim/Data/Author_Death/{name}.pkl")


def load_data(name):
    return pd.read_pickle(f"/home/maksim/Data/Author_Death/{name}.pkl")


def log_price(y):
    mean = y.mean()
    return np.log(y.apply(lambda x: mean if x < 10 else x))


class AuthorDeath:
    def __init__(self, data_path, stage):
        self.data_path = data_path
        self.author_features = []
        self.art_features = []
        self.target = ""
        self.numerical_features = []
        self.categorical_features = []
        self.stage = stage

    def read_csv(self):
        return pd.read_csv(self.data_path, dtype='unicode')

    def process_raw_data(self):
        if self.stage != "raw_data":
            return 1

        df = self.read_csv()
        df = df.astype({'sale_date': "string", 'Price': float, 'Author': "string", 'Art': "string", 'Owner': "string",
                        'Sign': bool, 'Technique': "string",
                        'Material': "string", 'Nazi': bool, 'Framed': bool, 'Size': "string", 'square_m': float,
                        'Currency': "string",
                        'EstimateFrom': float, 'EstimateTo': float, 'ExhibitedNum': int, 'ProvenanceNum': int,
                        'LiteratureNum': int, 'CataloguingLength': float, 'Time': "string", 'City': "string",
                        'Description': "string",
                        'tried_url': "string", 'Image': "string", 'date_of_birth': "string", 'date_of_death': float,
                        'century': int,
                        'nationality': "string", 'sex': "string", 'style': "string", 'repeat_sale': bool,
                        'feature1': "string", 'feature2': "string",
                        'number_of_day': int, 'day_of_week': "string", 'month': int, 'year': int,
                        'deal_time_.utc.': "string",
                        'normalized_price': float, 'hasFollowers': bool, 'hasAfter': bool, 'mannerOf': bool,
                        'circleOf': bool,
                        'isUntitled': bool, 'isNumbered': bool, 'normalized_estimatefrom': float,
                        'normalized_estimateto': float, 'Auction': "string"})

        def try_convert(x):
            try:
                return float(x)
            except ValueError:
                return None

        df["date_of_birth"] = df["date_of_birth"].apply(lambda x: try_convert(x))
        df = df.drop_duplicates(subset=['Author', 'Art'])
        df = df.set_index(['Author', 'Art'])
        df = df.drop(columns=['sale_date', 'Price', 'EstimateFrom', 'EstimateTo', 'Time', 'City',
                              'Description', 'tried_url', 'Image', 'month', 'year', 'deal_time_.utc.',
                              'normalized_estimatefrom', 'normalized_estimateto', 'Size', 'number_of_day',
                              'day_of_week'], axis=1)
        df['Owner'] = df['Owner'].isna() == False
        save_data(df, "cleaned_data")

    def process_data(self, path_cleaned_data="/home/maksim/Data/Author_Death/cleaned_data.pkl"):
        if self.stage != "processing":
            self.categorical_features = load_data("categorical_features")
            self.numerical_features = load_data("numerical_features")
            return load_data("X"), load_data("y")
        df = pd.read_pickle(path_cleaned_data)
        self.numerical_features = df.select_dtypes(include=[int, float]).columns.tolist()
        self.categorical_features = df.select_dtypes(include="string").columns.tolist()
        self.target = 'normalized_price'
        y = log_price(df.normalized_price)
        X = df.fillna("to_replace").replace({"to_replace": None})
        pipeline_creator = PipelineCreator(numeric_impute_strategy='median',
                                           categorical_impute_strategy='most_frequent',
                                           numerical_features=self.numerical_features,
                                           categorical_features=self.categorical_features)
        pipeline = pipeline_creator.create()
        X = pipeline.fit_transform(X)

        self.categorical_features = pipeline['preprocessor'].transformers_[1][1]['one-hot'] \
            .get_feature_names_out(self.categorical_features)
        save_data(X, "X")
        save_data(y, "y")
        save_data(self.categorical_features, "categorical_features")
        save_data(self.numerical_features, "numerical_features")



        return X, y

    def analyze_regression(self, X, y):

        pass
