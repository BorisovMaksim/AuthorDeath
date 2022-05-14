import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pipeline_creator import PipelineCreator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from myconstants import cols_to_drop
import statsmodels.stats.api as sms
from statsmodels.compat import lzip


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
                        'Sign': int, 'Technique': "string",
                        'Material': "string", 'Nazi': int, 'Framed': int, 'Size': "string", 'square_m': float,
                        'Currency': "string",
                        'EstimateFrom': float, 'EstimateTo': float, 'ExhibitedNum': int, 'ProvenanceNum': int,
                        'LiteratureNum': int, 'CataloguingLength': float, 'Time': "string", 'City': "string",
                        'Description': "string",
                        'tried_url': "string", 'Image': "string", 'date_of_birth': "string", 'date_of_death': float,
                        'century': int,
                        'nationality': "string", 'sex': "string", 'style': "string", 'repeat_sale': int,
                        'feature1': "string", 'feature2': "string",
                        'number_of_day': int, 'day_of_week': "string", 'month': int, 'year': int,
                        'deal_time_.utc.': "string",
                        'normalized_price': float, 'hasFollowers': int, 'hasAfter': int, 'mannerOf': int,
                        'circleOf': int,
                        'isUntitled': int, 'isNumbered': int, 'normalized_estimatefrom': float,
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
        self.numerical_features = df.select_dtypes(include=[int, float, bool]).columns.tolist()
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
        if self.stage != "analyzing":
            return 1
        all_cols = np.append(np.array(self.numerical_features), self.categorical_features)
        # X = pd.DataFrame(X, columns=all_cols, index=y.index).drop(cols_to_drop, axis=1)
        X = pd.DataFrame(X, columns=all_cols, index=y.index).drop(self.categorical_features, axis=1).\
            drop(['normalized_price', 'date_of_death'], axis=1)
        X = sm.add_constant(X)
        print((X['hasFollowers']).sum())
        X['square_m'] = np.log(abs(X['square_m']))
        X['ExhibitedNum'] = np.log(abs(X['ExhibitedNum']))
        X['ProvenanceNum'] = np.log(abs(X['ProvenanceNum']))
        X['LiteratureNum'] = np.log(abs(X['LiteratureNum']))
        X['CataloguingLength'] = np.log(X['CataloguingLength'] + 1)

        model = sm.OLS(y, X)
        results = model.fit()
        print(results.summary())
        regression = Model(results, X, y)
        regression.check_linearity()
        # regression.check_normality_errors()
        # regression.vif_test()
        # regression.het_breuschpagan()




class Model:
    def __init__(self, results, X, y):
        self.results = results
        self.X = X
        self.y = y

    def check_linearity(self):
        col = 'square_m'
        # for col in X.select_dtypes()
        plt.scatter(self.X[col], self.y)
        fig = plt.figure(figsize=(12, 8))
        sm.graphics.plot_regress_exog(self.results, col, fig=fig)
        plt.show()

        # line_coords = np.arange(self.y.min(), self.y.max())
        # plt.scatter(predictions, self.y)
        # plt.ylabel("Labels")
        # plt.xlabel("Predictions")
        # plt.plot(line_coords, line_coords, color='darkorange', linestyle='--', linewidth=3)
        # plt.savefig('/home/maksim/PycharmProjects/pythonProject/plots/plot_linearity.png', bbox_inches='tight')
        # plt.close()


    def vif_test(self):
        vif = pd.DataFrame([variance_inflation_factor(self.X.values, i)
                            for i in range(len(self.X.columns))], self.X.columns)
        print(vif[vif.values > 5] if not vif[vif.values > 5].empty else "All vif's < 5")

    def get_significant_features(self, results, level):
        return pd.DataFrame(results.pvalues[results.pvalues < level]).index.tolist()

    def check_normality_errors(self):
        plt.hist(self.results.predict(self.X))
        plt.show()
        plt.close()
        errors = self.results.resid
        plt.hist(errors, bins=40)
        plt.xlabel("Error")
        plt.ylabel("Occurrences")
        plt.savefig('/home/maksim/PycharmProjects/pythonProject/plots/histogram.png', bbox_inches='tight')
        plt.close()
        stats.probplot(errors, dist="norm", plot=plt)
        plt.savefig('/home/maksim/PycharmProjects/pythonProject/plots//qq-plot.png', bbox_inches='tight')
        plt.close()




    def het_breuschpagan(self):
        names = ['Lagrange multiplier statistic', 'p-value',
                 'f-value', 'f p-value']
        test = sms.het_breuschpagan(self.results.resid, self.results.model.exog)
        print(lzip(names, test)[1][1])