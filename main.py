import pandas as pd
import numpy as np
from author_death import AuthorDeath


def main(stage):
    model = AuthorDeath(data_path="/home/maksim/Data/Author_Death/data.csv", stage=stage)
    model.process_raw_data()
    X, y = model.process_data()
    model.analyze_regression(X, y)
    print(np.shape(X))
    print(np.shape(model.categorical_features))
    print(np.shape(model.numerical_features))

    return 1



if __name__ == '__main__':
    print("Stages are: \n1.raw_data\n2.processing\n3.train\n")
    main("train")

