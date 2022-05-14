
from author_death import AuthorDeath


def main(stage):
    model = AuthorDeath(data_path="/home/maksim/Data/Author_Death/data.csv", stage=stage)
    model.process_raw_data()
    X, y = model.process_data()
    model.analyze_regression(X, y)

    return 1



if __name__ == '__main__':
    print("Stages are: \n1.raw_data\n2.processing\n3.analyzing\n")
    # main("raw_data")
    # main("processing")
    main("analyzing")

