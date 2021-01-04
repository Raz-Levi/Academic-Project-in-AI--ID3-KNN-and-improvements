import pandas as pd

Matrix = list


def get_data_to_matrix(path: str) -> Matrix:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    return [list(row[0:-1]) for row in data_frame.values]


def tdidt_algorithm():
    pass


def max_ig(features, examples):
    pass


def id3_algorithm():
    pass


def main():
    pass

if __name__ == "__main__":
    main()
