from utils import get_N_matrix
import pandas as pd


def main():
    N = get_N_matrix('Ukrainekonflikt')
    print(N)


if __name__ == "__main__":
    main()
