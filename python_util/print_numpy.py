import numpy as np
import sys

def print_npy_file(file):
    # Load numpy array
    array = np.load(file)

    print(array.shape)

    # Print the array
    print(array)

if __name__ == "__main__":
    # sys.argv[0] is the script name itself, so we start from index 1
    file = sys.argv[1]

    print_npy_file(file)

