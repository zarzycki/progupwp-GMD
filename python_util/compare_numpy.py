import sys
import numpy as np

def compare_npy_files(file1, file2):
    # Load numpy arrays
    array1 = np.load(file1)
    array2 = np.load(file2)

    # Check if the two arrays are identical
    if np.array_equal(array1, array2):
        print("The arrays are identical")
    else:
        print("The arrays are not identical")

        # Check how they differ
        # The following line will return a boolean mask of the same shape as array1 that is True where array1 and array2 are not equal
        diff_mask = array1 != array2

        print("Differing elements:")
        print(array1[diff_mask])

        print("Indices of differing elements:")
        print(np.where(diff_mask))

if __name__ == "__main__":
    # sys.argv[0] is the script name itself, so we start from index 1
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    compare_npy_files(file1, file2)

