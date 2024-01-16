"""
Parameters to change (F = factor, N = numerical) :
 - dim. of matrix (F)
 - sparsity (F)
 - elements are integers or floats (F)
 - compute on old/modern CPU (F)
 - script VS. ipynb (or normal mode vs. debugging mode) (F)
 - random number generator: generate a ~ U(0,1) and then get random value via CDF of N()

"""

import numpy as np
import scipy as scp
import time


def generate_random_matrix(N, is_integer=True, MAXVAL=10):
    """
    Generate an NxN square matrix filled with random integers or floats.

    Parameters:
    - N: int, size of the square matrix.
    - is_integer: bool, if True, fill the matrix with random integers; otherwise, fill with random floats.
    - MAXVAL: int, the maximum value for random integers (ignored if is_integer is False).

    Returns:
    - A NumPy array representing the generated matrix.
    """
    if is_integer and MAXVAL is None:
        raise ValueError("If is_integer is True, MAXVAL must be specified for the range of random integers.")

    if is_integer:
        # Generate a matrix of random integers
        matrix = np.random.randint(1, MAXVAL + 1, size=(N, N))
    else:
        # Generate a matrix of random floats
        matrix = np.random.rand(N, N)

    return matrix


def generate_square_matrix(N, a, b, steps=True):
    """
    Args:
        N: dim of mtx
        a: lower bound for random elements
        b: upper bound for random elements
        steps: (bool) True=[random INT between a and b], False=[random FLOAT between a and b]

    Returns: square matrix

    """
    if steps:
        # Generate matrix with random integers between a and b
        matrix = np.random.randint(a, b + 1, size=(N, N))
    else:
        # Generate matrix with random floats between a and b
        matrix = np.random.uniform(a, b, size=(N, N))



    return matrix


def sparsify_mtx(matrix, threshold=1e-10):
    """
    Replace most elements of the matrix with zeros, ensuring the resulting matrix is not singular.
    Parameters:
    - matrix: 2D NumPy array
    - threshold: The threshold value for checking singularity (default is 1e-10)
    Returns:
    - modified_matrix: 2D NumPy array with most elements replaced by zeros
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")

    modified_matrix = matrix.copy()
    while True:
        # select random place in matrix_temp
        random_row_index = np.random.randint(0, matrix.shape[0])
        random_column_index = np.random.randint(0, matrix.shape[1])
        # assign 0 there and remember this matrix_temp
        matrix[random_row_index, random_column_index] = 0
        # check for det(matrix_temp): if it's close to 0, break out
        if np.abs(np.linalg.det(matrix)) < threshold:
            break
        # if it isn't, then matrix = matrix_temp
        modified_matrix = matrix.copy()

    return modified_matrix


def make_upper_triangular(matrix):
    # Ensure the input is a NumPy array for efficient operations
    matrix = np.array(matrix)

    # Extract the upper triangular part of the matrix
    upper_triangle = np.triu(matrix)

    # Create a new matrix with zeros above the diagonal
    result_matrix = np.zeros_like(matrix)
    result_matrix[np.triu_indices_from(result_matrix)] = upper_triangle[np.triu_indices_from(upper_triangle)]

    return result_matrix

def determ_time(matrix, ntimes: int = 3, type_of_result: str = 'med', sleep_pause:int = 0):
    """
    function that computes `det(matrix)` for `ntimes` times, measures execution time for each loop,
    and returns median time for the operation. (built on NumPy)
    :return: median_time .... [float]
    """
    vector_of_exec_times = []
    for _ in range(ntimes):
        time.sleep(sleep_pause)
        # Measure time needed to compute determinant
        start_time = time.time()
        determinant = scp.linalg.det(matrix)
        end_time = time.time()

        # Save the time taken for determinant computation
        det_time = end_time - start_time
        vector_of_exec_times.append(np.float64(det_time))

    if type_of_result == "med":
        result_time = np.median(vector_of_exec_times)
    elif type_of_result == "ave":
        result_time = np.mean(vector_of_exec_times)
    else:
        print("Unknown type_of_result in function 'determ_time'! ")
        return -100

    return result_time, vector_of_exec_times


if __name__ == "__main__":
    # Example usage:
    N = 100
    MAXVAL = 10
    integer_matrix = generate_random_matrix(N, is_integer=True, MAXVAL=MAXVAL)
    float_matrix = generate_random_matrix(N, is_integer=False)

    # print("Random Integer Matrix:")
    # print(integer_matrix)
    #
    # print("\nRandom Float Matrix:")
    # print(float_matrix)
    #
    # modified_matrix = sparsify_mtx(integer_matrix)
    # print(modified_matrix)
    # print(f"det(sparse_integer_matrix) = {np.linalg.det(modified_matrix)}")
    #
    modified_matrix = sparsify_mtx(integer_matrix)
    # print(modified_matrix)
    # print(f"det(sparse_float_matrix) = {np.linalg.det(modified_matrix)}")
    ntimes = 7
    print("INTEGER >> ")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(integer_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(integer_matrix, ntimes, type_of_result='ave')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(modified_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(modified_matrix, ntimes, type_of_result='ave')}")

    print("\nFLOAT >> ")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='ave')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='ave')}")

    N = 5
    MAXVAL = 10
    integer_matrix = generate_random_matrix(N, is_integer=True, MAXVAL=MAXVAL)
    float_matrix = generate_random_matrix(N, is_integer=False, MAXVAL=MAXVAL)
    print("INTEGER >> ")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(integer_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(integer_matrix, ntimes, type_of_result='ave')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(modified_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(modified_matrix, ntimes, type_of_result='ave')}")

    print("\nFLOAT >> ")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='ave')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='med')}")
    print(f"exec time for {N}-dim. SPARSIFIED mtx run for {ntimes} times is: {determ_time(float_matrix, ntimes, type_of_result='ave')}")













































