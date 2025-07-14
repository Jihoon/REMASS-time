# make a 100x100 matrix with random entries
import numpy as np
# matrix = np.random.rand(100, 100)
# # print the shape of the matrix
# print("Shape of the matrix:", matrix.shape)

# # make a random index vector for the matrix with length of 20
# import random
# random.seed(42)  # For reproducibility
# index_vector = random.sample(range(100), 20)

def partition_matrix(A_mat, index_vector):
    """
    Partitions the input dataframe into four submatrices based on the index vector.
    The input matrix A_mat is expected to be a pandas DataFrame with a multi-index.
    The index_vector is a list of indices that will be used to partition the matrix.
    
    The function returns four submatrices:
    One with the rows and columns indexed by the index vector (A11);
    one with the indexed rows and non-index columns (A12);
    one with the non-indexed rows and indexed columns (A21);
    and one with the remaining rows and columns (A22).
    """
    inv_index_vector = [not elem for elem in index_vector]
    A11 = A_mat.to_numpy()[np.ix_(index_vector, index_vector)]
    A12 = A_mat.to_numpy()[np.ix_(index_vector, inv_index_vector)]
    A21 = A_mat.to_numpy()[np.ix_(inv_index_vector, index_vector)]
    A22 = A_mat.to_numpy()[np.ix_(inv_index_vector, inv_index_vector)]
    
    return A11, A12, A21, A22

def HEM_matrices_2aI(A_mat, index_vector):
    """
    Computes the HEM matrices from the partitioned submatrices.
    Case 2a.I (Hertwich, 2024)
    """
    A11, A12, A21, A22 = partition_matrix(A_mat, index_vector)
    
    # Make an identity matrix of the same size as A22    
    I11 = np.eye(A11.shape[0])
    I22 = np.eye(A22.shape[0])
    
    # Inverse A22 matrix
    L11 = np.linalg.inv(I11 - A11) if np.linalg.det(I11 - A11) != 0 else None
    L22 = np.linalg.inv(I22 - A22) if np.linalg.det(I22 - A22) != 0 else None
    
    # Create H matrix by inverse(I − A11 − A12L22A21)
    h = np.eye(A11.shape[0]) - A11 - A12 @ L22 @ A21
    H = np.linalg.inv(h) if np.linalg.det(h) != 0 else None
    
    # Create matrix by concatenating H − L11, H@A12@L22, L22@A21@H, L22@A21@H@A12@L22 using numpy.bmat
    dL = np.bmat([[H - L11,         H @ A12 @ L22],
                 [L22 @ A21 @ H,    L22 @ A21 @ H @ A12 @ L22]])

    return H, dL

def HEM_matrices_1aI(A_mat, index_vector):
    """
    Computes the HEM matrices from the partitioned submatrices.
    Case 2a.I (Hertwich, 2024)
    """
    A11, A12, A21, A22 = partition_matrix(A_mat, index_vector)
    
    # Make an identity matrix of the same size as A22    
    I11 = np.eye(A11.shape[0])
    I22 = np.eye(A22.shape[0])
    
    # Inverse A22 matrix
    # L11 = np.linalg.inv(I11 - A11) if np.linalg.det(I11 - A11) != 0 else None
    L22 = np.linalg.inv(I22 - A22) if np.linalg.det(I22 - A22) != 0 else None
    
    # Create H matrix by inverse(I − A11 − A12L22A21)
    h = np.eye(A11.shape[0]) - A11 - A12 @ L22 @ A21
    H = np.linalg.inv(h) if np.linalg.det(h) != 0 else None
    
    # Create matrix by concatenating H − L11, H@A12@L22, L22@A21@H, L22@A21@H@A12@L22 using numpy.bmat
    dL = np.bmat([[H - I11,         H @ A12 @ L22],
                 [L22 @ A21 @ H,    L22 @ A21 @ H @ A12 @ L22]])

    return H, dL


# # Then we can call the function with exio.A and foodsectors
# H, dL = HEM_matrices_2aI(exio.A, foodsectors)
# exio.dL = dL
# exio.calc_system(H, dL)


def get_Y_agg(Y, keep=None):
    """
    Returns a diagonalized version of the input matrix Y, which is expected to be a pandas DataFrame with a multi-index.
    The function aggregates the data by region and then diagonalizes the columns to sectors.
    """
    # Importing the necessary module for diagonalization    
    import pymrio.tools.ioutil as ioutil

    # These are exerpts from pymrio code (MRIOSystem.calc_system())
    # https://github.com/IndEcol/pymrio/blob/aa3a67a5d4900595a270dac9423efbb82cdf79fd/pymrio/core/mriosystem.py#L1037
    idx = Y.T.index
    # Keep only rows if specified
    if keep is not None:
        Y_T = Y.T.loc[idx.get_level_values('category').isin(keep), :]

    # Sum the values in the DataFrame
    Y_agg = Y_T.groupby(level="region", sort=False).sum().T

    # Y_diag = ioutil.diagonalize_columns_to_sectors(Y_agg)
    return Y_agg, Y_T
