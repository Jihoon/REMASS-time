# make a 100x100 matrix with random entries
import numpy as np
matrix = np.random.rand(100, 100)
# print the shape of the matrix
print("Shape of the matrix:", matrix.shape)

# make a random index vector for the matrix with length of 20
import random
random.seed(42)  # For reproducibility
index_vector = random.sample(range(100), 20)

def partition_matrix(A_mat, index_vector):
    """
    Partitions the matrix into four submatrices based on the index vector.
    One with the rows and columns indexed by the index vector (A11);
    one with the indexed rows and non-index columns (A12);
    one with the non-indexed rows and indexed columns (A21);
    and one with the remaining rows and columns (A22).
    """
    A11 = A_mat[np.ix_(index_vector, index_vector)]
    A12 = A_mat[np.ix_(index_vector, [i for i in range(A_mat.shape[1]) if i not in index_vector])]
    A21 = A_mat[np.ix_([i for i in range(A_mat.shape[0]) if i not in index_vector], index_vector)]
    A22 = A_mat[np.ix_([i for i in range(A_mat.shape[0]) if i not in index_vector], [i for i in range(A_mat.shape[1]) if i not in index_vector])]
    
    return A11, A12, A21, A22

def HEM_matrices_2aI(matrix, index_vector):
    """
    Computes the HEM matrices from the partitioned submatrices.
    Case 2a.I (Hertwich, 2024)
    """
    A11, A12, A21, A22 = partition_matrix(matrix, index_vector)
    
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
    L0 = np.bmat([[H - L11,         H @ A12 @ L22],
                 [L22 @ A21 @ H,    L22 @ A21 @ H @ A12 @ L22]])

    return H, L0

def HEM_matrices_1aI(matrix, index_vector):
    """
    Computes the HEM matrices from the partitioned submatrices.
    Case 2a.I (Hertwich, 2024)
    """
    A11, A12, A21, A22 = partition_matrix(matrix, index_vector)
    
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
    L0 = np.bmat([[H - I11,         H @ A12 @ L22],
                 [L22 @ A21 @ H,    L22 @ A21 @ H @ A12 @ L22]])

    return H, L0


# Then we can call the function with exio.A and foodsectors
H, L0 = HEM_matrices_2aI(exio.A, foodsectors)
exio.L0 = L0
exio.calc_system(H, L0)



