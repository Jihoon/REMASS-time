import pymrio
from pathlib import Path
import util
import pandas as pd

# show version of pymrio
print(pymrio.__version__)
test_mrio = pymrio.load_test()


# Load EXIOBASE3 data
exio3_folder = "H:\MyDocuments\Data\EXIOBASE3"

exio_downloadlog = pymrio.download_exiobase3(
    storage_folder=exio3_folder, 
    system="pxp", 
    years=[2022]
)

p_2022 = Path(exio3_folder) / "IOT_2022_pxp.zip"
p_2020 = Path(exio3_folder) / "IOT_2020_pxp.zip"
print(p_2022.exists())

exio3 = pymrio.parse_exiobase3(path=p_2022)
exio3.calc_system() # This generates all including L.

# Look at the metadata of the EXIOBASE3 system
print(exio3.meta)
reg = exio3.get_regions()


# Sector indices for agriculture, processing, and food services
sec = exio3.get_sectors()
agri_ind = list(range(0, 14)) + [18] 
proc_ind = list(range(42, 53))
svc_ind = [155]

agri_sectors = sec[agri_ind]
proc_sectors = sec[proc_ind]
svc_sectors = sec[svc_ind]

# Combined index for all food sectors
food_sectors = sec[agri_ind + proc_ind + svc_ind]

print(agri_sectors)
print(proc_sectors)
print(svc_sectors)


# Partition Y matrix into food and non-food sectors
Y_food = exio3.Y.copy()
Y_nonfood = exio3.Y.copy()

# Set all row values to zero for 'sector' name match foodsectors
Y_food.loc[~Y_food.index.get_level_values('sector').isin(food_sectors), :] = 0
Y_nonfood.loc[Y_food.index.get_level_values('sector').isin(food_sectors), :] = 0


# Keep only the final demand categories that are related to food consumption
fd_cat = exio3.Y.columns.get_level_values('category').unique()
# Extract categories containing 'Final consumption' from fd_cat
fd_cat = [cat for cat in fd_cat if 'Final consumption' in cat]
Y_fd, Y_t = util.get_Y_agg(Y_food, fd_cat)


# Aggregate male vs. female employment hours
exio3.emp_gender = exio3.employment.copy()
exio3.emp_gender.name = "Aggregated Employment Stress by Gender"
# Keep only the employment categories containing 'hours'
exio3.emp_gender.F = exio3.emp_gender.F.loc[
    exio3.emp_gender.F.index.str.contains('hours')
]
# Separate the last word of the index as a new column, and the rest as another column
exio3.emp_gender.F['stressor'] = exio3.emp_gender.F.index.str.split().str[-1]
exio3.emp_gender.F['skill'] = exio3.emp_gender.F.index.str.split().str[:-1].str.join(' ')
exio3.emp_gender.F = exio3.emp_gender.F.set_index(['skill', 'stressor'])

# Partial sum by gender for employment hours
exio3.emp_gender.F = exio3.emp_gender.F.groupby('stressor').sum()
print("Partial sum by gender (employment hours):")
print(exio3.emp_gender.F)

exio3.emp_gender.unit = exio3.emp_gender.unit.iloc[0:2, :]  # Keep only the first row for units
exio3.emp_gender.unit.index = exio3.emp_gender.F.index  # Set the index of the unit to match the employment stressor index



# TODO: Need to decide whather to handle all six employment categories or just the two genders



# Add employment stressors for the 6 employment categories
exio3.emp = []
emp_ind = exio3.emp_gender.get_index()

# Calculate the employment stressors and impacts (D_xx) for each category (Entire economy)
# This gives a memory error if I do all six labor categories at once.
for i in range(0,2):
    exio3.emp.append(exio3.emp_gender.diag_stressor(emp_ind[i], name=emp_ind[i]))   
    print(f"Created employment stressor for category {i}: {emp_ind[i]}")
    # exio3.emp[i].calc_system(exio3.x, Y=Y_food) # Without L and F_Y, this generates only the S matrix.
    exio3.emp[i].calc_system(exio3.x, Y=Y_food, Y_agg=Y_fd , L=exio3.L) 
    
    # Without L and F_Y, calc_system() generates only the S matrix.    
    # With L and F_Y, this generates all of M, D_cba, D_pba, D_imp, D_exp, unit, D_cba_reg, D_pba_reg, D_imp_reg, D_exp_reg
    
    print(exio3.emp[i].name)
    print(str(exio3.emp[i]))
# => calc_system aggregates the given Y_food param and diagonalizes it to a square Y with 'diagonalize_columns_to_sectors(Y_agg)'.

# # Use calc_accounts to avoid the memory error
# for i in range(0,6):
#     a,b,c,d = pymrio.calc_accounts(S=exio3.emp[i].S, L=exio3.L, Y=Y_fd) # This does not work, as it requires the calc_system to be run first.
#     exio3.emp[i].D_cba = a
#     exio3.emp[i].D_pba = b
#     exio3.emp[i].D_imp = c
#     exio3.emp[i].D_exp = d

#     print(exio3.emp[i].name)
#     print(str(exio3.emp[i]))


print(exio3.emp[0].D_cba) # Impact of individual sectors and country/region
print(exio3.emp[0].D_cba_reg) # Sum by country/region


# Impact plots
for i in range(0,2):
    # exio3.emp[i].plot_account(row=('WF', 'Vegetables, fruit, nuts'))
    exio3.emp[i].plot_account(row=('US', 'Vegetables, fruit, nuts'))
    exio3.emp[i].plot_account(row=('US', svc_sectors.values[0]))
    # exio3.emp[i].plot_account(row=('US', 'Vegetables, fruit, nuts'))

    # print(f"Plotting D_cba for {exio3.emp[i].D_cba.loc[row]}")
    # print(f"Plotting D_pba for {exio3.emp[i].D_pba.loc[row]}")
    # print(f"Plotting D_imp for {exio3.emp[i].D_imp.loc[row]}")
    # print(f"Plotting D_exp for {exio3.emp[i].D_exp.loc[row]}")

# Test-view 
exio3.employment.calc_system(exio3.x, Y_food, L=exio3.L)
mat_S = exio3.employment.S.iloc[0,:]
mat_M = exio3.employment.M.iloc[0,:]
Y_food_HH = Y_food.loc[:, Y_food.columns.get_level_values('category').isin(['Final consumption expenditure by households'])]



# Get the distribution of values in the cells of exio3.emp[0].D_cba
import numpy as np
mat = exio3.emp[0].D_exp
values = mat.values.flatten()
# Calculate the distribution of values
values_distribution = np.histogram(values, bins=100)
# Print the distribution
print("Distribution of values:")
print(values_distribution)

# Find the index of the maximum value in exio3.emp[0].D_cba
max_index = np.unravel_index(np.argmax(mat.values, axis=None), mat.shape)
# Print the maximum value and its index
max_value = mat.values[max_index]
print(f"Maximum value: {max_value} at index {max_index}")
mat.index[max_index[0]], mat.columns[max_index[1]]



# Visualize
import matplotlib.pyplot as plt

m = exio3.emp[0].D_imp #.loc[:, Y_fd.columns.get_level_values('region').isin(["WF"])]
plt.figure(figsize=(15, 15))
plt.imshow(m)
plt.colorbar()
plt.xlabel("Countries - sectors")
plt.ylabel("Countries - M.hours")
plt.show()









# Note: I tried to adopt the idea of Rasul et al. (2024) to partition the matrix and create the HEM matrices.
# However, I found that the EROEI idea may not direactly be applicable to my case, as I am not calculating the EROEI but rather the labor going into the final consumption of food. 
# So I don't need the blue part (refer to the email to Edgar).

# For the HEM matrices, we need to partition the matrix and create the HEM matrices
A_mat = exio3.A.copy()

# Get the indices of the food sectors
food_loc = A_mat.index.get_level_values('sector').isin(food_sectors).tolist()
nonfood_loc = [not elem for elem in food_loc]
# Get the shape information of food_loc and nonfood_loc
print(f"Food sectors: {sum(food_loc)}")

# Get the multiindices of the food sectors in the columns
food_ind = A_mat.index[food_loc]
nonfood_ind = A_mat.index[nonfood_loc]


# Create HEM matrices
A11, A12, A21, A22 = util.partition_matrix(A_mat, food_loc)
H, dL = util.HEM_matrices_1aI(A_mat, food_loc)

# Calculate the delta x with the new dL matrix
# dL needs to be reordered to match the order of the original L
# Convert dL to a DataFrame with the same index as exio3.L
import pandas as pd
new_ind = food_ind.append(nonfood_ind) # All index for the partitioned order of the matrix
dL_df = pd.DataFrame(dL, index=new_ind, columns=new_ind)

# Reorder the dL matrix to match the original L or A matrix
dL_ord = dL_df.reindex(index=exio3.A.index, columns=exio3.A.columns)

# Calculate the delta x using the new dL matrix
dx = dL_ord @ exio3.Y
