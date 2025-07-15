import pymrio
import util
import pandas as pd


## show version of pymrio
print(pymrio.__version__)
test_mrio = pymrio.load_test()


## Download EXIOBASE3 data
exio3_folder = "H:\MyDocuments\Data\EXIOBASE3"

exio_downloadlog = pymrio.download_exiobase3(
    storage_folder=exio3_folder, 
    system="pxp", 
    years=[1995, 2022]
)


## Load EXIOBASE3 data
exio3 = util.LoadEXIOBASE3(year=2022, system="pxp")


## Sector indices for agriculture, processing, and food services
reg = exio3.get_regions()
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


## Prep final demand matrix for the analysis
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


## Aggregate male vs. female employment hours in the extension
# Note: ignore F_Y here
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

# Modify unit to match the new index
exio3.emp_gender.unit = exio3.emp_gender.unit.iloc[0:2, :]  # Keep only the first row for units
exio3.emp_gender.unit.index = exio3.emp_gender.F.index  # Set the index of the unit to match the employment stressor index


## TODO: Need to decide whether to handle all six employment categories or just the two genders


## Add employment stressors for the 2 gender categories
exio3.emp = []
emp_ind = exio3.emp_gender.get_index()

# Calculate the employment stressors and impacts (D_xx) for each gender labor hours (Entire economy)
# Note: This gives a memory error if I do all six labor categories at once.
for i in range(0,2):
    exio3.emp.append(exio3.emp_gender.diag_stressor(emp_ind[i], name=emp_ind[i]))   
    print(f"Created employment stressor for category {i}: {emp_ind[i]}")
    # exio3.emp[i].calc_system(exio3.x, Y=Y_food) # Without L and F_Y, this generates only the S matrix.
    exio3.emp[i].calc_system(exio3.x, Y=Y_food, Y_agg=Y_fd , L=exio3.L) 
    
    # Re: calc_system
    # Without L and F_Y, calc_system() generates only the S matrix.    
    # With L and F_Y, this generates all of M, D_cba, D_pba, D_imp, D_exp, D_cba_reg, D_pba_reg, D_imp_reg, D_exp_reg
    
    print(exio3.emp[i].name)
    print(str(exio3.emp[i]))
# => Without Y_agg explicitly given, calc_system aggregates the given Y=Y_food across fd categories 
# and diagonalizes it to a square Y with 'diagonalize_columns_to_sectors(Y_agg)'.


## Sankey diagram for employment impacts

# Example Sankey diagram for employment impacts (D_imp) by region and sector
hr_f_imp = exio3.emp[0].D_imp_reg
hr_f_all = exio3.emp[0].D_cba_reg
hr_m_imp = exio3.emp[1].D_imp_reg
hr_m_all = exio3.emp[1].D_cba_reg

# Regional sum of employment impacts (9800 is too much)
# mat_all = hr_f_all.groupby(hr_f_all.index.get_level_values('region')).sum()

import postprocess as pp
# mat = hr_m_imp.groupby('region').sum()
# pp.DrawSankey(mat, title="Employment Impact by Region and Sector (Male)", filename="output/region-region M.png")
# mat = hr_f_imp.groupby('region').sum()
# pp.DrawSankey(mat, title="Employment Impact by Region and Sector (Female)", filename="output/region-region F.png")

# Add a gender column to the employment impacts
hr_f_imp['gender'] = 'Female'
hr_m_imp['gender'] = 'Male'
# Combine the two DataFrames
hr_imp = pd.concat([hr_f_imp, hr_m_imp]).set_index(['gender'], append=True)
fig = pp.DrawSankey(hr_imp, 
                    title="Employment hours by Region and Gender (M.hrs)", 
                    filename="output/region-gender.png")

from dash import Dash, dcc, html
app = Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter




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
