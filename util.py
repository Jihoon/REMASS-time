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


def load_EXIOBASE3(year=2022, system="pxp"):
    """
    Loads the EXIOBASE3 dataset for the specified year and system.
    Returns an instance of the EXIOBASE3 class with the loaded data.
    """
    import pymrio
    from pathlib import Path

    exio3_folder = "H:\MyDocuments\Data\EXIOBASE3"

    fn = "IOT_" + str(year) + "_pxp.zip"
    p = Path(exio3_folder) / fn
    print(p.exists())

    exio3 = pymrio.parse_exiobase3(path=p)
    exio3.calc_system() # This generates all including L.

    # Look at the metadata of the EXIOBASE3 system
    print(exio3.meta)
    
    return exio3


def get_population(reg, year=2022):
    """
    Handles the population data for the specified regions and year.
    Returns a DataFrame with the population data.
    """
    import pycountry
    import world_bank_data as wb
    import pandas as pd
    import country_converter as coco

    pop_df = wb.get_series('SP.POP.TOTL', date='1995:2022', 
                           id_or_value='id', simplify_index=True)
    # Rename columns for clarity
    pop_df = pop_df.reset_index()
    # Convert the index to a DataFrame
    pop_df = pop_df.rename(columns={'Country':'iso3', 'SP.POP.TOTL':'Population'})
    # Change 'Year' type to number  
    pop_df['Year'] = pd.to_numeric(pop_df['Year'], errors='coerce')

    # Read 'EXIO-regions.csv' to get the country names and ISO codes
    # exio_regions = pd.read_csv('H:/MyDocuments/Data/EXIOBASE3/EXIO-regions.csv', 
    # index_col=0) \
    # .reset_index()
    exio_regions = pd.read_excel('H:/MyDocuments/Data/EXIOBASE3/EXIOBASE regions.xlsx', 
                                 sheet_name='Classification', index_col=False) \
        .reset_index(drop=True)
    exio_WX_regions = pd.read_excel('H:/MyDocuments/Data/EXIOBASE3/EXIOBASE regions.xlsx', 
                                 sheet_name='Rest of the World regions ISO3', index_col=False) \
        .reset_index(drop=True)
    
    # Add new rows to exio_regions based on exio_WX_regions
    wx_cols = [col for col in exio_WX_regions.columns if col.startswith('W')]
    wx_rows = []

    for col in wx_cols:
        for iso3 in exio_WX_regions[col].dropna():
            wx_rows.append({'region': col, 'region ISO3': iso3})
    exio_regions = pd.concat([exio_regions, pd.DataFrame(wx_rows)], ignore_index=True)

    # # Convert ISO3 to ISO2 codes using pycountry
    # def iso2_from_iso3(iso3):
    #     try:
    #         return pycountry.countries.get(alpha_3=iso3).alpha_2
    #     except:
    #         return None
        
    # def iso2_to_continent(country_alpha2):
    #     import pycountry_convert as pc

    #     # Some small countries may not have a continent code
    #     try:
    #         country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    #     except:
    #         return None
        
    #     country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    #     return country_continent_name

    # pop_df['iso2'] = pop_df['iso3'].apply(iso2_from_iso3)
    # # Filter out rows where ISO2 is None
    # pop_df = pop_df[pop_df['iso2'].notna()]
    # # Add country name column to pop_df
    # pop_df['country_name'] = pop_df['iso3'].apply(lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else None)
    # pop_df['continent_py'] = pop_df['iso2'].apply(iso2_to_continent)


    # Merge the population data with the EXIO regions based on iso2 codes
    popul = pop_df.merge(exio_regions, left_on='iso3', 
                         right_on='region ISO3', how='left')

    # def exio_region_from_continent(continent):
    #     import pandas as pd
    #     if pd.isna(continent):
    #         return None
    #     continent = continent.lower()
    #     if 'asia' in continent:
    #         return 'WA'
    #     elif 'africa' in continent:
    #         return 'WF'
    #     elif 'europe' in continent:
    #         return 'WE'
    #     elif 'america' in continent:
    #         return 'WL'
    #     elif 'oceania' in continent:
    #         return 'WA'
    #     elif 'middle east' in continent:
    #         return 'WM'
    #     else:
    #         return None

    # popul['Exiobase region code'] = popul['Exiobase region code'].fillna(
    #     popul['continent_py'].apply(exio_region_from_continent)
    # )

    # # if iso2 is one of these, then 'Exiobase region code'='WM' (Remove Malta)
    # # https://wits.worldbank.org/chatbot/SearchItem.aspx?RegionId=MEA 
    # popul.loc[popul['iso2'].isin(['AE', 'BH', 'DJ', 
    #                             #   'DZ', 'EG', 
    #                             'IR', 'IQ', 'IL', 'JO', 'KW', 'LB', 
    #                             #   'LY', 'MA', 
    #                             'OM', 'QA', 'SA', 'SY', 
    #                             #   'TN', 
    #                             'YE']),
    #         'Exiobase region code'] = 'WM'

    # Sum the population by Exiobase region code and year
    popul_sum = popul.groupby(['region', 'Year']) \
        .agg({'Population': 'sum'}) \
        .reset_index() 

    # Manually append 'Taiwan-populaton.csv' to popul_sum 
    # https://www.macrotrends.net/global-metrics/countries/twn/taiwan/population
    taiwan_popul = pd.read_csv('H:/MyDocuments/Data/EXIOBASE3/Taiwan-population.csv')
    popul_sum = pd.concat([popul_sum, taiwan_popul], ignore_index=True)

    # Put popul_sum regions in the same order as reg
    popul_sum['region'] = pd.Categorical(popul_sum['region'], categories=reg, ordered=True)
    # Sort the DataFrame by 'region' and 'Year'
    popul_sum = popul_sum.sort_values(by=['region', 'Year']).reset_index(drop=True)

    # Return the final DataFrame with population data for the given year
    popul_out = popul_sum[popul_sum['Year'] == year].drop(columns=['Year']).set_index('region', drop=True)
    return popul_out

