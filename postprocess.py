def DrawSankey(mat, title, filename):
    """
    Draws a Sankey diagram based on the provided matrix of employment impacts.
    
    Parameters:
    - mat: DataFrame containing employment impacts. (9800n x 49) 
        = Has a multi-index with 'region', 'sector', 'gender' 
    - title: Title of the Sankey diagram.
    - filename: Filename to save the Sankey diagram.
    """
    import plotly.graph_objects as go

    src_regs = mat.index.get_level_values('region').unique().tolist()
    genders = mat.index.get_level_values('gender').unique().tolist()
    tgt_regs = mat.columns.tolist()
    values = mat.values.flatten()

    # Create source and target indices for Sankey
    sources = []
    targets = []
    sankey_values = []

    for i, region in enumerate(src_regs):
        for j, sector in enumerate(tgt_regs):
            val = mat.iloc[i, j]
            if val > 0:
                sources.append(i)
                targets.append(len(src_regs) + j)
                sankey_values.append(val)

    labels = src_regs + tgt_regs

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=sankey_values,
        ))])

    fig.update_layout(title_text=title, font_size=10)
    fig.show()
    fig.write_image(filename)




#### Debug scripts

# print(exio3.emp[0].D_cba) # Impact of individual sectors and country/region
# print(exio3.emp[0].D_cba_reg) # Sum by country/region


# # Impact plots
# for i in range(0,2):
#     exio3.emp[i].plot_account(row=('US', 'Vegetables, fruit, nuts'))
#     exio3.emp[i].plot_account(row=('US', sec[70]))
#     exio3.emp[i].plot_account(row=('CN', sec[70]))

#     # print(f"Plotting D_cba for {exio3.emp[i].D_cba.loc[row]}")
#     # print(f"Plotting D_pba for {exio3.emp[i].D_pba.loc[row]}")
#     # print(f"Plotting D_imp for {exio3.emp[i].D_imp.loc[row]}")
#     # print(f"Plotting D_exp for {exio3.emp[i].D_exp.loc[row]}")

# # # Test-view 
# # exio3.employment.calc_system(exio3.x, Y_food, L=exio3.L)
# # mat_S = exio3.employment.S.iloc[0,:]
# # mat_M = exio3.employment.M.iloc[0,:]
# # Y_food_HH = Y_food.loc[:, Y_food.columns.get_level_values('category').isin(['Final consumption expenditure by households'])]



# # Get the distribution of values in the cells of exio3.emp[0].D_cba
# import numpy as np
# mat = exio3.emp[0].D_imp_reg
# values = mat.values.flatten()
# # Calculate the distribution of values
# values_distribution = np.histogram(values, bins=100)
# # Print the distribution
# print("Distribution of values:")
# print(values_distribution)

# # Find the index of the maximum value in exio3.emp[0].D_cba
# max_index = np.unravel_index(np.argmax(mat.values, axis=None), mat.shape)
# # Print the maximum value and its index
# max_value = mat.values[max_index]
# print(f"Maximum value: {max_value} at index {max_index}")
# mat.index[max_index[0]], mat.columns[max_index[1]]



# # Visualize
# import matplotlib.pyplot as plt

# m = exio3.emp[0].D_imp #.loc[:, Y_fd.columns.get_level_values('region').isin(["WF"])]
# plt.figure(figsize=(15, 15))
# plt.imshow(m)
# plt.colorbar()
# plt.xlabel("Countries - tgt_regs")
# plt.ylabel("Countries - M.hours")
# plt.show()



## Attempt to avoid the memory error

# # Use calc_accounts to avoid the memory error
# for i in range(0,6):
#     a,b,c,d = pymrio.calc_accounts(S=exio3.emp[i].S, L=exio3.L, Y=Y_fd) # This does not work, as it requires the calc_system to be run first.
#     exio3.emp[i].D_cba = a
#     exio3.emp[i].D_pba = b
#     exio3.emp[i].D_imp = c
#     exio3.emp[i].D_exp = d

#     print(exio3.emp[i].name)
#     print(str(exio3.emp[i]))

# => All four Dx are needed to use e.g., plot_account().