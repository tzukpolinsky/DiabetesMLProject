import pandas as pd
import numpy as np

payer_code_categories = {
    # 'nan': 'nan',
    'MC': '2',
    'MD': '2',
    'HM': '2',
    'UN': '2',
    'BC': '2',
    'SP': '1',
    'CP': '2',
    'SI': '2',
    'DM': '3',
    'CM': '3',
    'CH': '3',
    'PO': '2',
    'WC': '2',
    'OT': '2',
    'OG': '2',
    'MP': '3',
    'FR': '2'
}

age_groups = {
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 2,
    '[30-40)': 2,
    '[40-50)': 3,
    '[50-60)': 3,
    '[60-70)': 4,
    '[70-80)': 4,
    '[80-90)': 4,
    '[90-100)': 4
}
emergencyCodeToPatternIndex = {
    1: 4,
    2: 4,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    8: 1,
    7: 4
}


def groupByAttrIndexToDic(data, index_of_attr):
    results = {}
    for row in data:
        index_value = row[index_of_attr]
        if index_value not in results:
            results[index_value] = []
        results[index_value].append(row)
    return results


def prepareData(data, col_filter):
    filtered_data = data[col_filter]
    filtered_data = filtered_data.replace('?', np.nan)
    filtered_data = filtered_data.dropna()
    # Age groups:
    for age_group, replacement in age_groups.items():
        filtered_data.loc[filtered_data['age'] == age_group, 'age'] = replacement

    filtered_data = filtered_data.rename(columns={'age': 'age_group'})
    # "diabetesMed": Convert "Yes"/"No" to True/False
    filtered_data['diabetesMed'] = filtered_data['diabetesMed'].map({'Yes': True, 'No': False})
    # "change": Convert "Ch"/"No" to True/False
    filtered_data['change'] = filtered_data['change'].map({'Ch': True, 'No': False})
    filtered_data.head()
    # Payer code
    payer_codes = filtered_data['payer_code'].unique()
    # Define the payer code categories: 1 = self pay, 2 = mid class insurance, 3 = expensive/premium

    for payer_code_category, replacement in payer_code_categories.items():
        filtered_data.loc[filtered_data['payer_code'] == payer_code_category, 'payer_code'] = replacement

    #
    ## For each sample in every patient_nbr, label according to the sample that followed it (did s/he return in less than 30 days on i+1)

    # Create the collapsed_data DataFrame
    # collapsed_data = pd.DataFrame(columns=filtered_data.columns)
    # Iterate over each row in the filtered_data DataFrame
    Y = {}
    for indx1, row in filtered_data.iterrows():
        patient_nbr = row['patient_nbr']
        indx2 = filtered_data[filtered_data['patient_nbr'] == patient_nbr].index
        for i, row2 in enumerate(indx2):
            if row2 not in Y:
                Y[row2] = row
                if i + 1 < len(indx2):
                    Y[row2]['readmitted_less_than_30'] = filtered_data.loc[list(indx2)[i + 1], 'readmitted'] == '<30'
                    continue
                Y[row2]['readmitted_less_than_30'] = False
    collapsed_data_rows = []
    for key, value in Y.items():
        collapsed_data_rows.append(value)
    # Create the collapsed_data DataFrame using concat
    collapsed_data = pd.concat(collapsed_data_rows, axis=1).T

    collapsed_data.head()
    return collapsed_data

def createPatternsByIndex(grouped_data, attr_index, pattern_index, conversion_dict=None):
    patterns = {}
    for group_id, group in grouped_data.items():
        pattern = []
        for row in group:
            value = row[attr_index]
            if conversion_dict:
                value = conversion_dict[value]
            pattern.append(value)
        pattern = tuple(pattern)
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(group_id)
    return patterns
