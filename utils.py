import pandas as pd
import numpy as np
import copy

gender_categories = {
    'Female': 0,
    'Male': 1
}
race_categories = {
    'Caucasian': 0,
    'Asian': 1,
    'AfricanAmerican': 2,
    'Hispanic': 3,
    'Other': 4,
}
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

def createLabels(path_to_data, col_filter=None):
    if col_filter is None:
        col_filter = ['encounter_id', 'patient_nbr', 'age', 'admission_type_id',
                      'admission_source_id', 'diabetesMed',
                      'payer_code', 'number_diagnoses', 'readmitted', 'change',
                      'num_medications', 'discharge_disposition_id']
        # col_filter = ['encounter_id', 'patient_nbr', 'age',
        #               'payer_code','readmitted']
    data = pd.read_csv(path_to_data)
    ## Filtering and classifying
    data = prepareData(data, col_filter)
    data['age_group'] = data['age_group'].astype(int)
    data['admission_type_id'] = data['admission_type_id'].astype(int)
    data['admission_source_id'] = data['admission_source_id'].astype(int)
    data['diabetesMed'] = data['diabetesMed'].astype(int)
    data['payer_code'] = data['payer_code'].astype(int)
    data['number_diagnoses'] = data['number_diagnoses'].astype(int)
    data['change'] = data['change'].astype(int)
    data['num_medications'] = data['num_medications'].astype(int)
    data['discharge_disposition_id'] = data['discharge_disposition_id'].astype(int)
    return data
def prepareData(data, col_filter):
    filtered_data = data[col_filter]
    filtered_data = filtered_data.replace('?', np.nan)
    filtered_data = filtered_data.dropna()
    # Age groups:
    for age_group, replacement in age_groups.items():
        filtered_data.loc[filtered_data['age'] == age_group, 'age'] = replacement

    filtered_data = filtered_data.rename(columns={'age': 'age_group'})
    filtered_data['diabetesMed'] = filtered_data['diabetesMed'].map({'Yes': True, 'No': False})
    filtered_data['change'] = filtered_data['change'].map({'Ch': True, 'No': False})
    filtered_data['payer_code'] = filtered_data['payer_code'].map(payer_code_categories)
    filtered_data.head()
    Y = {}
    for patient_nbr,group in filtered_data.groupby('patient_nbr'):
        if len(group) ==1:
            continue
        Y[patient_nbr] ={}
        indx2 = group.index
        for i, row2 in enumerate(indx2):
            Y[patient_nbr][row2] = copy.deepcopy(group.loc[row2])
            Y[patient_nbr][row2]['readmitted_less_than_30'] = 0
            if i + 1 < len(indx2):
                Y[patient_nbr][row2]['readmitted_less_than_30'] = 1 if filtered_data.loc[
                                                              list(indx2)[i + 1], 'readmitted'] == '<30' else 0
    # for indx1, row in filtered_data.iterrows():
    #     patient_nbr = row['patient_nbr']
    #     indx2 = filtered_data[filtered_data['patient_nbr'] == patient_nbr].index
    #     # if len(indx2) == 1:
    #     #     continue
    #     if indx1 not in Y:
    #         for i, row2 in enumerate(indx2):
    #             Y[row2] = copy.deepcopy(filtered_data.loc[row2])
    #             Y[row2]['readmitted_less_than_30'] = 0
    #             if i + 1 < len(indx2):
    #                 Y[row2]['readmitted_less_than_30'] = 1 if filtered_data.loc[
    #                                                               list(indx2)[i + 1], 'readmitted'] == '<30' else 0
    #                 continue

    collapsed_data_rows = []
    for patient_nbr, values in Y.items():
        for row_num,row in values.items():
            collapsed_data_rows.append(row)
    # Create the collapsed_data DataFrame using concat
    collapsed_data = pd.concat(collapsed_data_rows, axis=1).T

    collapsed_data.head()
    collapsed_data.to_csv('dataCleaner.csv')
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
