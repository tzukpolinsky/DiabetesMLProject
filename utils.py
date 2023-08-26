import math

import pandas as pd
import numpy as np
import copy

import scipy.spatial.distance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

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
    'MC': 2,
    'MD': 1,
    'MP': 4,
    'BC': 4,
    'UN': 5,
    'SI': 2,
    'CP': 3,
    'HM': 5,
    'SP': 5,
    'WC': 2,
    'OT': 5,
    'OG': 3,
    'CH': 3,
    'PO': 3,
    'DM': 2,
    'CM': 4,
    'FR': 3,
}

age_groups = {
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 1,
    '[30-40)': 2,
    '[40-50)': 2,
    '[50-60)': 3,
    '[60-70)': 4,
    '[70-80)': 4,
    '[80-90)': 5,
    '[90-100)': 5
}
discharge_disposition_map = {
    1: 1,
    2: 4,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 1,
    8: 3,
    9: 4,
    10: 4,
    11: 5,
    12: 4,
    13: 2,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: np.nan,
    19: 5,
    20: 5,
    21: 5,
    22: 4,
    23: 4,
    24: 3,
    25: np.nan,
    26: np.nan,
    30: 3,
    27: 3,
    28: 4,
    29: 4
}
admission_source_map = {
    1: 4,
    2: 3,
    3: 3,
    4: 5,
    5: 5,
    6: 4,
    7: 5,
    8: 2,
    9: np.nan,
    10: 5,
    11: 8,
    12: 8,
    13: 8,
    14: 8,
    15: np.nan,
    17: np.nan,
    18: 3,
    19: 4,
    20: np.nan,
    21: np.nan,
    22: 4,
    23: 8,
    24: 8,
    25: 4,
    26: 4

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
        col_filter = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id', 'time_in_hospital',
                      'payer_code', 'change',
                      'num_medications', 'discharge_disposition_id', 'admission_source_id', 'readmitted']

        # col_filter = ['encounter_id', 'patient_nbr', 'age',
        #               'payer_code','readmitted']
    data = pd.read_csv(path_to_data)
    ## Filtering and classifying
    data = prepareData(data, col_filter)
    # data['age_group'] = data['age_group'].astype(int)
    # data['admission_type_id'] = data['admission_type_id'].astype(int)
    # data['admission_source_id'] = data['admission_source_id'].astype(int)
    # data['diabetesMed'] = data['diabetesMed'].astype(int)
    # data['payer_code'] = data['payer_code'].astype(int)
    # data['number_diagnoses'] = data['number_diagnoses'].astype(int)
    # data['change'] = data['change'].astype(int)
    # data['num_medications'] = data['num_medications'].astype(int)
    # data['discharge_disposition_id'] = data['discharge_disposition_id'].astype(int)
    # data['time_in_hospital'] = data['time_in_hospital'].astype(int)
    # data['race'] = data['race'].astype(int)
    # data['gender'] = data['gender'].astype(int)
    # data['num_procedures'] = data['num_procedures'].astype(int)
    return data


def prepareData(data, col_filter):
    filtered_data = data[col_filter]
    filtered_data = filtered_data.replace('?', np.nan)
    # Age groups:
    for age_group, replacement in age_groups.items():
        filtered_data.loc[filtered_data['age'] == age_group, 'age'] = replacement

    filtered_data = filtered_data.rename(columns={'age': 'age_group'})
    # filtered_data['diabetesMed'] = filtered_data['diabetesMed'].map({'Yes': 1, 'No': 0})
    filtered_data['change'] = filtered_data['change'].map({'Ch': 1, 'No': 0})
    filtered_data['readmitted'] = filtered_data['readmitted'].map({'NO': 0, '<30': 1, '>30': 0})
    filtered_data['payer_code'] = filtered_data['payer_code'].map(payer_code_categories)
    filtered_data['race'] = filtered_data['race'].map(race_categories)
    filtered_data['gender'] = filtered_data['gender'].map(gender_categories)
    filtered_data['admission_type_id'] = filtered_data['admission_type_id'].map(emergencyCodeToPatternIndex)
    filtered_data['discharge_disposition_id'] = filtered_data['discharge_disposition_id'].map(discharge_disposition_map)
    filtered_data['admission_source_id'] = filtered_data['admission_source_id'].map(admission_source_map)
    filtered_data.head()
    # IterativeImputer()  # KNNImputer(n_neighbors=math.ceil(filtered_data.shape[0]*0.005))#
    imp = SimpleImputer(strategy="most_frequent")
    imp.fit(filtered_data)
    filtered_data_imputed = imp.transform(filtered_data)
    filtered_data = pd.DataFrame(filtered_data_imputed, columns=filtered_data.columns)
    # cleanClassesDuplications(filtered_data)
    # return filtered_data
    Y = {}
    for patient_nbr, group in filtered_data.groupby('patient_nbr'):
        if group.values[0][-2] == 8:
            continue
        Y[patient_nbr] = copy.deepcopy(group.values[0][2:])
        count_true = 0
        for member in group.values:
            if member[-1]:
                count_true += 1
        if count_true > 0:
            np.append(Y[patient_nbr], int(group.values.shape[0] / count_true >= 0.5))
        else:
            np.append(Y[patient_nbr], 0)
    returned_data = np.array(list(Y.values()), dtype=float)
    neg = returned_data[returned_data[:, -1] == 0]
    pos = returned_data[returned_data[:, -1] == 1]
    neg_bad_indices = np.array([True] * neg.shape[0])
    pos_bad_indices = np.array([True] * pos.shape[0])
    for i,p in enumerate(pos):
        current_neg_bad_indices = np.linalg.norm(neg[:, :-1] - p[:-1], axis=1) >= 1
        if (current_neg_bad_indices.shape[0] - current_neg_bad_indices.sum()) > neg.shape[0]*0.005:
            pos_bad_indices[i] = False
        neg_bad_indices = np.logical_and(neg_bad_indices, current_neg_bad_indices)
    returned_data = np.concatenate([neg[neg_bad_indices], pos[pos_bad_indices]])
    return returned_data


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
