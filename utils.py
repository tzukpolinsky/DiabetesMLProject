import math

import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
import seaborn as sns

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
    '[90-100)': 5,
    '[75-100)': 5
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
"""
for project report
"""


def prepare_and_plot_project_statistics(data):
    data = data[['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id', 'diabetesMed',
                 'payer_code', 'change', 'insulin', 'number_diagnoses',
                 'num_medications', 'discharge_disposition_id', 'admission_source_id', 'readmitted']]
    data = data.replace('?', np.nan)

    data = data.rename(columns={'age': 'age_group'})
    data['age_group'] = data['age_group'].map(age_groups)
    data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})
    data['change'] = data['change'].map({'Ch': 1, 'No': 0})
    #data['readmitted'] = data['readmitted'].map({'NO': 0, '<30': 1, '>30': 0}) # This reduces the data by comibing NO with >30 --> we need them separate at this point.
    data['insulin'] = data['insulin'].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})
    data['payer_code'] = data['payer_code'].map(payer_code_categories)
    data['race'] = data['race'].map(race_categories)
    data['gender'] = data['gender'].map(gender_categories)
    data['admission_type_id'] = data['admission_type_id'].map(emergencyCodeToPatternIndex)
    data['discharge_disposition_id'] = data['discharge_disposition_id'].map(discharge_disposition_map)
    data['admission_source_id'] = data['admission_source_id'].map(admission_source_map)
    imp = SimpleImputer(strategy="mean")
    imp.fit(data)
    data_imputed = imp.transform(data)
    data = pd.DataFrame(data_imputed, columns=data.columns)
    print(data.describe())

    ## percent of less/more than 30. NOTE: not readmitted at all are not included
    total_readmitted_less_than_30 = (data['readmitted'] == '<30').sum()
    total_readmitted_greater_than_30 = (data['readmitted'] == '>30').sum()

    # Calculate the total number of patients in the dataset
    total_patients = len(data)

    # Calculate the percentages
    percentage_readmitted_less_than_30 = (total_readmitted_less_than_30 / total_patients) * 100
    percentage_readmitted_greater_than_30 = (total_readmitted_greater_than_30 / total_patients) * 100

    # Print the percentages
    print(f"Percentage of patients readmitted within 30 days ('<30'): {percentage_readmitted_less_than_30:.2f}%")
    print(f"Percentage of patients readmitted after 30 days ('>30'): {percentage_readmitted_greater_than_30:.2f}%")

    ## Change in meds && readmission rates
    num_patients_on_medication = data['diabetesMed'].sum()
    total_change_meds = data['change'].value_counts()
    total_readmitted = filtered_data['readmitted'].value_counts()
    total_readmitted_change_meds = data.groupby(['change', 'readmitted']).size().reset_index(name='count')
    print(f"The number of patients with yes medication: {num_patients_on_medication}")
    count_percentage_df = total_readmitted_change_meds.pivot_table(index='change', columns='readmitted', values='count',
                                                                   fill_value=0)

    # Calc %%
    count_percentage_df['Total'] = count_percentage_df.sum(axis=1)
    count_percentage_df['Percentage <30'] = (count_percentage_df['<30'] / count_percentage_df['Total'] * 100).round(2)
    count_percentage_df['Percentage >30'] = (count_percentage_df['>30'] / count_percentage_df['Total'] * 100).round(2)

    # Print
    for change_meds in count_percentage_df.index:
        count_less_than_30 = count_percentage_df.loc[change_meds, '<30']
        count_greater_than_30 = count_percentage_df.loc[change_meds, '>30']
        percentage_less_than_30 = count_percentage_df.loc[change_meds, 'Percentage <30']
        percentage_greater_than_30 = count_percentage_df.loc[change_meds, 'Percentage >30']
        print(f"Change in Medication: {change_meds}")
        print(f"Readmitted '<30': {count_less_than_30} ({percentage_less_than_30}%)")
        print(f"Readmitted '>30': {count_greater_than_30} ({percentage_greater_than_30}%)")
        print()

    #### FIGURES
    # Correlation Analysis
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    ## Diabetes Medication and Readmission
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='diabetesMed', hue='readmitted')
    plt.xlabel('Diabetes Medication')
    plt.ylabel('Count')
    plt.title('Diabetes Medication and Readmission')
    plt.legend(title='Readmitted', loc='upper right')
    plt.show()

    ## Admission Type and Source
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x='admission_type_id', hue='admission_source_id')
    plt.xlabel('Admission Type')
    plt.ylabel('Count')
    plt.title('Admission Type and Source')
    plt.legend(title='Admission Source', loc='upper right')
    plt.show()

    ## Race and Gender
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    data['race'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Distribution of Race')

    plt.subplot(1, 2, 2)
    data['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Distribution of Gender')

    plt.tight_layout()
    plt.show()

    ## Num of diagnoses
    plt.figure(figsize=(8, 6))
    sns.histplot(data['number_diagnoses'], bins=20, kde=False)
    plt.xlabel('Number of Diagnoses')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Diagnoses')
    plt.show()

    ## Age group && num of diags
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='age_group', y='number_diagnoses')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Diagnoses')
    plt.title('Age Group and Number of Diagnises')
    Xlbls = ['0-10', '10-20', '20-40', '40-60', '60-100']
    plt.xticks(ticks=range(len(Xlbls)), labels=Xlbls)
    plt.show()

    # Change in Medication and Readmission
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='change', hue='readmitted')
    plt.xlabel('Change in Medication')
    plt.ylabel('Count')
    plt.title('Change in Medication and Readmission')
    plt.legend(title='Readmitted', loc='upper right')
    plt.show()

    # Readmission Status
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='readmitted')
    plt.xlabel('Readmission Status')
    plt.ylabel('Count')
    plt.title('Readmission Status')
    plt.show()

    # descriptives table
    temp_df = data.drop_duplicates(subset='patient_nbr')
    age_group_mean = temp_df['age_group'].mean()
    age_group_std = temp_df['age_group'].std()
    gender_counts = temp_df['gender'].value_counts()
    gender_percentage = gender_counts / len(temp_df) * 100
    payer_code_counts = temp_df['payer_code'].value_counts()
    payer_code_percentage = payer_code_counts / len(temp_df) * 100
    descriptive_table = data({'Variable': ['Age Group', 'Gender', 'Payer Code'], 'Mean': [age_group_mean, '', ''],
                              'Standard Deviation': [age_group_std, '', ''],
                              'Percentage Male': ['', gender_percentage['Male'], ''],
                              'Percentage Female': ['', gender_percentage['Female'], ''],
                              'Percentage Mid Class': ['', '', payer_code_percentage['mid_class']],
                              'Percentage Expensive': ['', '', payer_code_percentage['expensive']],
                              'Percentage Self Pay': ['', '', payer_code_percentage['self_pay']], })
    print(descriptive_table)


def create_labels(path_to_data, col_filter=None):
    if col_filter is None:
        col_filter = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id',
                      'payer_code', 'change', 'insulin',
                      'num_medications', 'discharge_disposition_id', 'admission_source_id', 'readmitted']

        # col_filter = ['encounter_id', 'patient_nbr', 'age',
        #               'payer_code','readmitted']
    data = pd.read_csv(path_to_data)
    ## Filtering and classifying
    data = prepare_data(data, col_filter)
    return data


def clean_overlapping_data(data):
    neg = data[data[:, -1] == 0]
    pos = data[data[:, -1] == 1]
    neg_bad_indices = np.array([True] * neg.shape[0])
    pos_bad_indices = np.array([True] * pos.shape[0])
    for i, p in enumerate(pos):
        current_neg_bad_indices = np.linalg.norm(neg[:, :-1] - p[:-1], axis=1) >= 2
        if (current_neg_bad_indices.shape[0] - current_neg_bad_indices.sum()) > neg.shape[0] * 0.005:
            pos_bad_indices[i] = False
            continue
        neg_bad_indices = np.logical_and(neg_bad_indices, current_neg_bad_indices)
    print("filtered indices " + str(pos_bad_indices.shape[0] - pos_bad_indices.sum()))
    return np.concatenate([neg[neg_bad_indices], pos[pos_bad_indices]])


def prepare_data(data, col_filter):
    filtered_data = data[col_filter]
    filtered_data = filtered_data.replace('?', np.nan)
    # Age groups:
    for age_group, replacement in age_groups.items():
        filtered_data.loc[filtered_data['age'] == age_group, 'age'] = replacement

    filtered_data = filtered_data.rename(columns={'age': 'age_group'})
    filtered_data['diabetesMed'] = filtered_data['diabetesMed'].map({'Yes': 1, 'No': 0})
    filtered_data['change'] = filtered_data['change'].map({'Ch': 1, 'No': 0})
    filtered_data['readmitted'] = filtered_data['readmitted'].map({'NO': 0, '<30': 1, '>30': 0})
    filtered_data['insulin'] = filtered_data['insulin'].map({'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3})
    filtered_data['payer_code'] = filtered_data['payer_code'].map(payer_code_categories)
    filtered_data['race'] = filtered_data['race'].map(race_categories)
    filtered_data['gender'] = filtered_data['gender'].map(gender_categories)
    filtered_data['admission_type_id'] = filtered_data['admission_type_id'].map(emergencyCodeToPatternIndex)
    filtered_data['discharge_disposition_id'] = filtered_data['discharge_disposition_id'].map(discharge_disposition_map)
    filtered_data['admission_source_id'] = filtered_data['admission_source_id'].map(admission_source_map)
    filtered_data.head()
    # IterativeImputer()  # KNNImputer(n_neighbors=math.ceil(filtered_data.shape[0]*0.005))#
    imp = SimpleImputer(strategy="mean")
    imp.fit(filtered_data)
    filtered_data_imputed = imp.transform(filtered_data)
    filtered_data = pd.DataFrame(filtered_data_imputed, columns=filtered_data.columns)
    # cleanClassesDuplications(filtered_data)
    # return filtered_data
    Y = {}
    for patient_nbr, group in filtered_data.groupby('patient_nbr'):
        if group.values[0][-2] == 8:
            continue
        Y[patient_nbr] = copy.deepcopy(group.values[0][2:-1])
        count_true = 0
        pos_indices = []
        neg_indices = []
        for i, member in enumerate(group.values):
            if member[-1]:
                count_true += 1
                pos_indices.append(i)
            else:
                neg_indices.append(i)
        if count_true > 0:
            if group.values.shape[0] / count_true >= 0.5:
                Y[patient_nbr] = copy.deepcopy(group.values[pos_indices[0]][2:-1])
                Y[patient_nbr] = copy.deepcopy(np.append(Y[patient_nbr], 1))
            else:
                Y[patient_nbr] = copy.deepcopy(group.values[neg_indices[0]][2:-1])
                Y[patient_nbr] = copy.deepcopy(np.append(Y[patient_nbr], 0))

        else:
            Y[patient_nbr] = copy.deepcopy(np.append(Y[patient_nbr], 0))
    returned_data = np.array(list(Y.values()), dtype=float)
    return returned_data
