import sys


def main(argv):
    print("start here")

if __name__ == "__main__":
    main(sys.argv)
    
    
    import sys


def main(argv):
    print("start here")

if __name__ == "__main__":
    main(sys.argv)
    
    
    
#### 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'.\data\diabetic_data.csv')


## Filtering and classifying 
filtered_data = data[['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id',  'admission_source_id',
'payer_code', 'number_diagnoses', 'change', 'diabetesMed', 'readmitted']]

filtered_data = filtered_data.replace('?', np.nan)
filtered_data = filtered_data.dropna()

# Age groups:
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

for payer_code_categories, replacement in payer_code_categories.items():
    filtered_data.loc[filtered_data['payer_code'] == payer_code_categories, 'payer_code'] = replacement

# 
## For each sample in every patient_nbr, label according to the sample that followed it (did s/he return in less than 30 days on i+1)

# Create the collapsed_data DataFrame
collapsed_data = pd.DataFrame(columns=filtered_data.columns)
collapsed_data_rows = []

# Iterate over each row in the filtered_data DataFrame
for indx1, row in filtered_data.iterrows():
    patient_nbr = row['patient_nbr']
    readmitted = row['readmitted']
    
    # Check if patient_nbr appears more than once
    if filtered_data['patient_nbr'].value_counts()[patient_nbr] > 1:
        # Find the index of the second occurrence of patient_nbr
        indx2 = filtered_data[filtered_data['patient_nbr'] == patient_nbr].index[1]
        
        # Get the readmitted value for the second occurrence
        re_admitted = filtered_data.loc[indx2, 'readmitted']
        collapsed_data_rows.append(row)
        
        # readmitted_less_than_30? (based on the second visit)
        if re_admitted == '>30'
            collapsed_data_rows[-1]['readmitted_less_than_30'] = True
        else:
            collapsed_data_rows[-1]['readmitted_less_than_30'] = False

# Create the collapsed_data DataFrame using concat
collapsed_data = pd.concat(collapsed_data_rows, axis=1).T

collapsed_data.head()

# descriptive statstemp_df = filtered_data.drop_duplicates(subset='patient_nbr')
# Calculate mean and standard deviation for age_groupage_group_mean = temp_df['age_group'].mean()age_group_std = temp_df['age_group'].std()
# Calculate the percentage of males and femalesgender_counts = temp_df['gender'].value_counts()gender_percentage = gender_counts / len(temp_df) * 100
# Calculate the percentages of payer_code_categoriespayer_code_counts = temp_df['payer_code'].value_counts()payer_code_percentage = payer_code_counts / len(temp_df) * 100
descriptive_table = collapsed_data({    'Variable': ['Age Group', 'Gender', 'Payer Code'],    'Mean': [age_group_mean, '', ''],    'Standard Deviation': [age_group_std, '', ''],    'Percentage Male': ['', gender_percentage['Male'], ''],    'Percentage Female': ['', gender_percentage['Female'], ''],    'Percentage Mid Class': ['', '', payer_code_percentage['mid_class']],    'Percentage Expensive': ['', '', payer_code_percentage['expensive']],    'Percentage Self Pay': ['', '', payer_code_percentage['self_pay']],})
print(descriptive_table)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data Summary
print(filtered_data.describe())


## percent of less than 30
total_readmitted_less_than_30 = (filtered_data['readmitted'] == '<30').sum()
total_readmitted_greater_than_30 = (filtered_data['readmitted'] == '>30').sum()

# Calculate the total number of patients in the dataset
total_patients = len(filtered_data)

# Calculate the percentages
percentage_readmitted_less_than_30 = (total_readmitted_less_than_30 / total_patients) * 100
percentage_readmitted_greater_than_30 = (total_readmitted_greater_than_30 / total_patients) * 100

# Print the percentages
print(f"Percentage of patients readmitted within 30 days ('<30'): {percentage_readmitted_less_than_30:.2f}%")
print(f"Percentage of patients readmitted after 30 days ('>30'): {percentage_readmitted_greater_than_30:.2f}%")

## Change in meds && readmission rates
num_patients_on_medication = filtered_data['diabetesMed'].sum()
print(f"The number of patients with yes medication: {num_patients_on_medication}")
total_change_meds = filtered_data['change'].value_counts()
total_readmitted = filtered_data['readmitted'].value_counts()
total_readmitted_change_meds = filtered_data.groupby(['change', 'readmitted']).size().reset_index(name='count')

count_percentage_df = total_readmitted_change_meds.pivot_table(index='change', columns='readmitted', values='count', fill_value=0)

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
correlation_matrix = filtered_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

## Diabetes Medication and Readmission
plt.figure(figsize=(8, 6))
sns.countplot(data=filtered_data, x='diabetesMed', hue='readmitted')
plt.xlabel('Diabetes Medication')
plt.ylabel('Count')
plt.title('Diabetes Medication and Readmission')
plt.legend(title='Readmitted', loc='upper right')
plt.show()

## Admission Type and Source
plt.figure(figsize=(12, 6))
sns.countplot(data=filtered_data, x='admission_type_id', hue='admission_source_id')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.title('Admission Type and Source')
plt.legend(title='Admission Source', loc='upper right')
plt.show()

## Race and Gender
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
filtered_data['race'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Distribution of Race')

plt.subplot(1, 2, 2)
filtered_data['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Distribution of Gender')

plt.tight_layout()
plt.show()

## Num of diagnoses
plt.figure(figsize=(8, 6))
sns.histplot(filtered_data['number_diagnoses'], bins=20, kde=False)
plt.xlabel('Number of Diagnoses')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Diagnoses')
plt.show()

## Age group && num of diags
plt.figure(figsize=(8, 6))
sns.boxplot(data=filtered_data, x='age_group', y='number_diagnoses')
plt.xlabel('Age Group')
plt.ylabel('Number of Diagnoses')
plt.title('Age Group and Number of Diagnises')
Xlbls = ['0-10', '10-20', '20-40', '40-60', '60-100']
plt.xticks(ticks=range(len(Xlbls)), labels=Xlbls)
plt.show()


# Change in Medication and Readmission
plt.figure(figsize=(8, 6))
sns.countplot(data=filtered_data, x='change', hue='readmitted')
plt.xlabel('Change in Medication')
plt.ylabel('Count')
plt.title('Change in Medication and Readmission')
plt.legend(title='Readmitted', loc='upper right')
plt.show()

# Readmission Status
plt.figure(figsize=(8, 6))
sns.countplot(data=filtered_data, x='readmitted')
plt.xlabel('Readmission Status')
plt.ylabel('Count')
plt.title('Readmission Status')
plt.show()
