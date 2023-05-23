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

data = pd.read_csv(r'C:\Users\rikai\Desktop\Uni\דוקטורט\קורסים\ML\dataset_diabetes\diabetic_data.csv')


# Filtering and classifying 
filtered_data = data[['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id',  'admission_source_id',
'payer_code', 'number_diagnoses', 'change', 'diabetesMed', 'readmitted']]

filtered_data = filtered_data.replace('?', np.nan)

# age groups:
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


# payer code

payer_codes = filtered_data['payer_code'].unique()

# Define the payer code categories
payer_code_categories = {
    'nan': 'nan',
    'MC': 'mid_class',
    'MD': 'mid_class',
    'HM': 'mid_class',
    'UN': 'mid_class',
    'BC': 'mid_class',
    'SP': 'self_pay',
    'CP': 'mid_class',
    'SI': 'mid_class',
    'DM': 'expensive',
    'CM': 'expensive',
    'CH': 'expensive',
    'PO': 'mid_class',
    'WC': 'mid_class',
    'OT': 'mid_class',
    'OG': 'mid_class',
    'MP': 'expensive',
    'FR': 'mid_class'
}

for payer_code_categories, replacement in payer_code_categories.items():
    filtered_data.loc[filtered_data['payer_code'] == payer_code_categories, 'payer_code'] = replacement

filtered_data = filtered_data.dropna()

# 
