import sys
import pandas as pd
import numpy as np
import utils
from utils import groupByAttrIndexToDic, createPatternsByIndex, emergencyCodeToPatternIndex
import matplotlib.pyplot as plt

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
def tag_readmitted(filtered_data, index, row):
    re_admitted = filtered_data.loc[index, 'readmitted']
    # readmitted_less_than_30? (based on the second visit)
    if re_admitted == '>30':
        row['readmitted_less_than_30'] = True
    else:
        row['readmitted_less_than_30'] = False
    return row


def createLabels(path_to_data, col_filter=None):
    if col_filter is None:
        col_filter = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id',
                      'admission_source_id',
                      'payer_code', 'number_diagnoses', 'change', 'diabetesMed', 'readmitted']
    data = pd.read_csv(path_to_data)
    ## Filtering and classifying
    data = utils.prepareData(data, col_filter)
    print(len(data))

def createTrainAndTest():
    pass
def runRandomForest():
    pass
def main(argv):
    createLabels(argv[1])

    return
    pandas_data = pd.read_csv(argv[1])
    data = pandas_data.values
    group_by = groupByAttrIndexToDic(data, pandas_data.columns.get_loc("patient_nbr"))
    # groupBy = np.split(data[:, :2], np.unique(data[:, 1], return_index=True)[1][1:])
    patterns = createPatternsByIndex(group_by, pandas_data.columns.get_loc("admission_type_id"),
                                     pandas_data.columns.get_loc("patient_nbr"), emergencyCodeToPatternIndex)
    for index, (key, value) in enumerate(patterns.items()):
        plt.plot(list(range(len(key))), list(key), "ro")
        plt.savefig(argv[2] + "/pattern " + str(index) + " for " + str(len(value)))
        plt.clf()
    print()


if __name__ == "__main__":
    main(sys.argv)
