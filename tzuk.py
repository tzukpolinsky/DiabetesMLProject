import sys
import pandas
from utils import groupByAttrIndexToDic, createPatternsByIndex, emergencyCodeToPatternIndex
import matplotlib.pyplot as plt


def main(argv):
    pandas_data = pandas.read_csv(argv[1])
    data = pandas_data.values
    group_by = groupByAttrIndexToDic(data, pandas_data.columns.get_loc("patient_nbr"))
    # groupBy = np.split(data[:, :2], np.unique(data[:, 1], return_index=True)[1][1:])
    patterns = createPatternsByIndex(group_by, pandas_data.columns.get_loc("admission_type_id"),
                                     pandas_data.columns.get_loc("patient_nbr"), emergencyCodeToPatternIndex)
    for index, (key, value) in enumerate(patterns.items()):
        plt.plot(list(range(len(key))), list(key),"ro")
        plt.savefig(argv[2] + "/pattern " + str(index) + " for " + str(len(key)))
        plt.clf()
    print()


if __name__ == "__main__":
    main(sys.argv)
