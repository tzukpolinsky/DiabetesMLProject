def groupByAttrIndexToDic(data, index_of_attr):
    results = {}
    for row in data:
        index_value = row[index_of_attr]
        if index_value not in results:
            results[index_value] = []
        results[index_value].append(row)
    return results


emergencyCodeToPatternIndex = {
    1: 4,
    2: 3,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    8: 1,
    7: 2
}


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
