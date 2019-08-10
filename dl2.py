import numpy as np
import sys

def damerau_levenshtein_distance(string1, string2):
    """
    Given two strings `string1` and `string2` compute the Damerau-Levenshtein
    between them.
    """
    n1 = len(string1)
    n2 = len(string2)
    return get_levenshtein_distance_matrix(string1, string2)[n1, n2]

def get_basic_operations(string1, string2):
    dist_matrix = get_levenshtein_distance_matrix(string1, string2)
    i, j = dist_matrix.shape
    i -= 1
    j -= 1
    # operations conducted on string1 in order to transform it into string2
    operations = []
    while i != -1 and j != -1:
        if i > 1 and j > 1 and string1[i-1] == string2[j-2] \
                 and string1[i-2] == string2[j-1]:
            if dist_matrix[i-2, j-2] < dist_matrix[i, j]:
                operations.insert(0, ('transpose', i - 1, i - 2))
                i -= 2
                j -= 2
                continue
        index = np.argmin([dist_matrix[i-1, j-1], \
                           dist_matrix[i, j-1], \
                           dist_matrix[i-1, j]])
        if index == 0:
            if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
                operations.insert(0, ('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif index == 1:
            operations.insert(0, ('insert', i - 1, j - 1))
            j -= 1
        elif index == 2:
            operations.insert(0, ('delete', i - 1, i - 1))
            i -= 1
    return operations

def execute_operations(operations, string1, string2):
    # initialise the path from string1 to string2 with the first phase, i.e.
    # the string1 itself
    strings = [string1]
    # get single letters from the input string
    string = list(string1)

    shift = 0
    for op in operations:
        i, j = op[1], op[2]
        if op[0] == 'delete':
            del string[i + shift]
            shift -= 1
        elif op[0] == 'insert':
            string.insert(i + shift + 1, string2[j])
            shift += 1
        elif op[0] == 'replace':
            string[i + shift] = string2[j]
        elif op[0] == 'transpose':
            string[i + shift], string[j + shift] = string[j + shift], string[i + shift]
        strings.append(''.join(string))
    return strings

def get_levenshtein_distance_matrix(string1, string2):
    """
    Given two strings create a matrix that contains the Damerau-
    Levenshtein distance in the cell [n1, n2]
    """
    n1 = len(string1)
    n2 = len(string2)
    dl_matrix = np.zeros((n1 + 1, n2 + 1), dtype=int)
    # the first column: turn input (sub)string into an empty string
    for i in range(n1 + 1):
        dl_matrix[i, 0] = i
    # the first row: turn an empty string into the target (sub)string
    for j in range(n2 + 1):
        dl_matrix[0, j] = j
    # the matrix is analysed row-wise
    for i in range(n1):
        for j in range(n2):
            cost = 0 if string1[i] == string2[j] else 1

            # consider operations allowed in Levenshtein's distance
            dl_matrix[i+1, j+1] = min(dl_matrix[i, j+1] + 1, # insertion
                                      dl_matrix[i+1, j] + 1, # deletion
                                      dl_matrix[i, j] + cost) # substitution
            if i > 0 and j > 0 and string1[i] == string2[j-1] and \
            string1[i-1] == string2[j]:
                # handle transposition
                dl_matrix[i+1, j+1] = min(dl_matrix[i+1, j+1], \
                                        dl_matrix[i-1, j-1] + cost)
    return dl_matrix

if __name__ == "__main__":
    pass
