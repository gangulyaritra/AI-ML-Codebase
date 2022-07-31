import csv
import random

"""
The `load_from_csv()` function should have one parameter, a file name.
This function reads the CSV file and returns a matrix (list of lists) in which a row in the file is a row in the matrix.
"""


def load_from_csv(csv_name):
    matrix = []
    with open(csv_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            matrix.append(row)
        for (i, row) in enumerate(matrix):
            for (j, element) in enumerate(row):
                matrix[i][j] = float(element)
    return matrix


def getMatrix():
    file = load_from_csv("Data.csv")
    return file


"""
The `get_distance()` function has two parameters, both of them are lists.
This function returns the Manhattan distance between the two lists.
"""


def get_distance(a, b):
    manhattan = 0
    for i in range(len(a)):
        manhattan += abs(a[i] - b[i])
    return manhattan


"""
The `get_max()` function has two parameters, a matrix (list of lists) and a column number.
This function looks into all the elements of the data matrix in this column number and returns the highest value.
"""


def get_max(matrix, ncol):
    large = matrix[0][ncol]
    for i in range(len(matrix)):
        if matrix[i][ncol] > large:
            large = matrix[i][ncol]
    return large


"""
The `get_min()` function has two parameters, a matrix (list of lists) and a column number.
This function looks into all the elements of the data matrix in this column number and returns the lowest value.
"""


def get_min(matrix, ncol):
    small = matrix[0][ncol]
    for i in range(len(matrix)):
        if matrix[i][ncol] < small:
            small = matrix[i][ncol]
    return small


"""
The `get_standardised_matrix()` function takes one parameter, a matrix (list of lists).
This function returns a matrix containing the standardized version of the matrix passed as a parameter.
This function somehow use the `get_max()` and `get_min()` functions.
"""


def get_standardised_matrix(matrix):
    array = []
    for j in range(len(matrix[0])):
        summation = 0
        n = len(matrix)
        for i in range(n):
            summation += matrix[i][j]
        average = summation / n
        denominator = get_max(matrix, j) - get_min(matrix, j)
        for i in range(n):
            array.append(((matrix[i][j]) - average) / denominator)
    return array


"""
The `sortmatrix()` function has two parameters: a matrix (list of lists), and a column number.
This function returns all the elements (in ascending order) in the data matrix at the column number passed as a parameter.
"""


def sortmatrix(matrix, ncol):
    n = len(matrix)
    array = []
    for i in range(n):
        for j in range(0, n - i - 1):
            if matrix[j][ncol] > matrix[j + 1][ncol]:
                temp = matrix[j][ncol]
                matrix[j][ncol] = matrix[j + 1][ncol]
                matrix[j + 1][ncol] = temp
    for i in range(n):
        array.append(matrix[i][ncol])
    return array


"""
The `get_median()` function has two parameters: a matrix (list of lists) and a column number.
This function returns the median of the values in the data matrix at the column number passed as a parameter.
This function somehow use the `sortmatrix()` function.
"""


def get_median(matrix, ncol):
    median = 0
    n = len(matrix)
    array = sortmatrix(matrix, ncol)
    if n % 2 == 0:
        median = (array[int((n - 1) / 2)] + array[int(n / 2)]) / 2
    else:
        median = array[int(n / 2)]
    return median


"""
The `get_centroids()` function have three parameters: (i) a matrix (list of lists), (iii) the list S, (iii) the value of K.
This function implements Step 6 of the algorithm described in the appendix.
This function returns a list containing K elements. Each of these elements is also a list.
"""


def get_centroids(matrix, S, K):
    cluster_list = []
    for i in range(K):
        c = []
        for j in range(len(S)):
            if S[j] == i + 1:
                c.append(j)
        array = []
        for j in range(len(matrix)):
            for k in range(len(c)):
                if j == c[k]:
                    array.append(matrix[j])
        cluster_value = []
        for col in range(len(array[0])):
            median = get_median(array, col)
            cluster_value.append(median)
        cluster_list.append(cluster_value)
    return cluster_list


"""
The `get_groups()` function has two parameters: a data matrix (list of lists) and the number of groups to be created (K).
This function follows the algorithm described in the appendix. It should return a list S (defined in the appendix).
"""


def get_groups(matrix, K):
    if K > 1 and K < 178:
        lists = []
        for i in range(len(matrix)):
            lists.append(i)
        rows = random.choices(tuple(lists), k=K)
        c = []
        for i in rows:
            temp = []
            for j in range(len(matrix[i])):
                temp.append(matrix[i][j])
            c.append(temp)
        d = []
        small = 0
        for i in range(len(matrix)):
            dist = []
            for j in range(len(c)):
                dist.append(get_distance(matrix[i], c[j]))
            d.append(dist)
        transpose = []
        for j in range(len(d[0])):
            temp = []
            for i in range(len(d)):
                temp.append(d[i][j])
            transpose.append(temp)
        S = []
        for i in range(len(transpose[0])):
            small = get_min(transpose, i)
            for j in range(len(transpose)):
                if transpose[j][i] == small:
                    S.append(j + 1)
    return S


"""
The `goto_step4()` function has two parameters: a data matrix (list of lists) and a list.
This function follows the algorithm described in Step 4 of the appendix. It should return a list S.
"""


def goto_step4(matrix, c):
    d = []
    small = 0
    for i in range(len(matrix)):
        dist = []
        for j in range(len(c)):
            dist.append(get_distance(matrix[i], c[j]))
        d.append(dist)
    transpose = []
    for j in range(len(d[0])):
        temp = []
        for i in range(len(d)):
            temp.append(d[i][j])
        transpose.append(temp)
    S = []
    for i in range(len(transpose[0])):
        small = get_min(transpose, i)
        for j in range(len(transpose)):
            if transpose[j][i] == small:
                S.append(j + 1)
    return S


"""
The aim of this function is just to run a series of tests.
Here we can use hard-coded values for the strings containing the filenames of data and values for K.

It runs the algorithm, and shows on the screen how many entities have been assigned to each group.
You should run experiments with K = 2, 3, 4, 5, 6.
"""


def run_test():
    array = getMatrix()
    K = int(input("Determine the number of clusters: "))
    S = get_groups(array, K)
    iterations = 50
    for i in range(iterations):
        clusters = get_centroids(array, S, K)
        S = goto_step4(array, clusters)
    for i in range(K):
        c = []
        for j in range(len(S)):
            if S[j] == i + 1:
                c.append(j)
        print("Cluster ", i + 1, ": ", c)


if __name__ == "__main__":
    run_test()
