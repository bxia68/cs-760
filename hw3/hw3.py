import numpy as np
import pandas as pd


def knn(train_set: np.ndarray, test_point: np.ndarray, k: int) -> int:
    closest = np.zeros([k, 2])
    for i, train_point in enumerate(train_set):
        dist_vector = test_point - train_point[:-1]

        norm = np.linalg.norm(dist_vector)

        if i < k:
            closest[i] = [norm, train_point[-1]]

        largest_index = closest[:, 0].argmax()
        if norm < closest[largest_index, 0]:
            closest[largest_index] = [norm, train_point[-1]]

    return round(np.average(closest[:, 1]))


def read_file(file: str) -> np.ndarray:
    f = open(file, "r")
    file_list = f.readlines()
    array = np.zeros([len(file_list), 3])
    for i, line in enumerate(file_list):
        array[i:] = list(map(float, line.split()))
    return array
