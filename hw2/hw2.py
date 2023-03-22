from __future__ import annotations

import math
from typing import Optional
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

feature_types = [0, 1]


class Instance:
    def __init__(self, features: list[float], label: int):
        self.features = features
        self.label = label

    def __str__(self):
        return " ".join(map(str, self.features)) + " " + str(self.label)


class Node:
    def __init__(self, left_node: Optional[Node | LeafNode], right_node: Optional[Node | LeafNode], split_feature: int,
                 split_val: float):
        self.left_node = left_node
        self.right_node = right_node
        self.split_feature = split_feature
        self.split_val = split_val

    def __str__(self):
        return 'feature{} >= {}'.format(self.split_feature, self.split_val)

    def test(self, test_features: list[float]) -> int:
        if test_features[self.split_feature] >= self.split_val:
            return self.left_node.test(test_features)
        else:
            return self.right_node.test(test_features)

    def count_nodes(self):
        return self.left_node.count_nodes() + self.right_node.count_nodes() + 1


class Split:
    def __init__(self, left_instances: list[Instance], right_instances: list[Instance], split_feature: int,
                 split_val: float):
        self.left_instances = left_instances
        self.right_instances = right_instances
        self.split_feature = split_feature
        self.split_val = split_val


class LeafNode:
    def __init__(self, label: int):
        self.label = label

    def __str__(self):
        return 'label: {}'.format(self.label)

    def test(self, test_features: list[float]) -> int:
        return self.label

    def count_nodes(self):
        return 1


def find_best_split(instances: list[Instance]) -> Split | None:
    # return None if the instances list is empty
    if not instances:
        return None

    max_info = 0.0
    left_split, right_split = [], []
    split_feature = 0
    split_val = 0

    for feature in feature_types:
        instances = sorted(instances, key=lambda x: x.features[feature])
        prev_val = None
        for index, instance in enumerate(instances):
            # skip instance if it has the same feature value as the previous instance
            if instance.features[feature] == prev_val:
                continue
            current_info = get_gain_ratio(instances, index)
            print('split: feature_{} \\geq {} \\\\'.format(feature, instance.features[feature]))
            # print('GainRatio = {} \\\\'.format(current_info))
            if current_info > max_info:
                max_info = current_info
                split_feature = feature
                split_val = instance.features[feature]
                left_split, right_split = instances[index:], instances[:index]

            prev_val = instance.features[feature]

    # return None if all splits have 0 gain ratio
    if max_info == 0:
        return None

    return Split(left_split, right_split, split_feature, split_val)


def get_gain_ratio(instances: list[Instance], split_index: int) -> float:
    left_split = instances[split_index:]
    right_split = instances[:split_index]

    unconditional_entropy = get_entropy(count_labels(instances), len(instances))
    # print('unconditional_entropy =', unconditional_entropy)
    left_conditional = get_entropy(count_labels(left_split), len(left_split))
    right_conditional = get_entropy(count_labels(right_split), len(right_split))

    conditional_entropy = (len(left_split) / len(instances) * left_conditional +
                           len(right_split) / len(instances) * right_conditional)
    # print('conditional_entropy =', conditional_entropy)
    info_gain = unconditional_entropy - conditional_entropy
    split_entropy = get_entropy((len(left_split), len(right_split)), len(instances))
    print('InfoGain:', info_gain)



    if split_entropy == 0:
    # print('InfoGain:', info_gain)
        return 0

    gain_ratio = info_gain / split_entropy
    return gain_ratio


def get_entropy(y_counts: tuple[int, int], total: int) -> float:
    y0_entropy = 0 if y_counts[0] == 0 else - (y_counts[0] / total * math.log2(y_counts[0] / total))
    y1_entropy = 0 if y_counts[1] == 0 else - (y_counts[1] / total * math.log2(y_counts[1] / total))
    return y0_entropy + y1_entropy


def count_labels(instances: list[Instance]) -> tuple[int, int]:
    l1_count = sum(instance.label for instance in instances)
    l0_count = len(instances) - l1_count

    return l0_count, l1_count


def make_tree(instances: list[Instance]) -> Node | LeafNode:
    split = find_best_split(instances)

    # stopping condition (find_best_split will return None if the stopping condition is met)
    if split is None:
        average_label = sum(instance.label for instance in instances) / len(instances)
        return LeafNode(round(average_label))

    # create child nodes
    left_node = make_tree(split.left_instances)
    right_node = make_tree(split.right_instances)

    current = Node(left_node, right_node, split.split_feature, split.split_val)

    return current


def print_tree(node: Node | LeafNode, indent: str = '') -> None:
    print(indent + str(node))

    if isinstance(node, Node):
        print_tree(node.left_node, indent + ' ' * 4)
        print_tree(node.right_node, indent + ' ' * 4)


def read_file(file: str) -> list[Instance]:
    f = open(file, "r")
    lines = f.readlines()
    instances_list = []
    for line in lines:
        parsed_line = list(map(float, line.split()))
        instances_list.append(Instance(parsed_line[:2], int(parsed_line[2])))
    return instances_list


def test_tree(root: Node, test_set: list[Instance]) -> float:
    error_count = 0
    for instance in test_set:
        if root.test(instance.features) != instance.label:
            error_count += 1

    return error_count / len(test_set)


def test_sklearn_tree(clf, test_set: list[Instance]) -> float:
    X_test = [instance.features for instance in test_set]
    y_true = [instance.label for instance in test_set]
    y_pred = clf.predict(X_test)
    return 1 - accuracy_score(y_pred, y_true)


def plot(root: Node, title: str = None):
    num_points = 10000000

    X = np.random.uniform(-1.5, 1.5, size=(num_points, 2))
    pred = [root.test([X[i, 0], X[i, 1]]) for i in range(num_points)]

    plt.scatter(X[:, 0], X[:, 1], s=3, c=pred)

    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.xlabel('x_0')
    plt.ylabel('x_1')
    plt.title(title)
    plt.show()
