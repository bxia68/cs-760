from scipy.interpolate import lagrange
import random
from sklearn import tree
from hw2 import read_file, find_best_split, make_tree, print_tree, test_sklearn_tree, test_tree
import numpy as np
import matplotlib.pyplot as plt


def q2_3():
    instances_list = read_file('data/Druns.txt')
    find_best_split(instances_list)


def q2_4():
    instances_list = read_file("data/D3leaves.txt")
    root = make_tree(instances_list)
    print_tree(root)


def q2_5():
    instances_list = read_file('data/D1.txt')
    root = make_tree(instances_list)
    print_tree(root)


def q2_6():
    instances_list = read_file('data/D2.txt')
    root = make_tree(instances_list)

    feature0 = [instance.features[0] for instance in instances_list]
    feature1 = [instance.features[1] for instance in instances_list]
    label = [instance.label for instance in instances_list]
    pred = [root.test(instance.features) for instance in instances_list]

    plt.scatter(feature0, feature1, c=label)
    plt.show()
    # plot(root, 'D1')


def q2_7():
    instances_list = read_file('data/Dbig.txt')
    random.shuffle(instances_list)
    training_set = instances_list[:8192]
    test_set = instances_list[8192:]

    for i in range(5):
        size = pow(2, 5 + i * 2)
        current_training_set = training_set[:size]
        print('D_{}:'.format(size))
        root = make_tree(current_training_set)
        print('node_count_{} = {}'.format(size, root.count_nodes()))
        print('err_{} = {}'.format(size, test_tree(root, test_set)))
        # plot(root, 'D_{}:'.format(size))


def q2_7_2():
    node_count = [13, 19, 55, 139, 281]
    err = [0.11172566371681415, 0.0702433628318584, 0.05918141592920354, 0.046460176991150445, 0.020464601769911505]
    plt.scatter(node_count, err)
    plt.xlabel('node count')
    plt.ylabel('err')
    plt.title('learning curve')

    plt.show()


def q3():
    instances_list = read_file('data/Dbig.txt')
    random.shuffle(instances_list)
    training_set = instances_list[:8192]
    test_set = instances_list[8192:]

    node_count = []
    err_list = []
    for i in range(5):
        size = pow(2, 5 + i * 2)
        current_training_set = training_set[:size]
        print('D_{}:'.format(size))
        X = [instance.features for instance in current_training_set]
        Y = [instance.label for instance in current_training_set]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, Y)
        node_count.append(clf.tree_.node_count)
        err_list.append(test_sklearn_tree(clf, test_set))
        print('node_count_{} = {}'.format(size, clf.tree_.node_count))
        print('err_{} = {}'.format(size, test_sklearn_tree(clf, test_set)))

    plt.scatter(node_count, err_list)
    plt.xlabel('node count')
    plt.ylabel('err')
    plt.title('learning curve')
    plt.show()


def q4():
    X_train = np.linspace(0, 2 * np.pi, 100)

    Y_train = np.sin(X_train)
    f = lagrange(X_train, Y_train)

    X_test = np.linspace(0, 2 * np.pi, 1000)
    Y_test = np.sin(X_test)

    train_error = np.mean(np.abs(Y_train - f(X_train)))
    test_error = np.mean(np.abs(Y_test - f(X_test)))

    print('train_error: {}'.format(train_error))
    print('test_error: {}'.format(test_error))

    std_dev = 10
    noise = np.random.normal(0, std_dev, 100)
    X_noise = X_train + noise
    f = lagrange(X_noise, Y_train)

    train_error = np.mean(np.abs(Y_train - f(X_noise)))
    noise_test_error = np.mean(np.abs(Y_test - f(X_test)))

    print('train_error: {}'.format(train_error))
    print('noise_test_error: {}'.format(noise_test_error))

# q2_3()
# q2_4()
# q2_5()
# q2_6()
# q2_7()
# q2_7_2()
# q3()
# q4()
