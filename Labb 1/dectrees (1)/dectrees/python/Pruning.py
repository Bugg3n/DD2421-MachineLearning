import random
import dtree as d
import monkdata as m
import matplotlib.pyplot as plt

import statistics

def best_tree(current_tree):
    found = False
    for new_tree in d.allPruned(current_tree):
        if d.check(new_tree, monkval) > d.check(current_tree, monkval):
            found = True
            current_tree = new_tree

    if found:
        current_tree = best_tree(current_tree)
    return current_tree



def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


if __name__== '__main__':

    dataset_names = ('MONK-1', 'MONK-3')
    datasets = (m.monk1, m.monk3)
    datasets_test = (m.monk1test, m.monk3test)

    fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    n = 1000

    for i in range(len(dataset_names)):
        data = []
        mean_errors = []
        stdev = []

        for fraction in fractions:
            errors = []

            for j in range (n):
                monktrain, monkval = partition(datasets[i], fraction)
                tree = d.buildTree(monktrain, m.attributes)
                best = best_tree(tree)

                errors.append(1 - d.check(best, datasets_test[i]))            

            mean_error = statistics.mean(errors)
            mean_errors.append(statistics.mean(errors))

            stdev.append(statistics.stdev(errors))

            data.append([fraction, mean_error, statistics.mean(stdev)])

        print(data)


        plt.errorbar(fractions, mean_errors, yerr=stdev, marker='o')

        plt.title('{} (n = {})'.format(dataset_names[i], n))
        plt.xlabel('fraction')
        plt.ylabel('mean error')
        plt.show()


# Assignment 6
# High Bias: When the model overtrain on certain data, becoming too complex and specified for the training data
# Low Bias: Pruning helps reduce the complexity of the tree by removing branches that do not significnatlly imporve its performance
# This can lead to a lower bias as it allows the model to capture more relevant patterns.

# High variance: An unpruned decision tree is often highly sensitive to the specific training data it was trained on. It can adapt 
# itself too closely to the noise in the data, resulting in a model with high variance. Such a model will perform well on the 
# training data but poorly on new, unseen data because it essentially memorizes the training data instead of generalizing from it.
# Low vairance: Pruning helps reduce the variance by simplifying the tree structure. It removes branches that may have overfit the 
# training data by capturing noise or outliers. As a result, the pruned tree tends to be more robust and less sensitive to the 
# training data's noise. 

# Pruning creates a balance between bias and variance by removing complexity while keeping essential information about the data.

