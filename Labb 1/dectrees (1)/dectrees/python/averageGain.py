import dtree as d
import monkdata as m

# Assignment 3, 4



def averageGain(dataset_names, datasets):

    data = []

    for i in range(len(datasets)):
        row = [dataset_names[i]]

        for j in range(6):
            row.append(d.averageGain(datasets[i], m.attributes[j]))

        data.append(row)

    print(data)

    if __name__ == '__main__':
        dataset_names = ('MONK-1', 'MONK-2', 'MONK-3')
        datasets = (m.monk1, m.monk2, m.monk3)
        averageGain()

# [['MONK-1', 0.07527255560831925, 0.005838429962909286, 0.00470756661729721, 0.02631169650768228, 0.28703074971578435, 0.0007578557158638421], 
# ['MONK-2', 0.0037561773775118823, 0.0024584986660830532, 0.0010561477158920196, 0.015664247292643818, 0.01727717693791797, 0.006247622236881467], 
# ['MONK-3', 0.007120868396071844, 0.29373617350838865, 0.0008311140445336207, 0.002891817288654397, 0.25591172461972755, 0.007077026074097326]]

# Attribute 5 has the highest gain on MONK-1 and MONK-2, but in MONK-3 attribute 2 is the highest.

# When information gain is maximized, it means that the attribute A selected for the split results in the largest 
# reduction in uncertainty or entropy. Therefore, the entropy of the subsets, Sk, will be minimized or reduced to the 
# greatest extent possible.

# A reduction in entropy means that the subset is mor homogenus, meaning we get a better classification into the groups if we use 
# the attribute with the highest gain. 