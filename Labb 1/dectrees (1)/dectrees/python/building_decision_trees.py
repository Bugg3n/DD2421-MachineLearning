import dtree as d
import monkdata as m
import drawtree_qt5 as drawtree
import averageGain as averageGain

# First part
print("-----------First Part-----------")
subsets = []    
for value in m.attributes[4].values:
    subsets.append(d.select(m.monk1, m.attributes[4], value))
    subset_names = list('Subset{}'.format(i) for i in range(1, len(subsets) + 1))
subsetGain = averageGain.averageGain(subset_names, subsets)
#[['Subset1', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
# ['Subset2', 0.040216841609413634, 0.015063475072186083, 0.03727262736015946, 0.04889220262952931, 0.0, 0.025807284723902146], 
# ['Subset3', 0.03305510013455182, 0.002197183539100922, 0.017982293842278896, 0.01912275517747053, 0.0, 0.04510853782483648], 
# ['Subset4', 0.20629074641530198, 0.033898395077640586, 0.025906145434984817, 0.07593290844153944, 0.0, 0.0033239629631565126]]

# Probably attribute 4 should be used next.

print("-----------Second Part-----------")

tree = d.buildTree(m.monk1, m.attributes, maxdepth=2)
# drawtree.drawTree(tree)
data = []
for i in range(len(subsets)):
    row = [subset_names[i]]
    row.append(d.mostCommon(subsets[i]))
    data.append(row)
print(data)

# Most Common
# [['Subset1', True], 
# ['Subset2', False], 
# ['Subset3', False], 
# ['Subset4', False]]

print("-----------Third Part-----------")
tree = d.buildTree(m.monk1, m.attributes, maxdepth=2)
# tree = d.buildTree(m.monk1, m.attributes)  # All levels
# drawtree.drawTree(tree)  # Show tree
print(tree)

# Assignment 5
print("-----------Assignment 5-----------")
t1=d.buildTree(m.monk1, m.attributes)
print(d.check(t1, m.monk1test))
print(d.check(t1, m.monk1))

t2=d.buildTree(m.monk2, m.attributes)
print(d.check(t2, m.monk2test))
print(d.check(t2, m.monk2))

t3=d.buildTree(m.monk3, m.attributes)
print(d.check(t3, m.monk3test))
print(d.check(t3, m.monk3))

# 0.8287037037037037
# 1.0
# 0.6921296296296297
# 1.0
# 0.9444444444444444
# 1.0


