import dtree as d
import monkdata as m

# Assignment 1, 2

if __name__ == '__main__':

    data = [
        ['MONK-1', d.entropy(m.monk1)],
        ['MONK-2', d.entropy(m.monk2)],
        ['MONK-3', d.entropy(m.monk3)]
    ]

    print(data)

    # [['MONK-1', 1.0], ['MONK-2', 0.957117428264771], ['MONK-3', 0.9998061328047111]]

    # Entropy for a uniform distribution
    # Maximum entropy (maximum randomness), for example a fair six-sided die, or a coin-toss
    # Low entropy (lower randomness), When all outcomes are not equally liekely. For example a fixed die that has a higher probability of showing 6.