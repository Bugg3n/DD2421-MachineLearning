import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

P = None
alpha = None
non_zero_alphas = None
targets = None
datapoints = None
globalkernel = 0

def computematrixP():
    global datapoints
    global targets
    global P 
    #You can pre-compute a matrix with these values: Pi,j = ti*tj*kernal(xi;xj), P should be a NxN matrix
    N = len(datapoints)
    P = numpy.zeros((N, N))  
    for i in range(N):
        for j in range(N):
            P[i, j] = targets[i] * targets[j] * kernel(datapoints[i], datapoints[j])  

def objective(alpha):
    global P
    #1/2* ∑ over i(∑ over j(ai*aj*P[i,j])) -sum over i(ai)
    
    double_sum = 0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
             double_sum += alpha[i] * alpha[j] * P[i, j]

    result = 0.5 * double_sum - numpy.sum(alpha)
    
    return result

def kernel(datapoint1, datapoint2):
    global globalkernel
    if globalkernel == 0:
        # Linear Kernel
        return numpy.dot(datapoint1, datapoint2)
    elif globalkernel == 1:
        # Polynomial kernel
        coef = 1
        degree = 2
        return (numpy.dot(datapoint1, datapoint2) + coef) ** degree
    elif globalkernel == 2:
        # Radial Basis Kernel
        gamma = 1.0
        return numpy.exp(-gamma * numpy.linalg.norm(datapoint1 - datapoint2)**2)
    #Start with the linear kernel which is the same as an ordinary scalar product, but also
    #explore the other kernels in section 3.3
    #print("datapunkt1 " + str(datapoint1))
    #sum(xi * yi for xi, yi in zip(datapoint1, datapoint2))
    #numpy.dot(datapoint1, datapoint2)
    


def zerofun(al):
    global targets
    #use numpy.dot to be efficient sum over i(aiti)
    #print(numpy.dot(al, targets))
    return numpy.dot(al, targets)

def eq7():
    global targets
    global datapoints
    global alpha
    global non_zero_alphas

    suport_vector = [entry['data_point'] for entry in non_zero_alphas][0]
    suport_vector_target = [entry['target'] for entry in non_zero_alphas][0]
    index = [entry['index'] for entry in non_zero_alphas]
    b = 0
    for i in index:
        # chat gpt b += alpha[i] * support_vector_targets[i] * linearkernel(support_vectors, datapoints[i]) 
        b += alpha[i] * targets[i] * kernel(suport_vector, datapoints[i])
    b -= suport_vector_target
    print("b: " + str(b))
    return b

def predict(s,b):
    global targets
    global datapoints
    global alpha
    global non_zero_alphas

    index = [entry['index'] for entry in non_zero_alphas]
    prediction = 0
    for i in index:
        # chat gpt b += alpha[i] * support_vector_targets[i] * linearkernel(support_vectors, datapoints[i]) 
        prediction += alpha[i] * targets[i] * kernel(s, datapoints[i])
    
    prediction -= b
    return prediction

def input():
    global datapoints
    global targets
    
    classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1, 0.2], 
    numpy.random.randn(10, 2) * 0.5 + [-1, 1]))
    classB = numpy.random.randn(20, 2) * 0.5 + [-0.0, -0.0]

    datapoints = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
    -numpy.ones(classB.shape[0])))

    N = datapoints.shape[0]  # Number of rows (samples)
    permute = list(range(N))
    numpy.random.shuffle(permute)
    datapoints = datapoints[permute, :]
    targets = targets[permute]
    return classA,classB

def main():
    global alpha
    global datapoints
    global non_zero_alphas
    global targets

    classA, classB = input()
    computematrixP()
    #print("matrisen datapunkter" + str(datapoints))
    #print("targets " + str(targets))
    #print("matrisen P" + str(P))

    C = 0.01
    B=[(0, C) for b in range(len(targets))]
    XC={'type':'eq', 'fun': lambda at: zerofun(at)}

    start = numpy.zeros(len(targets))
    ret = minimize(objective, start, bounds=B, constraints=XC)
    #minimize returns a dictionary data structure; use the string 'x' as an index to pick out the actual alpha values.
    alpha = ret['x'] 
    print("success: ")
    print(ret['success'] )
    #print("alpha: ")
    #print(alpha)
    #allt alpha blir 0
    threshold = 1e-5
    non_zero_alphas = []
    for i in range(len(alpha)):
        if alpha[i] > threshold:
            non_zero_alphas.append({'alpha': alpha[i], 'data_point': datapoints[i], 'target': targets[i], 'index': i})
    
    b = eq7()
    #list1 = [1,2]
    #vector1 = numpy.array(list1)
    #print(predict(vector1,b))
    
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.') 
    x_grid = numpy.linspace(-5, 5)
    y_grid = numpy.linspace(-4, 4)
    grid = numpy.array([[predict([x, y], b) for x in x_grid] for y in y_grid])
    #print(grid)
    #print(grid)
    plt.contour(x_grid, y_grid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.axis('equal')  # Force same scale on both axes
    plt.savefig('svmplot.pdf')  # Save a copy in a file
    plt.show() 
main()