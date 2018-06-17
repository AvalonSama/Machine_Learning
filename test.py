import matplotlib.pyplot as plt
from numpy import *

threshold = 0
learning_rate = 0.1
weights = [0, 0, 0]
training_set = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]


def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights));

if __name__ == '__main__':
    while True:
        print("-" * 60)
        error_count = 0
        for input_vector, desired_output in training_set:
            print(weights)
            result = dot_product(input_vector, weights) > threshold
            error = desired_output - result
            if error != 0:
                error_count += 1
                for index, value in enumerate(input_vector):
                    weights[index] += learning_rate * error * value
        if error_count == 0:
            break
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for input_vector, desired_output in training_set:
        if desired_output == 1:
            xcord1.append(input_vector[1])
            ycord1.append(input_vector[2])
        else:
            xcord2.append(input_vector[1])
            ycord2.append(input_vector[2])
    fig = plt.figure();
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, marker='+')
    ax.scatter(xcord2, ycord2, s=10)
    x = arange(-1, 2, 0.1);
    y = (-weights[0] - weights[1] * x) / weights[2];
    ax.plot(x, y);
    plt.show();