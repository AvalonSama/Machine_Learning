import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

np.random.random(1,2)

class Model():
    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.n_in = 784  # 每张图片是 28 *28 像素
        self.n_out = 10  # 总共有 10个类
        self.max_epochs = 10000  # 最大训练步数 1000步
        self.Weights = np.random.rand(50, self.n_out)  # initialize W 0

        self.Weights1 = np.random.rand(self.n_in, 50)

        self.biases = np.random.rand(self.n_out)  # initialize bias 0
        self.biases1 = np.zeros(50)
        for i in range(self.max_epochs):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)

            self.train(batch_xs, batch_ys, 0.0001)
            if i % 500 == 0:
                accuracy_test = self.compute_accuracy(
                    np.array(mnist.test.images[:500]), np.array(mnist.test.labels[:500]))
                print("#"*30)
                print("compute_accuracy:", accuracy_test)
                print("cross_entropy:", self.cross_entropy(
                    batch_ys, self.output(batch_xs)))  # 输出交叉熵损失函数

    def train(self, batch_x, batch_y, learning_rate):  # 训练数据 更新权重
        #在下面补全（注意对齐空格）
        y = self.output(batch_x);
        temp = np.dot(batch_y,y);
        a = np.add(np.dot(batch_x,self.Weights1),self.biases1);
        




        y = self.output(batch_x);
        dw = -np.multiply(batch_y,y-1);
        db1 = np.add(dw,self.biases);
        dw1 = np.dot(np.transpose(dw),batch_x);
        n,m = shape(self.Weights);
        

        self.Weights = self.Weights + np.transpose(dw) ;
        return

    def output(self, batch_x):  # 输出预测值
        # 注意防止 上溢出和下溢出
        def softmax(x):
            e_x = np.exp(x-np.max(x))
            return e_x / (e_x.sum(axis=0)) + 1e-30  #

        def sigmoid(x):
            return 1/(1+np.exp(-x))
        a = np.add(np.dot(batch_x, self.Weights1), self.biases1)
        prediction = np.add(np.dot(a, self.Weights), self.biases)
        result = []
        for i in range(len(prediction)):
            result.append(softmax(prediction[i]))
        return np.array(result)

    def cross_entropy(self, batch_y, prediction_y):  # 交叉熵函数
        cross_entropy = - np.mean(
            np.sum(batch_y * np.log(prediction_y), axis=1))
        return cross_entropy

    def compute_accuracy(self, xs, ys):  # 计算预测精度
        pre_y = self.output(xs)
        pre_y_index = np.argmax(pre_y, axis=1)
        y_index = np.argmax(ys, axis=1)
        count_equal = np.equal(y_index, pre_y_index)
        count = np.sum([1 for e in count_equal if e])
        sum_count = len(xs)
        return count * 1.0 / sum_count


Model()
