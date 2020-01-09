# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2020-01-09

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.dataloader import DataLoader
from utils.logisticregression import LogisticRegressionModel


class TorchTrain(object):
    def __init__(self):
        self.batch_size = 100
        self.n_iters = 3000
        self.epoch = None
        self.learn_rate = 0.001
        self.base_dir = os.path.dirname(__file__)

    def set_num_epochs(self, train_data):
        self.epoch = int(self.n_iters / (len(train_data) / self.batch_size))

    @staticmethod
    def _set_loss():
        criterion = nn.CrossEntropyLoss()
        return criterion

    @staticmethod
    def _set_loader(train_data, test_data):
        train_loader = torch.utils.data.DataLoader(train_data)
        test_loader = torch.utils.data.DataLoader(test_data)
        return train_loader, test_loader

    def main(self):
        dl = DataLoader(os.path.join(self.base_dir, "static"))
        train_data, test_data = dl.data_loader()
        self.set_num_epochs(train_data)
        train_loader, test_loader = TorchTrain._set_loader(train_data, test_data)
        input_dim = 28 * 28
        output_dim = 10
        model = LogisticRegressionModel(input_dim, output_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learn_rate)
        iters = 0
        criterion = TorchTrain._set_loss()
        for epoch in range(self.epoch):
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, 28 * 28))
                labels = Variable(labels)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                iters += 1
                if iters % 500 == 0:
                    correct = 0
                    total = 0
                    for test_images, test_labels in test_loader:
                        test_images = Variable(test_images.view(-1, 28 * 28))
                        test_outputs = model(test_images)
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    accuracy = 100 * correct / total
                    print("Iteration:", iters, "loss:", loss.data[0], "Accuracy:", accuracy)
        torch.save(model.state_dict(), os.path.join(self.base_dir, "output", "mnist.pth"))

    def load_model(self):
        path = os.path.join(self.base_dir, "output")
        input_dim = 28 * 28
        output_dim = 10
        model = LogisticRegressionModel(input_dim, output_dim)
        model.load_state_dict(torch.load(path))
        model.eval()
        test_image = "xxx"
        with torch.no_grad():
            output = model(test_image)
        _, predict = torch.max(output, 1)
        classIndex = predict[0]
        print("预测结果：", classIndex)


if __name__ == '__main__':
    tt = TorchTrain()
    tt.main()
