import data_service, plotting_service, ann_service
from ann import ANN
import sys


class Runner:

    def __init__(self):
        pass

    def run(self, lambd = 0, keep_prob = 1):

        train_X, train_Y, test_X, test_Y = data_service.load_2D_dataset()

        ann = ANN()
        learning_rate = 0.3
        parameters, costs = ann.fit(train_X, train_Y, learning_rate=learning_rate, lambd=lambd, keep_prob=keep_prob)

        plotting_service.plot_loss_per_iteration_for_learning_rate(costs, learning_rate)

        plotting_service.plot_decision_boundary(lambda x: ann_service.predict_dec(parameters, x.T), train_X, train_Y)


if __name__ == "__main__":

    lambd = float(sys.argv[1])
    keep_prob = float(sys.argv[2])
    Runner().run(lambd, keep_prob)
