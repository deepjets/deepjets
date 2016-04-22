
from deepjets.bayesopt import ObjectiveFunction

files = ['datasets/test_train_kf0.h5', 'datasets/test_train_kf1.h5']
objective = ObjectiveFunction('test', files)
print objective([(0.001, 1024), (0.001, 512), (0.001, 256), (0.001, 128), (0.001, 64)])
