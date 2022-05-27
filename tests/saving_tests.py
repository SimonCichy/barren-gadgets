import sys
sys.path.append('../src')
sys.path.append('src')
from data_management import get_training_info, get_training_costs, get_training_labels

file = '../results/data/training/220527/training_nr0001.dat.npz'
get_training_info(file)
costs = get_training_costs(file)

print('test')
