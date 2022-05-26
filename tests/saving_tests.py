import sys
sys.path.append('../src')
sys.path.append('src')
from data_management import get_training_info

get_training_info('../results/data/training/220526/training_nr0001.dat.npz')
