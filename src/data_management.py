""" Functions used to save, load and get information about data"""
import os.path
import datetime
import numpy as np

def save_training(schedule, cost_lists):
    data_folder = '../results/data/'
    data_folder += 'training/'
    data_folder = create_todays_subfolder(data_folder)
    filename = create_filename(data_folder, data_type='training')
    np.savez(filename, 
             qubits = np.shape(schedule['ansaetze'][-1].gate_sequence)[1],
             dev = schedule['device'],
             random_seed = schedule['seed'],
             schedule_name = schedule['name'],
             optimizer_list = schedule['optimizers'],
             ansatz_list = schedule['ansaetze'],
             layers = [np.shape(a.gate_sequence)[0] for a in schedule['ansaetze']],
             initial_weights = schedule['initial weights'],
             training_obs = schedule['training observables'],
             monitoring_obs = schedule['monitoring observables'],
             label_list = ['Training cost'] + schedule['labels'],
             max_iter_list = [int(i) for i in schedule['iterations']],
             cost_data = cost_lists, 
             allow_pickle=True)

def create_todays_subfolder(data_folder):
    data_folder += '{}'.format(datetime.datetime.now().strftime("%y%m%d"))
    try:
        os.makedirs(data_folder)    
        print("Directory      " , data_folder ,  " created ")
    except FileExistsError:
        print("Directory      " , data_folder ,  " already exists")
    return data_folder + '/'

def create_filename(data_folder, data_type):
    filename = data_folder + data_type + '_nr' + "%04d"%1 + '.dat'
    while os.path.isfile(filename+'.npz'):
        # try:
        current_count = int(''.join([s for s in filename if s.isdigit()])[-4:]) + 1
        filename = data_folder + 'training_nr' + "%04d"%current_count + '.dat'
        # except (TypeError, ValueError):
        #     filename = filename[0:-4] + '_2.csv'
    print("Saving data in ", filename)
    return filename


def get_training_info(file):
    #TODO: test functionality
    data = np.load(file, allow_pickle=True)
    print("Schedule      : ", data['schedule_name'])
    print("Qubits        : ", data['qubits'])
    print("Random seed   : ", data['random_seed'])
    print("Device        : ", data['dev'])
    for phase in range(len(data['ansatz_list'])):
        print("Phase ", phase+1)
        print("  Optimizer           : ", data['optimizer_list'][phase])
        print("  Ansatz              : ", data['ansatz_list'][phase])
        print("  Depth               : ", data['layers'][phase], " layers")
        print("  Training observable :", data['training_obs'][phase])
        print("  Iterations          :", data['max_iter_list'][phase])
    print("Monitoring observables: ")
    for obs in data['monitoring_obs']:
        print(obs)
    

def get_training_costs(file):
    data = np.load(file)
    #TODO: check format of output variable
    return data['cost_data']

def get_training_labels(file):
    data = np.load(file, allow_pickle=True)
    return data['label_list']


