""" Functions used to save, load and get information about data"""
import os.path
import datetime
import numpy as np

def save_training(schedule, cost_lists, mode='new file', runtime=None):
    data_folder = '../results/data/'
    data_folder += 'training/'
    data_folder = create_todays_subfolder(data_folder, mode=mode)
    filename = create_filename(data_folder, data_type='training', mode=mode)
    np.savez(filename, 
             qubits = np.shape(schedule['ansaetze'][-1].gate_sequence)[1],
             dev = {
                 'name': schedule['device'].short_name,
                 'wires': schedule['device'].wires,
                 'shots': schedule['device'].shots, 
                 'version': schedule['device'].version,
             },
             random_seed = schedule['seed'],
             schedule_name = schedule['name'],
             optimizer_list = [type(opt) for opt in schedule['optimizers']],
             step_list = [opt.stepsize for opt in schedule['optimizers']],
             ansatz_list = [a.name for a in schedule['ansaetze']],
             gate_sequence_list = [a.gate_sequence for a in schedule['ansaetze']],
             layers = [np.shape(a.gate_sequence)[0] for a in schedule['ansaetze']],
             initial_weights = schedule['initial weights'],
             training_obs = schedule['training observables'],
             monitoring_obs = schedule['monitoring observables'],
            #  monitoring_obs_coeffs = [obs.coeffs for obs in schedule['monitoring observables']],
            #  monitoring_obs_ops = np.asanyarray([str(obs.ops) for obs in schedule['monitoring observables']], dtype=str),
             label_list = ['Training cost'] + schedule['labels'],
             max_iter_list = [int(i) for i in schedule['iterations']],
             cost_data = cost_lists, 
             runtime = runtime,
             allow_pickle=True)

def save_training2(schedule, cost_lists, mode='new file', runtime=None):
    data_folder = '../results/data/'
    data_folder += 'training/'
    data_folder = create_todays_subfolder(data_folder, mode=mode)
    filename = create_filename(data_folder, data_type='training', mode=mode)
    np.savez(filename, 
             gate_sequence_list = [a.gate_sequence for a in schedule['ansaetze']],
             initial_weights = schedule['initial weights'],
             max_iter_list = [int(i) for i in schedule['iterations']],
             step_list = [opt.stepsize for opt in schedule['optimizers']],
             cost_data = cost_lists, 
             allow_pickle=False)
    if mode == 'new file': 
        with open(filename + '.txt', 'w') as f:
            f.write(
                'qubits :        ' + 
                str(np.shape(schedule['ansaetze'][-1].gate_sequence)[1]) + '\n' +
                'device :        ' + '\n' +
                '- name :        ' + schedule['device'].short_name + '\n' +
                '- wires :       ' + str(schedule['device'].wires) + '\n' +
                '- shots :       ' + str(schedule['device'].shots) + '\n' +
                '- version :     ' + str(schedule['device'].version) + '\n' +
                'random_seed :   ' + str(schedule['seed']) + '\n' +
                'schedule_name : ' + schedule['name'] + '\n'
                )
            optimizer_list = ['- ' + str(type(opt)) for opt in schedule['optimizers']]
            f.write('optimizers : \n')
            f.write('\n'.join(optimizer_list))
            f.write('\n')
            ansatz_list = ['- ' + a.name for a in schedule['ansaetze']]
            f.write('ansätze : \n')
            f.write('\n'.join(ansatz_list))
            f.write('\n')
            f.write(
                'layers        : ' +
                str([np.shape(a.gate_sequence)[0] for a in schedule['ansaetze']])
            )
            f.write('\n')
            training_obs = ''
            for o, obs in enumerate (schedule['training observables']):
                training_obs += '- phase ' + str(o) + '\n'
                #  /!\ there is a \n that sneeks in there for some reason
                training_obs += str(obs.coeffs) + '\n'
                training_obs += str(obs.ops) + '\n'
            f.write('training observables : \n')
            f.write(training_obs)
            label_list = ['Training cost'] + schedule['labels']
            f.write('labels : \n - ')
            f.write('\n - '.join(label_list))
            f.write('\n')
    else:
        with open(filename + '.txt', 'a') as f:
            f.write('runtime: ' + str(runtime) + '\n')

def create_todays_subfolder(data_folder, mode='new file'):
    data_folder += '{}'.format(datetime.datetime.now().strftime("%y%m%d"))
    if mode == 'new file':
        try:
            os.makedirs(data_folder) 
            print("Directory      " , data_folder ,  " created ")
        except FileExistsError:
            print("Directory      " , data_folder ,  " already exists")
    return data_folder + '/'

def create_filename(data_folder, data_type, mode='new file'):
    # filename = data_folder + data_type + '_nr' + "%04d"%1 + '.dat'
    filename = data_folder + data_type + '_nr' + "%04d"%1
    while os.path.isfile(filename+'.npz'):
        current_count = int(''.join([s for s in filename if s.isdigit()])[-4:]) + 1
        # filename = data_folder + 'training_nr' + "%04d"%current_count + '.dat'
        filename = data_folder + 'training_nr' + "%04d"%current_count
    if mode == 'overwrite':
        current_count -= 1
        # filename = data_folder + 'training_nr' + "%04d"%current_count + '.dat'
        filename = data_folder + 'training_nr' + "%04d"%current_count
    if mode == 'new file':
        print("Saving data in ", filename)
    return filename


def get_training_info(file):
    #TODO: test functionality
    data = np.load(file, allow_pickle=True)
    print("Schedule      : ", data['schedule_name'])
    print("Qubits        : ", data['qubits'])
    print("Random seed   : ", data['random_seed'])
    print("Device name   : ", data['dev'].item()['name'])
    print("       wires  : ", data['dev'].item()['wires'])
    print("       shots  : ", data['dev'].item()['shots'])
    print("       version: ", data['dev'].item()['version'])
    for phase in range(len(data['ansatz_list'])):
        print("Phase ", phase+1)
        print("  Optimizer           : ", data['optimizer_list'][phase])
        print("  Ansatz              : ", data['ansatz_list'][phase])
        print("  Depth               : ", data['layers'][phase], " layers")
        print("  Training observable : ", data['training_obs'][phase])
        print("  Iterations          : ", data['max_iter_list'][phase])
    print("Monitoring observables: ")
    for o, obs in enumerate(data['monitoring_obs']):
        print(obs)
        # print("  Observable ", o, " - coeffs:", obs[0])
        # print("  Observable ", o, " - ops:   ", 
        #       str(obs[1].ops)[8:-3].split(')), expval('))
    

def get_training_costs(file):
    data = np.load(file)
    #TODO: check format of output variable
    return data['cost_data']

def get_training_labels(file):
    data = np.load(file, allow_pickle=True)
    return data['label_list']


