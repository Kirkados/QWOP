"""
Created on Sat Mar 17 22:40:15 2018

Run batches of main.py with a hyperparameter sweep. Automates testing of many hyperparameters
so you can maximize your compute time. Desktop runs ~3x faster than laptop.

Add the best parameters as the defaults in the main.py script. Any you want to sweep across,
just add them here and they will override the defaults set in main.py. So good!

Possibly add multi-core functionality


@author: Kirk
"""
import ddpg

#%% Hyperparameters to sweep
sweep_parameters = {'batch_size':  [8,32,128,256,1024]}

#%% Running each option
#for i in range(len(actor_learning_rate))
names = []
values = []
for name,value in sweep_parameters.items():
       names.append(name)
       values.append(value)
#
#arguments = []
#for_loop_depth = len(names) # how many nested for loops we need
#for i in range(len(values[0])):
#       for j in range(len(values[1])):
#              for k in range(len(values[2])):
#                     argument = {names[0]: values[0][i], names[1]: values[1][j], names[2]: values[2][k]}
#                     arguments.append(argument)
#
#arguments.append({'actor_learning_rate':  0.1,'critic_learning_rate': 0.1,'target_learning_rate': 0.001,'gamma':0.9})
#arguments.append({'actor_learning_rate':  0.1,'critic_learning_rate': 0.01,'target_learning_rate': 0.001,'gamma':0.9})
#arguments.append({'actor_learning_rate':  0.01,'critic_learning_rate': 0.0001,'target_learning_rate': 0.001,'gamma':0.9})
#arguments.append({'actor_learning_rate':  0.001,'critic_learning_rate': 0.0001,'target_learning_rate': 0.001,'gamma':0.99})
#arguments.append({'actor_learning_rate':  0.001,'critic_learning_rate': 0.0001,'target_learning_rate': 0.001,'gamma':0.999})
#arguments.append({'actor_learning_rate':  0.001,'critic_learning_rate': 0.0001,'target_learning_rate': 0.001,'gamma':0.75})
#print(arguments)

for i in range(len(arguments)):
       print('############################################## Hyperparameter sweep ' +str(i+1) + ' of ' + str(len(arguments)) + ' #########################################################')
       ddpg.main(arguments[i])
       

#
#import multiprocessing
#import multiprocessing_import_worker
#
#def worker(args):
#    """thread worker function"""
#    print('running')
#    ddpg.main(args)
#    return
#
#if __name__ == '__main__':
#    __spec__ = None
#    jobs = []
#    for i in range(3):
#           print('here')
#           p = multiprocessing.Process(target=multiprocessing_import_worker.worker, args=arguments[i])
#           jobs.append(p)
#           p.start()
#
#
#
#
#
