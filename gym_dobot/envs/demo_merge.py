import numpy as np
import sys
import os
import datetime
import glob


actions = []
observations = []
infos = []
count = 0

for filename in glob.glob('Demos/*.npz'):
    data = np.load(filename)
    episodeAcs = data['epacs']
    episodeObs = data['epobs']
    episodeInfo = data['epinfo']
    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    count +=1

fname = datetime.datetime.now().strftime(str(count)+"_Demos_Merged_%d%b_%H-%M-%S.npz")
dirpath = 'Demos/Merged/'
if not os.path.exists(dirpath):
    try:
        os.makedirs(dirpath)
        print("Directory Demos/Merged created.")
    except:
        print("Failed to create directory. Please create one manually.")
np.savez_compressed(dirpath+fname,acs=actions, obs=observations, info=infos)
print("Saved "+fname)
