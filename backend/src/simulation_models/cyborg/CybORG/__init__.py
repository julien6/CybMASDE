import inspect
# allows import of CybORG class as:
# from simulation_models.cyborg.CybORG import CybORG
from simulation_models.cyborg.CybORG.env import CybORG

path = str(inspect.getfile(CybORG))
path = path[:-7] + '/version.txt'
with open(path) as f:
    CYBORG_VERSION = f.read()[:-1]
