# Copyright DST Group. Licensed under the MIT license.
from simulation_models.cyborg.CybORG.Simulator.Actions.ShellActionsFolder.ServiceManipulationFolder.ServiceManipulation import ServiceManipulation


class StartService(ServiceManipulation):
    def __init__(self, session, agent):
        super().__init__(session, agent)

    def execute(self, state):
        pass