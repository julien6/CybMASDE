# Copyright DST Group. Licensed under the MIT license.
from simulation_models.cyborg.CybORG.Simulator.Actions.ShellActionsFolder.ShellAction import ShellAction


class NetworkScan(ShellAction):
    def __init__(self, session, agent, subnet):
        super().__init__(session, agent)
        self.subnet = subnet

    def execute(self, state):
        pass
