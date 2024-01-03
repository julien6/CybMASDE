# Copyright DST Group. Licensed under the MIT license.
from simulation_models.cyborg.CybORG.Simulator.Actions.ShellActionsFolder.ShellAction import ShellAction
from simulation_models.cyborg.CybORG.Simulator.State import State


class OpenConnection(ShellAction):
    def __init__(self, session, agent):
        super().__init__(session, agent)

    def execute(self, state: State):
        pass
