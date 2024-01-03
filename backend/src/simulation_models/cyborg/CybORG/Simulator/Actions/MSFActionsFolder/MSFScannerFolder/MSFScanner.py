# Copyright DST Group. Licensed under the MIT license.
from simulation_models.cyborg.CybORG.Simulator.Actions.MSFActionsFolder.MSFAction import MSFAction
from simulation_models.cyborg.CybORG.Simulator.State import State


class MSFScanner(MSFAction):
    def __init__(self, session, agent):
        super().__init__(session, agent)

    def execute(self, state: State):
        pass
