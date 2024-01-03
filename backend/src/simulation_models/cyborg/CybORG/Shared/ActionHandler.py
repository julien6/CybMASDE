# Copyright DST Group. Licensed under the MIT license.
from simulation_models.cyborg.CybORG.Simulator.Actions.Action import Action
from simulation_models.cyborg.CybORG.Shared.Observation import Observation


class ActionHandler:
    def __init__(self):
        pass

    def perform(self, action: Action) -> Observation:
        raise NotImplementedError
