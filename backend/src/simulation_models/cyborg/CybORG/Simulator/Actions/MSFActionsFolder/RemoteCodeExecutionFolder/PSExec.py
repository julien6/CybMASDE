# Copyright DST Group. Licensed under the MIT license.
from ipaddress import IPv4Address

from simulation_models.cyborg.CybORG.Simulator.Actions.MSFActionsFolder.RemoteCodeExecutionFolder.RemoteCodeExecution import RemoteCodeExecution


class PSExec(RemoteCodeExecution):
    def __init__(self, session: int, target_ip_address:IPv4Address):
        super().__init__()

    def execute(self, state):
        pass
