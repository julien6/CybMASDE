# Copyright DST Group. Licensed under the MIT license.
from simulation_models.cyborg.CybORG.Simulator.Actions.ShellActionsFolder.AccountManipulationFolder.AccountManipulation import AccountManipulation
from simulation_models.cyborg.CybORG.Shared.Enums import OperatingSystemType
from simulation_models.cyborg.CybORG.Shared.Observation import Observation


class DisableUserLinux(AccountManipulation):
    def __init__(self, session, agent, username):
        super().__init__(session, agent)
        self.username = username

    def execute(self, state):
        obs = Observation()
        obs.set_success(False)
        if self.session not in state.sessions[self.agent]:
            return obs

        if state.sessions[self.agent][self.session].active:
            host = state.sessions[self.agent][self.session].hostname
            obs.add_system_info(hostid="hostid0", os_type=host.os_type)
            if host.os_type == OperatingSystemType.LINUX:
                result = host.disable_user(username=self.username)
                obs.set_success(result)
            else:
                obs.set_success(False)
        else:
            obs.set_success(False)
        return obs
