from simulation_models.cyborg.CybORG.Shared import Observation
from simulation_models.cyborg.CybORG.Simulator.Actions import Action
from simulation_models.cyborg.CybORG.Simulator.Session import VelociraptorServer
from simulation_models.cyborg.CybORG.Simulator.State import State


class Monitor(Action):
    def __init__(self, session: int, agent: str):
        super().__init__()
        self.agent = agent
        self.session = session

    def execute(self, state: State) -> Observation:
        obs = Observation(True)
        session: VelociraptorServer = state.sessions[self.agent][self.session]
        for child in session.children.values():
            for artifact in session.artifacts:
                if len(state.hosts[child.hostname].events[artifact]) > 0:
                    obs.add_system_info(hostid=child.hostname, **state.hosts[child.hostname].get_state())
                for event in state.hosts[child.hostname].events[artifact]:
                    if 'pid' in event:
                        session.add_sus_pids(hostname=child.hostname, pid=event['pid'])
                    obs.add_process(hostid=child.hostname, **event)
                state.hosts[child.hostname].events[artifact] = []
        return obs


    def __str__(self):
        return f"{self.__class__.__name__}"
