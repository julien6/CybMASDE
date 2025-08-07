import itertools
from typing import Callable, Dict, List, Tuple, Union

from mma_wrapper.organizational_model import deontic_specifications, organizational_model, time_constraint_type, obligation, permission


class deontic_specifications_ttl:

    def __init__(self, osr_model: organizational_model) -> None:
        self.osr_model = osr_model
        self.ttl: Dict[str, Dict[Union[obligation, permission], Dict[str, int]]] = {
            "obligations": {}, "permissions": {}}
        if osr_model.deontic_specifications.obligations is not None:
            for obl in osr_model.deontic_specifications.obligations:
                if self.ttl["obligations"].get(obl, None) == None:
                    self.ttl["obligations"].setdefault(obl, {})
                for agent_name in osr_model.deontic_specifications.obligations[obl]:
                    if self.ttl["obligations"][obl].get(agent_name, None) == None:
                        self.ttl["obligations"][obl].setdefault(agent_name, {})
                    self.ttl["obligations"][obl][agent_name] = self.tc_to_time_unit(
                        obl.time_constraint)

        if osr_model.deontic_specifications.permissions is not None:
            for perm in osr_model.deontic_specifications.permissions:
                if self.ttl["permissions"].get(perm, None) == None:
                    self.ttl["permissions"].setdefault(perm, {})
                for agent_name in osr_model.deontic_specifications.permissions[perm]:
                    if self.ttl["permissions"][perm].get(agent_name, None) == None:
                        self.ttl["permissions"][perm].setdefault(agent_name, {})
                    self.ttl["permissions"][perm][agent_name] = self.tc_to_time_unit(
                        perm.time_constraint)

    def tc_to_time_unit(self,  time_constraint: Union[time_constraint_type, str]):
        if type(time_constraint) == time_constraint_type:
            if time_constraint == time_constraint_type.ANY:
                return -1
        elif type(time_constraint) == str:
            if time_constraint.isdigit():
                return int(time_constraint)
        raise Exception("Invalid Time Constraint value")

    def decrease(self):
        for obligation in self.ttl["obligations"]:
            for agent_name in self.ttl["obligations"][obligation]:
                ttl_value = self.ttl["obligations"][obligation][agent_name]
                if ttl_value > 0:
                    ttl_value -= 1
                    if ttl_value == 0:
                        self.osr_model.deontic_specifications.obligations[obligation].remove(
                            agent_name)
                        if len(self.osr_model.deontic_specifications.obligations[obligation]) == 0:
                            del self.osr_model.deontic_specifications.obligations[obligation]
                        if len(self.osr_model.deontic_specifications.obligations) == 0:
                            self.osr_model.deontic_specifications.obligations = {}
                    self.ttl["obligations"][obligation][agent_name] = ttl_value

        for permission in self.ttl["permissions"]:
            for agent_name in self.ttl["permissions"][permission]:
                ttl_value = self.ttl["permissions"][permission][agent_name]
                if ttl_value > 0:
                    ttl_value -= 1
                    if ttl_value == 0:
                        self.osr_model.deontic_specifications.permissions[permission].remove(
                            agent_name)
                        if len(self.osr_model.deontic_specifications.permissions[permission]) == 0:
                            del self.osr_model.deontic_specifications.permissions[permission]
                        if len(self.osr_model.deontic_specifications.permissions) == 0:
                            self.osr_model.deontic_specifications.permissions = {}
                    self.ttl["permissions"][permission][agent_name] = ttl_value


if __name__ == '__main__':

    permissions = {
        permission(
            role='role_1',
            mission='mission1',
            time_constraint=time_constraint_type.ANY
        ): ["agent_1", "agent_2"],
        permission(
            role='role_3',
            mission='mission1',
            time_constraint='3'
        ): ["agent_3"]
    }
    obligations = {
        obligation(
            role='role_2',
            mission='mission2',
            time_constraint='3'
        ): ["agent_2"]
    }
    deontic_specs = deontic_specifications(
        permissions=permissions,
        obligations=obligations
    )

    org_model = organizational_model(
        structural_specifications=None,
        functional_specifications=None,
        deontic_specifications=deontic_specs
    )

    ds_ttl = deontic_specifications_ttl(org_model)
    print(ds_ttl.osr_model)

    print("\n\n")

    ds_ttl.decrease()
    print(ds_ttl.ttl)

    print("")

    ds_ttl.decrease()
    print(ds_ttl.ttl)

    print("")

    ds_ttl.decrease()
    print(ds_ttl.ttl)

    print("\n\n")
    print(ds_ttl.osr_model)
