import json
import dataclasses

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from mma_wrapper.utils import cardinality
from mma_wrapper.organizational_specification_logic import role_logic, goal_logic
from mma_wrapper.label_manager import label_manager


class os_encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, sub_plan):
            return o.to_dict()
        if isinstance(o, role_logic):
            return o.to_dict()
        if isinstance(o, goal_logic):
            return o.to_dict()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class organizational_specification:
    """The basic class
    """
    pass


class role(str, organizational_specification):
    pass


class group_tag(str):
    pass


class link_type(str, Enum):
    ACQ = 'ACQ'
    COM = 'COM'
    AUT = 'AUT'


@dataclass
class link(organizational_specification):
    source: role
    destination: role
    type: link_type

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'link':
        if d is None:
            d = {}
        return link(
            source=d.get('source', None),
            destination=d.get('destination', None),
            type=link_type(d.get('type', None))
        )


@dataclass
class compatibility(organizational_specification):
    source: role
    destination: role

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'compatibility':
        if d is None:
            d = {}
        return compatibility(
            source=d.get('source', None),
            destination=d.get('destination', None)
        )


@dataclass
class group_specifications(organizational_specification):
    roles: List[role]
    sub_groups: Dict[group_tag, 'group_specifications']
    intra_links: List[link]
    inter_links: List[link]
    intra_compatibilities: List[compatibility]
    inter_compatibilities: List[compatibility]
    # by default: cardinality(0, INFINITE)
    role_cardinalities: Dict[role, cardinality]
    # by default: cardinality(0, INFINITE)
    sub_group_cardinalities: Dict[group_tag, cardinality]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'group_specifications':
        if d is None:
            d = {}
        return group_specifications(
            roles=[role(r) for r in d.get('roles', [])],
            sub_groups={group_tag(k): group_specifications.from_dict(v)
                        for k, v in d.get('sub_groups', {}).items()},
            intra_links=[link.from_dict(l) for l in d.get('intra_links', [])],
            inter_links=[link.from_dict(l) for l in d.get('inter_links', [])],
            intra_compatibilities=[compatibility.from_dict(
                c) for c in d.get('intra_compatibilities', [])],
            inter_compatibilities=[compatibility.from_dict(
                c) for c in d.get('inter_compatibilities', [])],
            role_cardinalities={role(k): cardinality.from_dict(v)
                                for k, v in d.get('role_cardinalities', {}).items()},
            sub_group_cardinalities={group_tag(k): cardinality.from_dict(
                v) for k, v in d.get('sub_group_cardinalities', {}).items()}
        )


@dataclass
class structural_specifications(organizational_specification):
    roles: Dict[role, role_logic]
    role_inheritance_relations: Dict[role, List[role]]
    root_groups: Dict[group_tag, group_specifications]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'structural_specifications':
        if d is None:
            d = {}
        return structural_specifications(
            roles={role(r): role_logic.from_dict(d)
                   for r, d in d.get('roles', {}).items()},
            role_inheritance_relations={role(
                k): [role(r) for r in v] for k, v in d.get('role_inheritance_relations', {}).items()},
            root_groups={group_tag(k): group_specifications.from_dict(v)
                         for k, v in d.get('root_groups', {}).items()}
        )


class goal(str, organizational_specification):
    pass


class mission(str, organizational_specification):
    pass


class plan_operator(str, Enum):
    SEQUENCE = 'SEQUENCE'
    CHOICE = 'CHOICE'
    PARALLEL = 'PARALLEL'


class sub_plan(organizational_specification):
    operator: plan_operator
    sub_goals: List['plan']

    def __init__(self, operator: plan_operator, sub_goals: List[Union['goal', 'plan']]):
        self.operator = operator
        self.sub_goals = [plan(sub_goal) if type(
            sub_goal) == str else sub_goal for sub_goal in sub_goals]

    def to_dict(self) -> Dict:
        return {
            'operator': self.operator,
            'sub_goals': [sub_goal.goal if (sub_goal.sub_plan is None and sub_goal.probability == 1.0) else sub_goal for sub_goal in self.sub_goals]
        }

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__())

    def __eq__(self, other):
        return self.operator == other.operator and self.sub_goals == other.sub_goals


@dataclass
class plan(organizational_specification):
    goal: goal
    sub_plan: 'sub_plan' = None
    probability: float = 1.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'plan':
        if d is None:
            return {}
        if type(d) == str:
            return plan(goal=d)
        else:
            return plan(
                goal=d.get('goal', {}),
                sub_plan=sub_plan(operator=None if d.get('sub_plan', {}).get('operator', None) is None else plan_operator(d['sub_plan']['operator']),
                                  sub_goals=[plan.from_dict(sub_goal) for sub_goal in d.get('sub_plan', {}).get('sub_goals', [])]),
                probability=d.get('probability', 1.0))


@dataclass
class social_scheme(organizational_specification):
    goals_structure: plan
    mission_to_goals: Dict[mission, List[goal]]
    # by default: cardinality(1, INFINITE)
    mission_to_agent_cardinality: Dict[mission, cardinality]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'social_scheme':
        if d is None:
            d = {}
        return social_scheme(
            goals_structure={} if d.get(
                'goals_structure', None) is None else plan.from_dict(d['goals_structure']),
            mission_to_goals={k: [goal(g) for g in v]
                              for k, v in d.get('mission_to_goals', {}).items()},
            mission_to_agent_cardinality={k: cardinality.from_dict(
                v) for k, v in d.get('mission_to_agent_cardinality', {}).items()}
        )


class social_scheme_tag(str):
    pass


@dataclass
class mission_preference(organizational_specification):
    prefer: mission
    over: mission

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'mission_preference':
        if d is None:
            return {}
        return mission_preference(
            prefer=d.get('prefer', None),
            over=d.get('over', None)
        )


@dataclass
class functional_specifications(organizational_specification):
    goals: Dict[goal, goal_logic]
    social_scheme: Dict[social_scheme_tag, social_scheme]
    mission_preferences: List[mission_preference]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'functional_specifications':
        if d is None:
            d = {}
        return functional_specifications(
            goals={goal: goal_logic.from_dict(goal_lgc) if goal_lgc is not None else goal_logic()
                   for goal, goal_lgc in d.get('goals', {}).items()},
            social_scheme={k: social_scheme.from_dict(v)
                           for k, v in d.get('social_scheme', {}).items()},
            mission_preferences=[mission_preference.from_dict(
                p) for p in d.get('mission_preferences', [])]
        )


class time_constraint_type(str, Enum):
    ANY = 'ANY'


@dataclass
class deontic_specification(organizational_specification):
    role: role
    agents: List[str]
    missions: List[mission]
    time_constraint: time_constraint_type = time_constraint_type.ANY

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'deontic_specification':
        if d is None:
            d = {}
        return deontic_specification(
            role=d.get('role', None),
            agents=d.get('agents', []),
            missions=d.get('missions', []),
            time_constraint=d.get('time_constraint', time_constraint_type.ANY)
        )


@dataclass
class deontic_specifications(organizational_specification):
    permissions: List[deontic_specification]
    obligations: List[deontic_specification]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'deontic_specifications':
        if d is None:
            d = {}
        return deontic_specifications(
            permissions=[deontic_specification.from_dict(
                permission) for permission in d.get('permissions', [])],
            obligations=[deontic_specification.from_dict(
                obligation) for obligation in d.get('obligations', [])]
        )


@dataclass
class organizational_model:
    structural_specifications: 'structural_specifications'
    functional_specifications: 'functional_specifications'
    deontic_specifications: 'deontic_specifications'

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'organizational_model':
        return organizational_model(
            structural_specifications=structural_specifications.from_dict(
                d.get('structural_specifications', None)),
            functional_specifications=functional_specifications.from_dict(
                d.get('functional_specifications', None)),
            deontic_specifications=deontic_specifications.from_dict(
                d.get('deontic_specifications', None))
        )

    def to_dict(self) -> Dict:
        return json.loads(json.dumps(self, indent=4, cls=os_encoder))
