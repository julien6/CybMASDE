import random

from mma_wrapper.label_manager import label_manager
from mma_wrapper.utils import label, trajectory

# ============ organizational_specs_script.py =============


def function_for_role_0(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:

    # Inferred rules for role_0
    rules = {
        "[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0, -1.0, 1.0, 3.0]": {
            "0": {
                "agents_owner": [
                    [
                        "agent_1",
                        62
                    ],
                    [
                        "agent_0",
                        38
                    ]
                ],
                "weight": 0.05
            }
        },
        "[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 1.0, 2.0]": {
            "0": {
                "agents_owner": [
                    [
                        "agent_1",
                        62
                    ],
                    [
                        "agent_0",
                        38
                    ]
                ],
                "weight": 0.05
            }
        }
    }

    # Sample actions based on their weights
    actions = sorted([(act, weight) for act, weight in rules.get(str(observation.tolist()), {
                     "0": {"weight": 1}}).items()], key=lambda x: x[1]["weight"], reverse=True)
    actions = [act for act, weight in rules.get(str(observation.tolist()), {
                                                "0": {"weight": 1}}).items() if random.random() < weight]
    return actions[0] if actions else None


def function_for_role_1(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    return 0


role_to_function = {
    "role_0": function_for_role_0,
    "role_1": function_for_role_1
}


def function_for_goal_0(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:

    # Inferred goals for goal_0
    observations = {
        "[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0, -1.0, 1.0, 3.0]": {
            "agents_owner": [
                [
                    "agent_1",
                    62
                ],
                [
                    "agent_0",
                    38
                ]
            ],
            "weight": 0.05
        },
        "[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 1.0, 2.0]": {
            "agents_owner": [
                [
                    "agent_1",
                    62
                ],
                [
                    "agent_0",
                    38
                ]
            ],
            "weight": 0.05
        },
    }

    # Sample actions based on their weights
    return observations.get(str(observation.tolist()), 0)


def function_for_goal_1(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    return 0


goal_to_function = {
    "goal_0": function_for_goal_0,
    "goal_1": function_for_goal_1
}
