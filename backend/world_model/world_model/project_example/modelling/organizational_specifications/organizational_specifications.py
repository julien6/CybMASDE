"""
MOISE+MARL — Organizational Model Template (focus on `organizational_model`)
----------------------------------------------------------------------------
This single-file skeleton focuses ONLY on building the `organizational_model`:
- Structural specifications (roles, inheritance, groups)
- Functional specifications (goals, missions / social scheme, preferences)
- Deontic specifications (permissions, obligations, prohibitions if supported)
- Role-attached script rules (functions) with clear placeholders

No `label_manager` is defined here. We assume your environment / wrapper provides
whatever is needed to decode observations and encode actions. Role functions thus
keep a generic signature and may return low-level actions directly (e.g., ints)
or a high-level token that your runtime will map to concrete actions.

How to use
==========
1) Copy this file into your project, e.g. `organizational_model_template.py`.
2) Fill the TODOs with your roles, goals, missions, and constraints.
3) Import `org_model` from this file and pass it to your env factory, e.g.:

   env = marl.make_env(
       environment_name="overcooked",
       map_name="asymmetric_advantages",
       organizational_model=org_model,
   )

API note
========
This template uses the MMA (MOISE+MARL API) classes:
- structural_specifications
- functional_specifications
- deontic_specifications / deontic_specification
- organizational_model
- role_logic, goal_logic, role_factory, goal_factory
- time_constraint_type

Adapt names if your local API differs.
"""

from typing import Any, Dict, List, Optional

# Core MMA (MOISE+MARL API)
from mma_wrapper.organizational_model import (
    organizational_model,
    structural_specifications,
    functional_specifications,
    deontic_specifications,
    deontic_specification,
    time_constraint_type,
)
from mma_wrapper.organizational_specification_logic import (
    role_logic,
    goal_logic,
    goal_factory,
    role_factory,
)

# =====================================================================
# 1) Role-attached functions (script rules) — keep generic & env-agnostic
# =====================================================================
# Signature is intentionally simple. If your runtime injects a label manager or
# helper, it can be passed via **kwargs. Return the form your runtime expects
# (e.g., an int action id, or a symbolic action like "noop").


def role_primary_fun(traj: Any, obs: Any, agent_name: str, **kwargs) -> Any:
    """Primary role script example.
    - traj: opaque trajectory/history object (use if you track patterns)
    - obs: environment observation in whatever format your env provides
    - agent_name: current agent id
    - **kwargs: optional helpers (e.g., lm, utils, config)

    TODO: implement simple, readable heuristics referencing your obs.
    """
    # EXAMPLE (symbolic action): return "noop"
    # EXAMPLE (low-level discrete action id): return 0
    return "noop"


def role_secondary_fun(traj: Any, obs: Any, agent_name: str, **kwargs) -> Any:
    """Secondary role script example."""
    return "noop"

# You can define as many role functions as needed:
# def role_scout_fun(...): ...
# def role_defender_fun(...): ...
# def role_cleaner_fun(...): ...


# =========================================================
# 2) Functional layer — optional goals and goal-logic hooks
# =========================================================
# If you want the organizational layer to expose named goals with satisfaction
# predicates, define them here. These can then be referenced in missions.

# Example goal predicate (commented):
# def goal_is_ready(obs: Dict[str, Any]) -> bool:
#     return bool(obs.get("is_ready", 0))

# gf = goal_factory()
# G1 = gf.make_goal(
#     goal_id="G1",
#     description="Example goal that becomes true when obs['is_ready']==1",
#     is_satisfied=goal_logic(lambda o: bool(o.get("is_ready", 0)))
# )


# ============================================================
# 3) Structural specs — roles, inheritance, and (optional) groups
# ============================================================
# Define your roles and attach script rules. Use role_logic().registrer_script_rule
# (or the equivalent in your API). If your API supports role inheritance or groups,
# declare them here as well.

roles = {
    "role_primary": role_logic().registrer_script_rule(role_primary_fun),
    "role_secondary": role_logic().registrer_script_rule(role_secondary_fun),
    # Add more roles and attach their rule functions here...
}

role_inheritance = {
    # Optional: child inherits parent's rules & permissions
    # "role_specialized": "role_primary",
}

root_groups = {
    # Optional: MOISE+ groups archetypes with cardinalities and role sets
    # "team_A": {"min": 2, "max": 3, "roles": ["role_primary", "role_secondary"]}
}

struct = structural_specifications(
    roles=roles,
    role_inheritance_relations=role_inheritance,
    root_groups=root_groups,
)


# =================================================================
# 4) Functional specs — missions/social scheme and preferences order
# =================================================================
# Reference your goals (if any) and define missions that specify which roles
# are expected to contribute to which goals / tasks. Many APIs support expressing
# AND/OR decompositions or precedence; keep it simple here.

func = functional_specifications(
    goals={
        # "G1": G1,
    },
    social_scheme={
        # Example mission skeletons:
        # "mission_collect": {
        #     "requires": ["G1"],     # the set of goals to reach
        #     "roles": ["role_primary", "role_secondary"],
        # },
        # "mission_guard": {
        #     "requires": [],
        #     "roles": ["role_secondary"],
        # },
    },
    mission_preferences=[
        # Optional: order/prioritize missions, e.g. (mission_name, weight)
        # ("mission_collect", 1.0),
    ],
)


# ==========================================================
# 5) Deontic specs — permissions / obligations / prohibitions
# ==========================================================
# Bind roles to concrete agents and optionally impose timing constraints.
# time_constraint_type examples (depending on your API): ANY, ALWAYS, INITIALLY, FINALLY.

agents_primary: List[str] = ["agent_0"]
agents_secondary: List[str] = ["agent_1"]

permissions = [
    # Agents allowed to enact certain roles
    deontic_specification("role_primary", agents_primary,
                          [], time_constraint_type.ANY),
    deontic_specification("role_secondary", agents_secondary,
                          [], time_constraint_type.ANY),
]

obligations = [
    # Hard requirements (use carefully). Example commented out:
    # deontic_specification("role_primary", agents_primary, [], time_constraint_type.ALWAYS),
]

# Some APIs also allow prohibitions; if not, simply omit permissions to implicitly forbid.
# prohibitions = [ ... ]

deon = deontic_specifications(
    permissions=permissions,
    obligations=obligations,
    # prohibitions=prohibitions,  # uncomment if supported
)


# ============================================================
# 6) Build the organizational model (the object you will attach)
# ============================================================
org_model = organizational_model(
    structural_specifications=struct,
    functional_specifications=func,
    deontic_specifications=deon,
)


# ==================================================================================
# 7) (Optional) Advanced hooks — patterns/shielding & TRF (trajectory reward bonus)
# ==================================================================================
# If you use historical constraints (OAC) or shielding, wire them in your runtime
# where role functions are called. Example wrappers are sketched below for reference.

# Example shielding wrapper (pseudo):
# def with_shield(rule_fun):
#     def _wrapped(traj, obs, agent_name, **kwargs):
#         a = rule_fun(traj, obs, agent_name, **kwargs)
#         # allowed = get_allowed_actions(history=traj, observation=obs, role=current_role)
#         # if a not in allowed: a = choose_safe_fallback(allowed)
#         return a
#     return _wrapped
#
# roles["role_primary"] = role_logic().registrer_script_rule(with_shield(role_primary_fun))

# Example TRF hook (reward shaping from trajectory similarity): integrate in your
# env wrapper or callbacks during training.
# def trajectory_reward_bonus(history) -> float:
#     return 0.0
