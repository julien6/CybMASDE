from mma_wrapper.label_manager import label_manager


class ComponentFunctions:
    def __init__(self, label_manager: "label_manager" = None):
        """
        Args:
            label_manager: Optional instance to decode/encode observations and actions.
        """
        self.label_manager = label_manager

    def reward_fn(self, current_obs, action, next_obs):
        """
        Compute the reward from observations/actions (possibly one-hot encoded).
        Args:
            current_obs: dict {agent_id: one-hot obs}
            action: dict {agent_id: one-hot action}
            next_obs: dict {agent_id: one-hot obs}
        Returns:
            float or dict {agent_id: float}
        """
        # Example: use label_manager to decode if provided
        # decoded_obs = self.label_manager.one_hot_decode_observation(current_obs[agent_id])
        raise NotImplementedError

    def done_fn(self, current_obs, action, next_obs):
        """
        Determine episode termination.
        Args are the same as reward_fn.
        Returns:
            bool or dict {agent_id: bool}
        """
        raise NotImplementedError

    def render_fn(self, current_obs, action, next_obs):
        """
        Custom rendering function.
        Args are the same as reward_fn.
        Returns:
            Any displayable object or None
        """
        raise NotImplementedError
