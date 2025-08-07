class JointPolicy:

    def __init__(self):
        self.policy = {}

    def from_dict(self, data):
        self.policy = data

    def to_dict(self):
        return self.policy

    def next_action(self, observation):
        """Compute the next joint action based on the observation."""
        # This is a placeholder for the actual logic to compute the joint action
        # based on the observation and the policy.
        return [self.policy.get(agent, None) for agent in observation]
