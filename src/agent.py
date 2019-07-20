from rl_policy_network import RLPolicyNetwork


class Agent:
    def __init__(self,
                 batch_size,
                 xp,
                 player: RLPolicyNetwork,
                 optimizer
                 ) -> None:
        self.batch_size = batch_size
        self.xp = xp
        self.player = player
        self.optimizer = optimizer
        self.policies = []

    def act(self,
            obs,
            ):
        action_distribution = self.player(obs)
        action_indices = action_distribution.sample().array
        policies = action_distribution.log_prob(action_indices)
        self.policies.append(policies)
        return action_indices

    def update(self,
               true_rewards
               ):
        loss = 0
        for r, p in zip(true_rewards, self.policies):
            loss -= sum(r * p)
        loss /= float(self.batch_size)
        self.player.cleargrads()
        loss.backward()
        self.optimizer.update()
