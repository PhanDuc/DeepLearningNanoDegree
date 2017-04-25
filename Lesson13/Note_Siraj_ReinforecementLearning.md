#  Siraj's Reinforcement Learning

[Code](https://github.com/llSourcell/how_to_win_slot_machines) for reference

## Resources

- [Karpathy on Deep RL](http://karpathy.github.io/2016/05/31/rl/)
- [RL with policy gradient](http://minpy.readthedocs.io/en/latest/tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.html)
- [Multi-arm bandit](https://dataorigami.net/blogs/napkin-folding/79031811-multi-armed-bandits)
- [Live for Policy Gradient to beat pong](https://www.youtube.com/watch?v=PDbXPBwOavc)
- [Cartpole algo](http://kvfrans.com/simple-algoritms-for-solving-cartpole/)

## RL

Explore/Learn v.s. Exploit

obsevation - (reinforcement) -> policy modification -> action ->observation

### Policy Gradient

epsilon greedy exploration policy. 

$$Loss = Log(P) * A$$

where $A(advantage) = Reward - Baseline$, where $Baseline$ is flexible to choose, maybe 0.