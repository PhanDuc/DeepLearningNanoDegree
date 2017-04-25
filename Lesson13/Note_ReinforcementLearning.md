#  Reinforcement Learning

## Resources

- [Series Blog of Deep Reinforcement Learning](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
- [Nature Paper on Human-level control through deep reinforcement
learning](http://www.davidqiu.com:8888/research/nature14236.pdf) and its [tensorflow implementation](https://github.com/devsisters/DQN-tensorflow)
- [Cart-Pole Balancing](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)

## Q-Learning

### States

> One Particular arrangement of all the objects in the environment

### Actions

> An agent's change of state

### Q-table

**Representation:** 2D state-action probabilistic transition matrix: 

- row: state
- col: action

**Update:** Bellman Equation

$$Q(s,a) = r(s) + \gamma(max(Q(s',a')))$$

- Initialzation: $Q(s,a) = 0$ for all $s, a$
- $Q'(s,a) = r(s) + \gamma(max(Q(s',a')))$
- $\Delta Q(s,a) = Q'(s,a) - Q(s,a)$
- $Q(s,a) = Q(s,a) + \eta \Delta Q$

### Neural Network and Q-table

States & Action -> Neural Network (Use Neural Network to approximate Q-table)

## Deep Q-Learning

details see the code