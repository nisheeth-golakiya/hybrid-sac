Although I have implemented the algorithm to the best of my knowledge, the correctness of the implementation remains to be checked. Any suggestions are welcome!

# hybrid-sac
[cleanRL](https://github.com/vwxyzjn/cleanrl)-style single-file pytorch implementation of hybrid-SAC algorithm from the paper [Discrete and Continuous Action Representation for Practical RL in Video Games](https://arxiv.org/pdf/1912.11077.pdf)

## Dependencies
* Requirements for training are same as cleanRL. So `pip install cleanrl` will do.
* Environments: [platform](https://github.com/cycraig/gym-platform), [goal](https://github.com/cycraig/gym-goal), [soccer](https://github.com/cycraig/gym-soccer)

Inspired by cleanRL, this repo gives the single-file implementation of the algorithm. Hence, some funcitonalities (like multi-actor execution) are not possible. Although the training is not as efficient as it could be, it is helpful for understanding the algorithm.

The paper experiments the with the following three environments:

## Platform

## Goal

## Soccer
