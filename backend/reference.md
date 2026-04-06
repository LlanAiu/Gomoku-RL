# The RL Structure

If you poke around the `rl` module, you'll find abstract classes for most the RL machinery, defining what functions are needed for it to work properly.

A quick overview of where everything is:
```
modules/rl/
| - agent/
  | - policies/ (types of policies)
  | - ... (agent + types of value functions)
| - elements/
  | - (state, action, and rewards)
| - environment/
  | - environment.py (the RL environment -- episodic only)
| - optimization/
  | - action_value/ (action value methods)
  | - policy_gradient/ (policy gradient methods)
  | - optimization_method.py (overall optimization method base class)
| - train/
  | - trainer.py (framework to actually run training episodes)
```

This same setup is mirrored in `modules/game` except all concrete implementations pertaining to Gomoku.

## Notes and References

Everything else in this document is just  hints, for if you need some more guidance on where to start. Also note that if you have the repository, you can check how I implemented everything in the `reference` branch.

### Environment
- An action will change the board, which counts as a next state
    - Actually change `new_board` to reflect this
- Create a new state using the updated value of `new_board`

### Reward Signal
- `old_state` is actually unused in this function, we only care about `new_state`, and `action` is only used to get the player index to check win vs. loss
- There's only four distinct states we care about for reward: WIN, LOSS, DRAW (technically impossible, but still), ONGOING
- The rest is up to you -- how much is a win worth? A loss?
    - Remember scale is relative, so if WIN_REWARD = 5, LOSS_REWARD = -1, then you're telling the agent that one win is "equal" to five losses and it'll learn to play that way
- You can also penalize the agent for long episodes by assigning a negative reward for ONGOING states, but think about whether you want this behavior in this particular environment

### State
- You have a lot of free reign here, but it basically revolves around representing the board in a way that the model can learn how to play the game from
- Don't consider player 1 vs player 2, consider it as the player vs the opponent (so you can reuse this function, and thus the entire agent, to play against itself)
- Return some sort of numpy array (or collection of numpy arrays) because of computation speed / everything else uses it

### Value Function
- Linearly map the feature vector to a scalar value using the weights
- If necessary (it was for my implementation), preprocess the vector representation you get from the state
  - Mainly to make the feature per-player because player_index isn't stored at a state level
- Remember to change FEATURE_IN_DIM to match the size of your state feature

### Parametrized Policy
- Linearly map the feature vector to a vector of action preferences, these will then be masked (to disallow invalid actions), and passed into the softmax
- Similar preprocessing needed as that of the value function
- The eligibility vector is not super easy to calculate, because you need the softmax jacobian via chain rule, I'll just tell you that the proper output is: embedding * (action - action_probabilities) ^ T.
    - Embedding: the state feature
    - Action: the selected action, represented as a one-hot vector
    - Action probabilities: the probabilities of selecting each action by the policy
    - Outer product at the end
- Feel free to just steal the reference code for this one, it's a clunky build with not that much importance to the general concept

### Action Value Function
- Similar to value function, just make sure you're taking the selected action into account as well
    - This is still a linear mapping (everything here is)
- Again, don't forget to update the FEATURE_INT_DIM

### Epsilon Greedy Policy
- Choose action flow should be:
    - Pick random number
    - If less than epsilon, do something random
    - Else, pick the best action according to action value function
- Choose action inference should always be greedy (i.e. pick best action)
- after_step simply multiplies epsilon by decay rate, but make sure epsilon is not less than `epsilon_min`

### Trainer
- Previous reward stores the state, action, and reward for the other player
- The effective transition goes from that stored state, to the state after the opponent plays 
    - This is our source of non-determinism, because otherwise the game dynamics are completely known and this model learns nothing
    - Thus, this is the state-action pair we use for the TD delta
- Be careful that terminal updates must be handled immediately (because no next state)