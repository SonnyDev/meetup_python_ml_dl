from pearl.pearl_agent import PearlAgent
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers import (
    BasicReplayBuffer,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
import torch
import time

env = GymEnvironment("CartPole-v1", render_mode="human")

num_actions = env.action_space.n
agent = PearlAgent(
    policy_learner=DeepQLearning(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        hidden_dims=[64, 64],
        training_rounds=20,
        action_representation_module=OneHotActionTensorRepresentationModule(
            max_number_actions=num_actions
        ),
        network_kwargs={
            "output_activation": None
        },
        output_transform=lambda x: x.squeeze(-1)
    ),
    replay_buffer=BasicReplayBuffer(10_000),
)

num_episodes = 500
for episode in range(num_episodes):
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(exploit=True)
        action_result = env.step(action)
        agent.observe(action_result)
        agent.learn()
        
        total_reward += action_result.reward
        done = action_result.done
        
        time.sleep(0.01)
    
    print(f"Épisode {episode + 1}, Récompense totale: {total_reward}")

env.close()