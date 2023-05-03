import numpy as np
import gym

env = gym.make('ALE/Boxing-v5')
env.observation_space = gym.spaces.flatten_space(env.observation_space)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
qtable = np.zeros((state_space, action_space))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001
episodes = 10000
max_steps = 100

for episode in range(episodes):
    print("Episode: {}".format(episode))
    state = env.reset()
    state = gym.spaces.flatten(env.observation_space, state[0])
    done = False
    step = 0
    for step in range(max_steps):
        exp_exp_tradeoff = np.random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info, _ = env.step(action)
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state
        if done:
            break
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

env.close()

env = gym.make('ALE/Boxing-v5', render_mode='human')
env.observation_space = gym.spaces.flatten_space(env.observation_space)
state = env.reset()
done = False
while not done:
    state = gym.spaces.flatten(env.observation_space, state[0])
    action = np.argmax(qtable[state, :])
    new_state, reward, done, info, _ = env.step(action)
    state = new_state

print("Reward: {}".format(reward))
env.close()
