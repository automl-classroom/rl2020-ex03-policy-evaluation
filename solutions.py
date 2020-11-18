import numpy as np
from env import MarsRover

def dynamic_programming_step(v, pi, transition_probabilities, rewards, gamma=0.9):
    new_v = np.copy(v).astype(float)
    for s in range(5):
        action = pi[s]
        next_state = min(4, max(0, s + action + (action-1)))
        alternate_state = min(4, max(0, s - action + np.abs(action-1)))
        new_v[s] = rewards[int(next_state)] + gamma * (transition_probabilities[s][int(action)]*v[int(next_state)]+(1-transition_probabilities[s][int(action)])*v[int(alternate_state)])
    return new_v

def monte_carlo_step(v, first_visits, total_returns, sample, gamma=0.9):
    v_new = np.copy(v)
    updated_visits = np.copy(first_visits)
    updated_returns = np.copy(total_returns)
    visited_this_episode = np.zeros(5)
    for i in range(len(sample)):
        if i%3 == 0 and not visited_this_episode[sample[i]]:
            updated_visits[sample[i]] += 1
            future_rewards = [sample[j] for j in range(i+2, len(sample), 3)]
            acc_future_rewards = 0
            for k in range(len(future_rewards)):
                acc_future_rewards += (gamma ** k) * future_rewards[k]
            updated_returns[sample[i]] += acc_future_rewards
            v_new[sample[i]] = updated_returns[sample[i]] / updated_visits[sample[i]]
            visited_this_episode[sample[i]] = 1
    return v_new, updated_visits, updated_returns

def td_zero_step(v, sample, alpha=0.1, gamma=0.9):
    v_new = np.copy(v)
    state = sample[0]
    reward = sample[2]
    next_state = sample[3]

    v_new[state] = v[state] + alpha * (reward + gamma * v[next_state] - v[state])
    return v_new

def sample_transitions(pi, transition_probabilities, rewards):
    env = MarsRover(transition_probabilities=transition_probabilities, rewards=rewards)

    mars_rover_samples = []
    for _ in range(500):
        state = env.reset()
        done = False
        while not done:
            action = pi[state]
            next_state, reward, done = env.step(action)
            mars_rover_samples.append([state, action, reward, next_state])
            state = next_state
    np.random.shuffle(mars_rover_samples)
    return mars_rover_samples

def sample_episodes(pi, transition_probabilities, rewards):
    env = MarsRover(transition_probabilities=transition_probabilities, rewards=rewards)

    mars_rover_episodes = []
    for _ in range(500):
        state = env.reset()
        done = False
        ep = [state]
        while not done:
            action = pi[state]
            next_state, reward, done = env.step(action)
            ep.append(action)
            ep.append(reward)
            ep.append(next_state)
            state = next_state
        mars_rover_episodes.append(ep)
    np.random.shuffle(mars_rover_episodes)
    return mars_rover_episodes

def evaluate_policy_dp(pi=np.random.randint(2, size=5), transition_probabilities=np.ones((5,2)), rewards=[1, 0, 0, 0, 10]):
    env = MarsRover(transition_probabilities=transition_probabilities, rewards=rewards)
    v = np.ones(5) * np.inf
    v_new = np.zeros(5)
    i = 0
    while not np.array_equal(v, v_new):
        i += 1
        v = np.copy(v_new)
        v_new = dynamic_programming_step(v, pi, transition_probabilities, rewards)
    print(f"Policy was evaluated in {i} steps with resulting v {v}")
    return v, i

def evaluate_policy_mc(pi=np.random.randint(2, size=5), transition_probabilities=np.ones((5,2)), rewards=[1, 0, 0, 0, 10], max_steps=10):
    env = MarsRover(transition_probabilities=transition_probabilities, rewards=rewards)
    v = np.ones(5) * np.inf
    v_new = np.zeros(5)
    first_visits = np.zeros(5)
    total_returns = np.zeros(5)
    i = 0

    sampled_episodes = sample_episodes(pi, transition_probabilities, rewards)

    while not np.array_equal(v, v_new) and i < max_steps:
        i += 1
        v = np.copy(v_new)
        random_index = np.random.randint(len(sampled_episodes))
        sample = sampled_episodes[random_index]
        v_new, first_visits, total_returns = monte_carlo_step(v, first_visits, total_returns, sample, gamma=0.9)
    print(f"Policy was evaluated in {i} steps with resulting v {v}")
    return v, i

def evaluate_policy_td_zero(pi=np.random.randint(2, size=5), transition_probabilities=np.ones((5,2)), rewards=[1, 0, 0, 0, 10], max_steps=100):
    env = MarsRover(transition_probabilities=transition_probabilities, rewards=rewards)
    v = np.ones(5) * np.inf
    v_new = np.zeros(5)
    i = 0

    sampled_transitions = sample_transitions(pi, transition_probabilities, rewards)

    while not np.array_equal(v, v_new) and i < max_steps:
        i += 1
        v = np.copy(v_new)
        random_index = np.random.randint(len(sampled_transitions))
        sample = sampled_transitions[random_index]
        v_new = td_zero_step(v, sample)
    print(f"Policy was evaluated in {i} steps with resulting v {v}")
    return v, i
