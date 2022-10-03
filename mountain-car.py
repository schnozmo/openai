
import gym, numpy as np, pandas as pd

class TabQLearner():
    def __init__(self, env, default_q = -1, eps = 0.99, decay = 0.9975, alpha = 0.3, gamma = 0.3, lval = 0.9):
        self.n_actions = env.action_space.n
        self.s1_space = 20
        self.s2_space = 12
        self.q = np.zeros((self.s1_space, self.s2_space, self.n_actions)) + default_q
        #The position is clipped to the range `[-1.2, 0.6]` and velocity is clipped to the range `[-0.07, 0.07]`.
        self.eps = eps
        self.decay = decay
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.lval = lval
        self.e_trace = np.zeros((self.s1_space, self.s2_space, self.n_actions))

    def selectAction(self, state, greedy=False):
        if greedy == False and np.random.random() < self.eps:
            return np.random.randint(self.n_actions)
        else:
            return self.q[state].argmax()
    
    def actionProbs(self, state):
        t = self.q[state].copy()
        if t.min() < 0.0001:
            t += t.min() + 0.0001
        return t/t.sum()

    def update(self, state, action, state_prime, r):
        td_new_tgt = r + self.gamma * self.q[state_prime[0], state_prime[1]].max()
        td_diff = td_new_tgt - self.q[state[0], state[1], action]
        self.e_trace[state[0], state[1], action] += 1

        #print(f"UPDATE: {state}, {action}, {state_prime}, {r} --> {td_new_tgt}, {td_diff}")

        #visit all S, A
        for s1 in range(self.q.shape[0]):
            for s2 in range(self.q.shape[1]):
                for a in range(self.q.shape[2]):
                    self.q[s1, s2, a] = self.q[s1, s2, a] + (self.alpha * td_diff * self.e_trace[s1, s2, a])
                    self.e_trace[s1, s2, a] = self.gamma * self.lval * self.e_trace[s1, s2, a]

    def epsDecay(self):
        self.eps *= self.decay
        print(f"Decayed epsilon to {self.eps}")

    def qVals(self):
        return self.q
    
    def printQTable(self):
        for i in range(len(self.q)):
            print(f"X = {i}")
            print(pd.DataFrame(self.q[i]))

    def resetEligibilityTrace(self):
        self.e_trace = np.zeros((self.s1_space, self.s2_space, self.n_actions))
               
    def eligibilityTrace(self):
        return self.e_trace


def obs_to_state(observation):
    s1 = int(((observation[0] + 1.2) / 1.8 ) * 20)
    s2 = int(((observation[1] + 0.07)/ 0.14) * 12)
    return min(s1, 19), min(s2, 11)

env = gym.make('MountainCar-v0')

tdq = TabQLearner(env = env)

# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)

t = 0
success = 0

episode = 0
consec_actions = (4,4)

while episode < 5000:
    episode += 1
    observation = env.reset()
    tdq.resetEligibilityTrace()
    state = obs_to_state(observation)
    while True:
        #action = env.action_space.sample()
        action = tdq.selectAction(obs_to_state(observation), greedy=False)

        a_to_take = consec_actions[0]
        if observation[0] > 0.2:
            a_to_take = consec_actions[1]

        for i in range(a_to_take):
            observation, reward, done, info = env.step(action)
            state_prime = obs_to_state(observation)
            t += 1

            #env.render()
            if done and 'TimeLimit.truncated' not in info.keys():
                reward = 1.95
            #measured_reward = (observation[0] - 0.5) #+ (np.abs(observation[1])**2 * -1)
            tdq.update(state, action, state_prime, reward)
            #print(f"{state}, {action} --> {state_prime}, {reward} | {observation} {done}, {info} {t}")
            state = state_prime
            if done:
                break

        #env.render()
        if done:
            #print(np.max(tdq.qVals()), np.argmax(tdq.qVals()))
            #print(tdq.qVals())
            #print(tdq.eligibilityTrace())
            #if episode & 20 == 0:
                #tdq.printQTable()

            if reward != -1:
                tdq.epsDecay()
                #print("WOOHOO!!!!")
                success += 1
                #print(tdq.qVals())
                #env.render()
                print(f"Success {success} on episode {episode} ({success/episode}) achieved after {t+1} timesteps: {action} -> {observation}, {obs_to_state(observation)}, {reward}, {info}")
                #print(tdq.eligibilityTrace())
            break

print(tdq.printQTable())
#print(tdq.eligibilityTrace())
test_success = 0
test_episodes = 20
t=0
for episode in range(test_episodes):
    observation = env.reset()
    while True:
        #action = env.action_space.sample()
        action = tdq.selectAction(obs_to_state(observation), greedy=True)

        observation, reward, done, info = env.step(action)
        t += 1
        env.render()
        if done:
            if 'TimeLimit.truncated' not in info.keys():
                test_success += 1
                print(f"Test success {test_success} on episode {episode} ({test_success/(episode+1)}) achieved after {t+1} timesteps")
            else:
                print("Fail")
            break
        if done:
            break
env.close()