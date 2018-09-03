from test import test
from model import ActorCritic
import torch
import torch.optim as optim
import gym

def train():
    # Hyperparameters : 
    env = gym.make('LunarLander-v2')
    render = True
    gamma = 0.99
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=0.03, betas=(0.81, 0.999))
    
    for i_episode in range(1, 10000):
        running_reward = 0
        state = env.reset()
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            if render and i_episode > 1000:
                env.render()
            if done:
                break
        
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()
        policy.clearMemory()
        
        if i_episode > 1000:
            torch.save(policy.state_dict(), './preTrained/LunarLander.pth')
            
        if i_episode % 10 == 0:
            print('Episode {}\tlength: {}'.format(
                i_episode, t))
        if running_reward > 1000:
            print("Solved!")
            test()
            break

if __name__ == '__main__':
    train()
