from model import ActorCritic
import torch
import gym
from PIL import Image

def test(n_episodes=10):
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()
    
    policy.load_state_dict(torch.load('./preTrained/LunarLander_TWO.pth'))
    
    render = True
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
            
if __name__ == '__main__':
    test()
