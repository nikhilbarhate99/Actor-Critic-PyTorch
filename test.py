from model import ActorCritic
import torch
import gym
from PIL import Image
def test():
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()
    policy.load_state_dict(torch.load('./preTrained/LunarLander TWO.pth'))
    render = True
    save_gif = False

    for i_episode in range(1, 10):
        state = env.reset()
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.png'.format(t))
            if done:
                break
            
if __name__ == '__main__':
    test()
