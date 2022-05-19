import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.ddpg import DDPG
from common import *
from log_path import *
from env.chooseenv import make


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    ctrl_agent_index2 = [3, 4, 5]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 26
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model1 = DDPG(obs_dim, act_dim, ctrl_agent_num, args)
    model2 = DDPG(obs_dim, act_dim, ctrl_agent_num, args)

    if args.load_model:
        args.load_model_run=13
        args.load_model_run_episode=20000
        args.load_model_run2=13
        args.load_model_run_episode2=20000
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model1.load_model(load_dir, episode=args.load_model_run_episode)
        load_dir2 = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run2))
        model2.load_model(load_dir2, episode=args.load_model_run_episode2)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained trained_model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs1 = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)
        obs2 = get_observations(state_to_training, ctrl_agent_index2, obs_dim, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:

            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model1.choose_action(obs1)
            logit2 = model1.choose_action(obs2)

            # ============================== add opponent actions =================================
            # we use rule-based greedy agent here. Or, you can switch to random agent.
            actions = logits_greedy(state_to_training, logits,logit2, height, width)
            # actions = logits_random(act_dim, logits)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)
            next_obs2 = get_observations(next_state_to_training, ctrl_agent_index2, obs_dim, height, width)

            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward,episode_reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward,episode_reward, score=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward,episode_reward, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward,episode_reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward,episode_reward, score=4)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, episode_reward,score=0)

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward2 = get_reward(info, ctrl_agent_index2, reward, episode_reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward2 = get_reward(info, ctrl_agent_index2, reward, episode_reward, score=2)
                else:
                    step_reward2 = get_reward(info, ctrl_agent_index2, reward, episode_reward, score=0)
                

            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward2 = get_reward(info, ctrl_agent_index2, reward, episode_reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward2 = get_reward(info, ctrl_agent_index2, reward, episode_reward,score=4)
                else:
                    step_reward2 = get_reward(info, ctrl_agent_index2, reward, episode_reward, score=0)

            done = np.array([done] * ctrl_agent_num)

            # ================================== collect data ========================================
            # Store transition in R
            i=1
            if episode%2000==0:
                i=1-i
                model1.epsilon=0.8
                model2.epsilon=0.8
                

            
            if i==1:
                model1.replay_buffer.push(obs1, logits, step_reward, next_obs, done)
                model1.update()
            else:
                model2.replay_buffer.push(obs2, logit2, step_reward2, next_obs2, done)
                model2.update()

            obs1 = next_obs
            obs2 = next_obs2
            step += 1

            if args.episode_length <= step or (True in done):

                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} epsilon: {model1.eps:.2f}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')

                reward_tag = 'reward'
                loss_tag = 'loss'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                    'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})
                if model1.c_loss and model1.a_loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'actor': model1.a_loss, 'critic': model1.c_loss})

                if model1.c_loss and model1.a_loss:
                    print(f'\t\t\t\ta_loss {model1.a_loss:.3f} c_loss {model1.c_loss:.3f}')

                if episode % args.save_interval == 0:
                    model1.save_model(run_dir, episode)

                env.reset()
                break
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.98, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.8, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)

    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)
