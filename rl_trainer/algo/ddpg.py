import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys

from rl_trainer.algo.maddpg import MultiAgentReplayBuffer

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from rl_trainer.replay_buffer import ReplayBuffer
from rl_trainer.common import soft_update, hard_update, device
from rl_trainer.algo.network import Actor, Critic


class DDPG:

    def __init__(self, obs_dim, act_dim, num_agent, args,agent_idx):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        self.agent_idx = agent_idx
        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor(obs_dim, act_dim, num_agent, args,agent_idx, self.output_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agent, args,agent_idx).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # Initialise target network and critic network with ξ' ← ξ and θ' ← θ
        self.actor_target = Actor(obs_dim, act_dim, num_agent, args,agent_idx, self.output_activation).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, num_agent, args,agent_idx).to(self.device)
        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size,num_agent)

        self.c_loss = None
        self.a_loss = None

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps or evaluation:
            obs = np.array([obs])
            obs = torch.Tensor([obs]).to(self.device)
            #obs = torch.Tensor(obs).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()[0]
        else:
            action = self.random_action()

        self.eps *= self.decay_speed
        return action
    def choose_action2(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps or evaluation:
            obs = torch.Tensor(obs).to(self.device)
            #obs = torch.Tensor(obs).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()
        else:
            action = self.random_action()

        self.eps *= self.decay_speed
        return action

    def random_action(self):
        if self.output_activation == 'tanh':
            #return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
            return np.random.uniform(low=-1, high=1, size=(1, self.act_dim))
        #return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(1, self.act_dim))

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample a greedy_min mini-batch of M transitions from R
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)

        # Compute target value for each agents in each transition using the Bi-RNN
        with torch.no_grad():
            target_next_actions = self.actor_target(next_state_batch)
            target_next_q = self.critic_target(next_state_batch, target_next_actions)
            q_hat = reward_batch + self.gamma * target_next_q * (1 - done_batch)

        # Compute critic gradient estimation according to Eq.(8)
        main_q = self.critic(state_batch, action_batch)
        loss_critic = torch.nn.MSELoss()(q_hat, main_q)

        # Update the critic networks based on Adam
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Compute actor gradient estimation according to Eq.(7)
        # and replace Q-value with the critic estimation
        loss_actor = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Update the actor networks based on Adam
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        # Update the target networks
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load trained_model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor{}_".format(self.agent_idx) + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path,"critic{}_".format(self.agent_idx) + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor{}_".format(self.agent_idx) + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic{}_".format(self.agent_idx) + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)



class maddpg:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.agents = []
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        #self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, num_agent)
        self.c_loss = None
        self.a_loss = None
        self.step = 0
        self.max_episodes = 10000
        #obs_dim is the number of dimensions of an actor obs, critic dimision is obs_dim*num_agent\
        obs_dims = []
        for i in range(num_agent):
            obs_dims.append(obs_dim)
        self.MultiAgentReplayBuffer = MultiAgentReplayBuffer(args.buffer_size,obs_dim*num_agent,obs_dims,act_dim,num_agent, args.batch_size)
        for agent_idx in range(num_agent):
            self.agents.append(DDPG(obs_dim, act_dim, num_agent, args,agent_idx))

    def choose_action(self,obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(obs[agent_idx])
            actions.append(action)
        self.eps *= self.decay_speed
        #print(actions)
        #actions = np.array(actions)
        return actions

    def update(self):
        if not self.MultiAgentReplayBuffer.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = self.MultiAgentReplayBuffer.sample_buffer()

        a_loss = []
        c_loss = []
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_idx],
                                  dtype=torch.float).to(device)

            new_pi = agent.actor_target.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = torch.tensor(actor_states[agent_idx],
                                 dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi.detach())
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)
        #print("I'm here {}".format(len(self.agents)))

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.critic_target.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
           # print(critic_value_)
            critic_value = agent.critic.forward(states, old_actions).flatten()
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            #critic loss
            #critic_loss = torch.nn.MSELoss()(target.float(), critic_value.float())
            critic_loss = torch.mean((target.float()-critic_value.float())**2)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            clip_grad_norm_(agent.critic.parameters(), 1)
            agent.critic_optimizer.step()
            #actor loss
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            clip_grad_norm_(agent.actor.parameters(), 1)
            agent.actor_optimizer.step()

            ep_ratio = 1 - (self.step / self.max_episodes)
            for g in agent.actor_optimizer.param_groups:
                lr_now = self.a_lr * ep_ratio
                g['lr'] = lr_now
            for g in agent.critic_optimizer.param_groups:
                lr_now = self.a_lr * ep_ratio
                g['lr'] = lr_now

            soft_update(agent.actor, agent.actor_target, self.tau)
            soft_update(agent.critic, agent.critic_target, self.tau)
            a_loss.append(np.array(actor_loss.detach().cpu().numpy()))
            c_loss.append(np.array(critic_loss.detach().cpu().numpy()))
        a_loss = np.array(a_loss)
        c_loss = np.array(c_loss)
        self.a_loss = np.mean(a_loss)
        self.c_loss = np.mean(c_loss)

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        for agent_idx, agent in enumerate(self.agents):
            agent.load_model(run_dir, episode)

    def save_model(self, run_dir, episode):
        for agent_idx, agent in enumerate(self.agents):
            agent.save_model(run_dir, episode)

'''
        def update(self):
        if not self.MultiAgentReplayBuffer.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = self.MultiAgentReplayBuffer.sample_buffer()

        a_loss = []
        c_loss = []
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_idx],
                                  dtype=torch.float).to(device)

            new_pi = agent.actor_target.forward(new_states).detach()

            all_agents_new_actions.append(new_pi)
            mu_states = torch.tensor(actor_states[agent_idx],
                                 dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi.detach())
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)
        #print("I'm here {}".format(len(self.agents)))

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.critic_target.forward(states_, new_actions).detach().flatten()
            critic_value_[dones[:, 0]] = 0.0
            print(critic_value_)
            critic_value = agent.critic.forward(states, old_actions).flatten()
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            target = target.detach()
            #critic loss
            critic_loss = torch.nn.MSELoss()(target.float(), critic_value.float())
            agent.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            clip_grad_norm_(agent.critic.parameters(), 1)
            agent.critic_optimizer.step()
            #actor loss
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            clip_grad_norm_(agent.actor.parameters(), 1)
            agent.actor_optimizer.step()


            soft_update(agent.actor, agent.actor_target, self.tau)
            soft_update(agent.critic, agent.critic_target, self.tau)
            a_loss.append(np.array(actor_loss.detach().cpu().numpy()))
            c_loss.append(np.array(critic_loss.detach().cpu().numpy()))
        a_loss = np.array(a_loss)
        c_loss = np.array(c_loss)
        self.a_loss = np.mean(a_loss)
        self.c_loss = np.mean(c_loss)

'''
'''
    def update(self):
        if not self.MultiAgentReplayBuffer.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = self.MultiAgentReplayBuffer.sample_buffer()

        a_loss = []
        c_loss = []
        state_batch = torch.Tensor(actor_states).to(self.device)
        action_batch = torch.Tensor(actions).to(self.device)
        reward_batch = torch.Tensor(rewards).to(self.device)
        next_state_batch = torch.Tensor(actor_new_states).to(self.device)
        done_batch = torch.Tensor(dones).to(self.device)
        states = torch.Tensor(states).to(self.device)
        states_ = torch.Tensor(states_).to(self.device)
        target_next_actions = []
        old_agents_actions = []
        all_agent_actions = []
        for agent_idx, agent in enumerate(self.agents):
            with torch.no_grad():
                target_next_action = agent.actor_target(next_state_batch[agent_idx])
                target_next_actions.append(target_next_action)
                all_agent_actions.append(agent.actor(state_batch[agent_idx]))
                old_agents_actions.append(action_batch[agent_idx])
        #target_next_actions = np.array(target_next_actions)
        target_next_actions = torch.cat([acts for acts in target_next_actions], dim=1)
        #old_agents_actions = np.array(old_agents_actions)
        all_agent_actions= torch.cat([acts for acts in all_agent_actions], dim=1)
        old_agents_actions = torch.cat([acts for acts in old_agents_actions], dim=1)
        # Compute target value for each agents in each transition using the Bi-RNN
        for agent_idx, agent in enumerate(self.agents):
            with torch.no_grad():
                target_next_q = agent.critic_target(states_, target_next_actions)
                #print(reward_batch[:,agent_idx]+self.gamma *target_next_q.flatten()* (1 - done_batch[:,agent_idx]))
                q_hat = reward_batch[:,agent_idx] + self.gamma * target_next_q.flatten() * (1 - done_batch[:,agent_idx])
            # Compute critic gradient estimation according to Eq.(8)
            main_q = agent.critic(states, old_agents_actions)
           # print( reward_batch)
            loss_critic = torch.nn.MSELoss()(q_hat, main_q.flatten())

            # Update the critic networks based on Adam
            agent.critic_optimizer.zero_grad()
            loss_critic.backward(retain_graph = True)
            clip_grad_norm_(agent.critic.parameters(), 1)
            agent.critic_optimizer.step()

            # Compute actor gradient estimation according to Eq.(7)
            # and replace Q-value with the critic estimation
            loss_actor = -agent.critic(states, all_agent_actions).mean()

            # Update the actor networks based on Adam
            agent.actor_optimizer.zero_grad()
            loss_actor.backward(retain_graph = True)
            clip_grad_norm_(agent.actor.parameters(), 1)
            agent.actor_optimizer.step()

            self.c_loss = loss_critic.item()
            self.a_loss = loss_actor.item()

            # Update the target networks
            soft_update(agent.actor, agent.actor_target, agent.tau)
            soft_update(agent.critic, agent.critic_target, agent.tau)
            a_loss.append(np.array(loss_actor.detach().cpu().numpy()))
            c_loss.append(np.array(loss_critic.detach().cpu().numpy()))
        a_loss = np.array(a_loss)
        c_loss = np.array(c_loss)
        self.a_loss = np.mean(a_loss)
        self.c_loss = np.mean(c_loss)
'''
