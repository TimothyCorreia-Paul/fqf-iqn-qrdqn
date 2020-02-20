import torch
from torch.optim import Adam, RMSprop

from fqf_iqn.model import FQF
from fqf_iqn.utils import disable_gradients, update_params,\
    calculate_quantile_huber_loss, evaluate_quantile_at_action, LinearAnneaer
from .base_agent import BaseAgent


class FQFAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, num_taus=32, num_cosines=64, ent_coef=0,
                 kappa=1.0, quantile_lr=5e-5, fraction_lr=2.5e-9,
                 memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0):
        super(FQFAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = FQF(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, num_taus=num_taus,
            num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)
        # Target network.
        self.target_net = FQF(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, num_taus=num_taus,
            num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        # NOTE: In the paper, learning rate of Fraction Proposal Net is
        # swept between (0, 2.5e-5) and finally fixed at 2.5e-9. However,
        # I don't sweep it for simplicity.
        self.quantile_optim = Adam(
            list(self.online_net.dqn_net.parameters())
            + list(self.online_net.quantile_net.parameters()),
            lr=quantile_lr, eps=1e-2/batch_size)
        self.fraction_optim = RMSprop(
            self.online_net.fraction_net.parameters(),
            lr=fraction_lr, alpha=0.95, eps=0.00001)

        # NOTE: The author said the training of Fraction Proposal Net is
        # unstable and value distribution degenerates into a deterministic
        # one rarely (e.g. 1 out of 20 seeds). So you can use entropy of value
        # distribution as a regularizer to stabilize (but possibly slow down)
        # training.
        self.ent_coef = LinearAnneaer(
            start_value=ent_coef, end_value=0,
            num_steps=(num_steps-start_steps)//update_interval)
        self.num_taus = num_taus
        self.num_cosines = num_cosines
        self.kappa = kappa

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.

        with torch.no_grad():
            # Calculate Q and get greedy action.
            action = self.online_net.calculate_q(states=state).argmax().item()

        return action

    def learn(self):
        self.learning_steps += 1
        self.ent_coef.step()

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        # Calculate embeddings of current states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)
        # Calculate fractions and entropies.
        taus, tau_hats, entropies =\
            self.online_net.calculate_fractions(
                state_embeddings=state_embeddings)

        fraction_loss = self.calculate_fraction_loss(
            state_embeddings, taus, tau_hats)

        quantile_loss = self.calculate_quantile_loss(
            state_embeddings, taus, tau_hats, actions, rewards,
            next_states, dones)

        entropy_loss = -self.ent_coef.get() * entropies.mean()

        update_params(
            self.fraction_optim, fraction_loss + entropy_loss,
            networks=[self.online_net.fraction_net], retain_graph=True,
            grad_cliping=self.grad_cliping)
        update_params(
            self.quantile_optim, quantile_loss + entropy_loss,
            networks=[self.online_net.dqn_net, self.online_net.quantile_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/fraction_loss', fraction_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/entropy_loss', entropy_loss.detach().item(),
                self.learning_steps)

            with torch.no_grad():
                mean_q = self.online_net.calculate_q(
                    taus, tau_hats, state_embeddings=state_embeddings).mean()

            self.writer.add_scalar(
                'stats/mean_Q', mean_q, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_entropy_of_value_distribution',
                entropies.mean().detach().item(), self.learning_steps)
            self.writer.add_scalar(
                'time/mean_learning_time', self.learning_time.get(),
                self.learning_steps)

    def calculate_fraction_loss(self, state_embeddings, taus, tau_hats):

        # NOTE: Gradients of taus should be calculated at current state and
        # any possible actions, however, calculating fractions of each state
        # and action is quite heavy. So Fraction Proposal Net depends only on
        # the state and fractions are shared across all actions.

        batch_size = state_embeddings.shape[0]

        with torch.no_grad():
            quantile_tau_i = self.online_net.calculate_quantiles(
                taus=taus[:, 1:-1], state_embeddings=state_embeddings)
            assert quantile_tau_i.shape == (
                batch_size, self.num_taus-1, self.num_actions)

            quantile_hat_tau_i = self.online_net.calculate_quantiles(
                taus=tau_hats[:, 1:], state_embeddings=state_embeddings)
            assert quantile_hat_tau_i.shape == (
                batch_size, self.num_taus-1, self.num_actions)

            quantile_hat_tau_i_minus_1 = self.online_net.calculate_quantiles(
                taus=tau_hats[:, :-1], state_embeddings=state_embeddings)
            assert quantile_hat_tau_i_minus_1.shape == (
                batch_size, self.num_taus-1, self.num_actions)

        gradient_of_taus =\
            2*quantile_tau_i - quantile_hat_tau_i - quantile_hat_tau_i_minus_1

        # Gradients of the network parameters and corresponding loss
        # are calculated using chain rule.
        fraction_loss = (
            gradient_of_taus.mean(dim=2) * taus[:, 1:-1]).sum(dim=1).mean()

        return fraction_loss

    def calculate_quantile_loss(self, state_embeddings, taus, tau_hats,
                                actions, rewards, next_states, dones):

        # Calculate quantile values of current states and all actions.
        current_s_quantiles = self.online_net.calculate_quantiles(
            tau_hats, state_embeddings=state_embeddings)

        # Get quantile values of current states and current actions.
        current_sa_quantiles = evaluate_quantile_at_action(
            current_s_quantiles, actions)
        assert current_sa_quantiles.shape == (
            self.batch_size, self.num_taus, 1)

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)
                next_taus, next_tau_hats, _ =\
                    self.target_net.calculate_fractions(
                        state_embeddings=next_state_embeddings)
                next_q = self.target_net.calculate_q(
                    taus=next_taus, tau_hats=next_tau_hats,
                    state_embeddings=next_state_embeddings)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate features of next states.
            if self.double_q_learning:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)
                next_taus, next_tau_hats, _ =\
                    self.target_net.calculate_fractions(
                        state_embeddings=next_state_embeddings)

            # Calculate quantile values of next states and all actions.
            next_s_quantiles = self.target_net.calculate_quantiles(
                taus=next_tau_hats, state_embeddings=next_state_embeddings)

            # Get quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                next_s_quantiles, next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (
                self.batch_size, 1, self.num_taus)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, 1, self.num_taus)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (
            self.batch_size, self.num_taus, self.num_taus)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, tau_hats.detach(), self.kappa)

        return quantile_huber_loss
