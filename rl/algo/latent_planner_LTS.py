import numpy as np
import torch
import datetime
import os
import os.path as osp
import sys

from rl import logger

from rl.utils import mpi_utils
from rl.algo.core import BaseAlgo

from rl.replay.planner import sample_her_transitions
from rl.search.latent_planner import Planner


class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        test_env,
        agent, replay, monitor, learner,
        reward_func,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            agent, replay, monitor, learner,
            reward_func,
            name=name,
        )
        self.planner = Planner(agent, replay, monitor, args)   # TODO: Floyd algorithm?
        self.test_env = test_env
        self.fps_landmarks = None
        self._clusters_initialized = False
    
    # Are there enough samples in the replay buffer to form centroids/landmarks and start planning?
    def can_plan(self):
        replay_big_enough = self.replay.current_size > self.args.start_planning_n_traj
        #print("can_plan = ", replay_big_enough, "\nself.replay.current_size = ", self.replay.current_size)
        return replay_big_enough
    
    # Get action with optional noise or randomisation
    def get_actions(self, ob, bg, a_max=1.0, act_randomly=False):
        act = self.agent.get_actions(ob, bg)
        if self.args.noise_eps > 0.0:   # Add noise to act vector
            act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
            act = np.clip(act, -a_max, a_max)
        if self.args.random_eps > 0.0:  # Add value based on binomial distribution
            a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
            mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
            if self.num_envs > 1:
                mask = np.expand_dims(mask, -1)
            act += mask * (a_rand - act)
        if act_randomly:  # Select random action, ignore self.agent.get_actions(..)
            act = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
        return act
    
    # Train agent and clustering
    def agent_optimize(self):
        self.timer.start('train')
        
        for n_train in range(self.args.n_batches):
            batch = self.replay.sample(batch_size=self.args.batch_size)
            self.learner.update(batch, train_embed=True)
            self.opt_steps += 1
            if self.opt_steps % self.args.target_update_freq == 0:
                self.learner.target_update()
            
            # Cluster training
            if self.opt_steps % self.args.fps_sample_freq == 0 or self.fps_landmarks is None:
                self.fps_landmarks, _ = self.planner.fps_sample_batch(
                    initial_sample=self.args.initial_sample, batch_size=self.args.latent_batch_size)
            if self.opt_steps % self.args.cluster_update_freq == 0:
                if not self._clusters_initialized and self.agent.cluster.n_mix <= self.args.latent_batch_size:
                    self.learner.initialize_cluster(self.fps_landmarks[:self.agent.cluster.n_mix])
                self.learner.update_cluster(self.fps_landmarks, to_train=self._clusters_initialized)
        
        self.timer.end('train')
        self.monitor.store(TimePerTrainIter=self.timer.get_time('train') / self.args.n_batches)
    

    # Fill replay buffer with new samples, possibility to optimize trainer concurrently
    def collect_experience(self, act_randomly=False, train_agent=True):
        ob_list, ag_list, bg_list, a_list = [], [], [], []
        observation = self.env.reset()
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']
        ag_origin = ag.copy()

        a_max = self.env_params['action_max']
        self.planner.reset()
        can_plan = self.can_plan()  # Enough samples in replay buffer for planning?

        if not act_randomly and can_plan:
            self.planner.update(goals=bg.copy(), test_time=False)
        
        # Perform actions in environment
        for timestep in range(self.env_params['max_timesteps']):
            act = self.get_actions(ob, bg, a_max=a_max, act_randomly=act_randomly)
            
            # Let the agent decide the action
            if not act_randomly and can_plan and np.random.uniform() < self.args.plan_eps:
                sub_goals = self.planner.get_subgoals(ob, bg.copy())  # Based on this, bg = final goal
                act = self.agent.get_actions(ob, sub_goals)
            else:
                self.planner.forward_empty_step()
            
            ob_list.append(ob.copy())
            ag_list.append(ag.copy())
            bg_list.append(bg.copy())
            a_list.append(act.copy())

            observation, _, _, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(Inner_Train_AgChangeRatio=np.mean(ag_changed))
            self.total_timesteps += self.num_envs * self.n_mpi

            for every_env_step in range(self.num_envs):
                self.env_steps += 1
                if self.env_steps % self.args.optimize_every == 0 and train_agent:
                    self.agent_optimize()  # Train agent concurrently
            
        ob_list.append(ob.copy())
        ag_list.append(ag.copy())
        
        # Every entry consists out of state, subgoal, final goal, action list
        experience = dict(ob=ob_list, ag=ag_list, bg=bg_list, a=a_list)
        experience = {k: np.array(v) for k, v in experience.items()}

        if experience['ob'].ndim == 2:
            experience = {k: np.expand_dims(v, 0) for k, v in experience.items()}
        else:
            experience = {k: np.swapaxes(v, 0, 1) for k, v in experience.items()}
        
        bg_achieve = self.reward_func(bg, ag, None) + 1.  # Is final goal reached?

        self.monitor.store(TrainSuccess=np.mean(bg_achieve))
        ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
        self.monitor.store(TrainAgChangeRatio=np.mean(ag_changed))
        self.monitor.store(Train_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
        self.replay.store(experience)
        self.update_normalizer(experience)
    
    def update_normalizer(self, buffer):
        # Hindsight Experience Replay
        transitions = sample_her_transitions(
            buffer=buffer, reward_func=self.reward_func,
            batch_size=self.env_params['max_timesteps'] * self.num_envs,
            future_step=self.args.future_step,
            future_p=self.args.future_p)
        self.agent.normalizer_update(obs=transitions['ob'], goal=transitions['bg'])
    
    def initialize_clusters(self):
        pts, _ = self.planner.fps_sample_batch(
            initial_sample=self.args.initial_sample, batch_size=self.agent.cluster.n_mix)
        self.learner.initialize_cluster(pts)
        self._clusters_initialized = True
    

    # Starting point of training the agent
    def run(self):
        # Fill the replay buffer before starting to train the agent
        # Actions are randomly selected
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(act_randomly=True, train_agent=False)
        
        # Train agent over N epochs
        for epoch in range(self.args.n_epochs):
            if self.planner.scheduler:
                if self.can_plan and self.planner.scheduler.empty:
                    self.planner.scheduler.activate(epoch)
                self.planner.scheduler.set_epoch(epoch)
            if mpi_utils.is_root():
                print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
                sys.stdout.flush()
            
            # Epoch consists out of iterations
            for n_iter in range(self.args.n_cycles):
                if mpi_utils.is_root():
                    print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                    sys.stdout.flush()
                self.timer.start('rollout')
                
                # Iteration consists out of rollouts
                for n_rollout in range(self.args.num_rollouts_per_mpi):
                    self.collect_experience(train_agent=True)  # Collect samples and train concurrently
                    if self.can_plan() and not self._clusters_initialized:
                        self.initialize_clusters()
                
                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout') / self.args.num_rollouts_per_mpi)
            
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(opt_steps=self.opt_steps)
            self.monitor.store(replay_size=self.replay.current_size)
            self.monitor.store(replay_fill_ratio=float(self.replay.current_size / self.replay.size))
            
            her_success = self.run_eval(use_test_env=False)

            # Samples are gathered, now start using those samples to plan
            # First train the planning
            train_env_plan_success = self.run_train_env_plan_eval() if self._clusters_initialized else 0.0
            # Then evaluate the planning
            test_env_plan_success = self.run_test_env_plan_eval() if self._clusters_initialized \
                else self.run_eval(use_test_env=True)

            if mpi_utils.is_root():
                print('Epoch %d her eval %.3f, test-env plan %.3f, train-env plan %.3f' %
                      (epoch, her_success, test_env_plan_success, train_env_plan_success))
                print('Log Path:', self.log_path)
            logger.record_tabular("Epoch", epoch)
            self.monitor.store(Test_TrainEnv_HerSuccessRate=her_success)
            self.monitor.store(Test_TrainEnv_PlanSuccessRate=train_env_plan_success)
            self.monitor.store(Test_TestEnv_PlanSuccessRate=test_env_plan_success)
            self.log_everything()
            self.save_all(self.model_path)


    # Run demo of agent, creates plots
    def demo(self, args):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D
        
        #FIXME absolutely remove this line!!!
        print("success_rate = ", self.run_test_env_plan_eval())

        if self.planner.scheduler and self.planner.scheduler.empty:
            self.planner.scheduler.activate(27)

        def evaluate(args):
            import imageio
            T, fail = [], []
            success, sum_duration, sum_subgoals_used = 0, 0, 0
            env = self.test_env
            demo_length = max(args.video, args.plot)

            with imageio.get_writer(args.output+".mp4", fps=30) as video:
                for i in range(demo_length):
                    # Prepare environment for new playout
                    info = None
                    obs = env.reset()
                    ob = obs['observation']
                    bg = obs['desired_goal']
                    ag = obs['achieved_goal']

                    if i == 0:
                        start = ag
                        goal = bg

                    self.planner.reset()
                    self.planner.update(goals=bg.copy(), test_time=True)

                    t, s = [], []

                    # Perform playout
                    for timestep in range(env._max_episode_steps):
                        if i < args.video:
                            video.append_data(env.render(mode='rgb_array'))
                        
                        sub_goals = self.planner.get_subgoals(ob, bg.copy())
                        """
                        #bg.copy()
                        sub_goals = np.expand_dims(bg.copy(), axis=0)#self.planner.get_subgoals(ob, bg.copy())
                        """
                        s.append(tuple(sub_goals[0]))

                        # Plot next subgoal
                        plt.plot(sub_goals[0][0], sub_goals[0][1], linestyle='None', marker='o', color='limegreen', markersize=7, alpha=0.2)

                        # New action
                        a = self.agent.get_actions(ob, sub_goals)
                        obs, _, _, info = env.step(a)
                        ob = obs['observation']
                        bg = obs['desired_goal']
                        ag = obs['achieved_goal']

                        t.append(ag)

                        if info['is_success']:
                            if i < args.plot:
                                success += 1
                                t.append(bg)
                                sum_duration += timestep
                            break
                    
                    if not info['is_success'] and i < args.plot:
                        sum_duration += env._max_episode_steps
                        fail.append(ag)
                    if i < args.plot:
                        T.append(t)
                        sum_subgoals_used += len(np.unique(s, axis=0))
            
            # Plot paths
            for t in T:
                plt.plot([i[0] for i in t], [i[1] for i in t], color='navy', alpha=0.1)
            plt.plot(start[0], start[1], linestyle='None', marker='D', color='navy', markersize=10)     # STARTING POINT
            plt.plot(goal[0], goal[1], linestyle='None', marker='*', color='darkorange', markersize=8)  # GOAL

            for f in fail:
                plt.plot(f[0], f[1], color="red", marker="x", alpha=1.0)

            if args.plot == 0:
                return 0, 0, 0
            return round(success/args.plot, 3), round(sum_duration/args.plot, 3), round(sum_subgoals_used/args.plot, 3)
        
        success_rate, duration, subgoals = evaluate(args)
        if args.plot > 0:
            # Plot landmarks
            A = self.agent.ae.decoder(self.agent.cluster.comp_mean)
            X = [i[0] for i in A.cpu().detach().numpy()]
            Y = [i[1] for i in A.cpu().detach().numpy()]
            plt.plot(X, Y, linestyle='None', marker='.', color='black')

            plt.xlim(-2, 10)
            plt.ylim(-2, 10)
            plt.tick_params(
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                left=False,
                right=False,
                labelleft=False
            )

            plt.gca().set_aspect("equal")

            # Plot non-accessible area
            rect = patches.Rectangle((-2, 2), 8, 4, edgecolor="k", linewidth=0.7, facecolor="lightgrey")
            plt.gca().add_patch(rect)

            # Create textbox containing various statistics over the testruns
            rx, ry = rect.get_xy()  # Compute position of textbox
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/2.0
            plt.annotate(f"Success rate: {success_rate}\nAvg. timesteps taken: {duration}\nAvg. number of subgoals: {subgoals}", (cx, cy), color='k', bbox=dict(facecolor="white", edgecolor="black", linewidth=0.7, boxstyle="round,pad=0.3"), fontfamily="monospace", fontsize=9, va="center", ha="center", ma="left")

            # Create legend for path-plot
            if args.legend:
                entries = [Line2D([0], [0], color="navy", lw=1.5, label="Path"),
                        Line2D([0], [0], marker='.', color="white", markerfacecolor="black", markeredgecolor="black", label="Landmark"),
                        Line2D([0], [0], marker="D", color="white", markerfacecolor="navy", markeredgecolor="navy", label="Starting point"),
                        Line2D([0], [0], marker="*", markersize=9, color="white", markerfacecolor="darkorange", markeredgecolor="darkorange", label="Goal"),
                        Line2D([0], [0], marker="o", markersize=7, color="white", markerfacecolor="limegreen", markeredgecolor="limegreen", label="Subgoal"),
                        Line2D([0], [0], marker="x", color="white", markerfacecolor="red", markeredgecolor="red", label="Failed run")]
                plt.legend(handles=entries, ncol=3, loc='upper center', columnspacing=1.5, bbox_to_anchor=(0.5, -0.05))

            plt.title(r"$d^2$")
            plt.savefig(args.output+".png", dpi=600, bbox_inches="tight")

    # Evaluate the planning-component of L3P
    def run_test_env_plan_eval(self):
        env = self.env
        if hasattr(self, 'test_env'):
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        if self.planner.scheduler and self.planner.scheduler.empty:
            self.planner.scheduler.activate(9)
        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            ag_origin = ag.copy()

            # Prepare planner with new final goal bg
            self.planner.reset()
            self.planner.update(goals=bg.copy(), test_time=True)

            for timestep in range(env._max_episode_steps):
                sub_goals = self.planner.get_subgoals(ob, bg.copy())
                a = self.agent.get_actions(ob, sub_goals)
                observation, _, _, info = env.step(a)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
                ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
                self.monitor.store(Inner_PlanTest_AgChangeRatio=np.mean(ag_changed))
                #if info['is_success'] == 1.0:
                    #print("stopped early!")
                    #break
            
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(TestPlan_AgChangeRatio=np.mean(ag_changed))
            self.monitor.store(TestPlan_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())

            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info['is_success'] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                if info['is_success'] == 1.0:
                    total_success_count += 1
        
        success_rate = total_success_count / total_trial_count
        if mpi_utils.use_mpi():
            success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]
        print("success_rate = ", success_rate)
        return success_rate
    
    # Train the planning-component of L3P
    def run_train_env_plan_eval(self):
        env = self.env
        total_success_count = 0
        total_trial_count = 0

        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']

            # Prepare planner with new final goal bg
            self.planner.reset()
            self.planner.update(goals=bg.copy(), test_time=True)

            for timestep in range(env._max_episode_steps):
                sub_goals = self.planner.get_subgoals(ob, bg.copy())
                a = self.agent.get_actions(ob, sub_goals)
                observation, _, _, info = env.step(a)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
            
            self.monitor.store(TrainEnvTestPlan_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())

            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info['is_success'] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                if info['is_success'] == 1.0:
                    total_success_count += 1
        
        success_rate = total_success_count / total_trial_count
        if mpi_utils.use_mpi():
            success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]
        return success_rate
    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']
