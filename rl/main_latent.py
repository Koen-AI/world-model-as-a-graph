
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env_name', type=str, default='PointMaze-v1')
    parser.add_argument('--test_env_name', type=str, default='PointMazeTest-v1')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_dir', type=str, default='experiments/')
    parser.add_argument('--ckpt_name', type=str, default='')  # Filename of checkpoint
    parser.add_argument('--resume_ckpt', type=str, default='')
    
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')  # GPU usage
    parser.add_argument('--num_rollouts_per_mpi', type=int, default=1)
    
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_cycles', type=int, default=15)
    parser.add_argument('--optimize_every', type=int, default=2)
    parser.add_argument('--n_batches', type=int, default=1)
    
    parser.add_argument('--hid_size', type=int, default=256)
    parser.add_argument('--n_hids', type=int, default=3)
    parser.add_argument('--activ', type=str, default='relu')
    parser.add_argument('--noise_eps', type=float, default=0.2)
    parser.add_argument('--random_eps', type=float, default=0.2)
    
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--future_p', type=float, default=0.85)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--clip_inputs', action='store_true')
    parser.add_argument('--clip_obs', type=float, default=200)
    
    parser.add_argument('--normalize_inputs', action='store_true')
    parser.add_argument('--clip_range', type=float, default=5)
    
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--clip_return', type=float, default=80)
    
    parser.add_argument('--action_l2', type=float, default=0.5)
    parser.add_argument('--lr_actor', type=float, default=2e-4)
    parser.add_argument('--lr_critic', type=float, default=2e-4)
    
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--target_update_freq', type=int, default=10)
    
    parser.add_argument('--n_initial_rollouts', type=int, default=100)
    parser.add_argument('--n_test_rollouts', type=int, default=15)   # Number of playouts to evaluate agent
    parser.add_argument('--play', action='store_true')
    
    parser.add_argument('--future_step', type=int, default=80)
    parser.add_argument('--plan_eps', type=float, default=0.5)
    parser.add_argument('--initial_sample', type=int, default=2000)
    parser.add_argument('--embed_epsilon', type=float, default=0.1)
    parser.add_argument('--n_latent_landmarks', type=int, default=50)
    
    parser.add_argument('--latent_batch_size', type=int, default=256)
    parser.add_argument('--fps_sample_freq', type=int, default=5)
    parser.add_argument('--cluster_update_freq', type=int, default=2)
    parser.add_argument('--n_extra_landmark', type=int, default=150)
    parser.add_argument('--embed_op', type=str, default='sum', choices=['sum', 'mean'])
    
    parser.add_argument('--ae_hid_size', type=int, default=128)
    parser.add_argument('--ae_n_hids', type=int, default=2)
    parser.add_argument('--lr_ae', type=float, default=3e-4)
    parser.add_argument('--lr_cluster', type=float, default=3e-4)
    parser.add_argument('--latent_repel', type=float, default=1.0)
    parser.add_argument('--elbo_beta', type=float, default=1.0)
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--learned_prior', action='store_true')
    
    parser.add_argument('--inf_value', type=float, default=1e6)
    parser.add_argument('--dist_clip', type=float, default=-20.0)  # -4.0
    parser.add_argument('--temp', type=float, default=1.1)
    parser.add_argument('--vi_iter', type=int, default=20)
    parser.add_argument('--local_horizon', type=int, default=10)
    parser.add_argument('--goal_eps', type=float, default=0.2)  # Radius to reach goal (since continuous problems)
    parser.add_argument('--q_offset', action='store_true')
    parser.add_argument('--cluster_std_reg', type=float, default=0.0)
    parser.add_argument('--start_planning_n_traj', type=int, default=500)
    
    parser.add_argument('--grad_norm_clipping', type=float, default=-1.0)
    parser.add_argument('--grad_value_clipping', type=float, default=5.0)
    parser.add_argument('--use_forward_empty_step', action='store_true')
    
    # Our contributions
    parser.add_argument('--square', type=bool, default=False)
    
    #Scheduler related arguments
    parser.add_argument('--d_scheduler', type=str, default=None)
    parser.add_argument('--scheduler_min', action='store_true')
    parser.add_argument('--scheduler_max', action='store_true')
    parser.add_argument('--omega_max', type=float, default=0.9)
    parser.add_argument('--omega_min', type=float, default=1.0)
    parser.add_argument('--varpi', type=float, default=0.8)
    parser.add_argument('--test_scheduler', type=bool, default=False)#only used to test scheduler class
    

    # Demo related arguments
    parser.add_argument('--video', type=int, default=0)
    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--output', type=str, default="out")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    n_threads = str(args.n_workers)

    if args.d_scheduler != None:
        if not (args.scheduler_min or args.scheduler_max):
            raise ValueError('Scheduler of type: ', args.d_scheduler, "; not used for a minimal or maximal clipping.\nPlease use '--scheduler_min' and/or '--scheduler_max'")
    import os
    os.environ['OMP_NUM_THREADS'] = n_threads   # Set number of threads utilised by torch
    os.environ['MKL_NUM_THREADS'] = n_threads
    os.environ['IN_MPI'] = n_threads
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    from rl.launcher_latent import launch
    if args.test_scheduler:
        from rl.search.scheduler import Scheduler
        scheduler = Scheduler(args)
        scheduler.plot_schedule()
        exit()
    algo = launch(args)

    
    if (args.video + args.plot > 0):
        #print("success rate = ", algo.run_test_env_plan_eval())
        algo.demo(args) 
         
    else:
        algo.run()
    
    
    
    
