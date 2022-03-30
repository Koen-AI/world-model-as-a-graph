python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 33 --cuda --d_scheduler exponential --scheduler_min #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 65 --cuda --d_scheduler exponential --scheduler_min #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 129 --cuda --d_scheduler exponential --scheduler_min #--n_workers 3

python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 42 --cuda --d_scheduler linear --scheduler_min #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 74 --cuda --d_scheduler linear --scheduler_min #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 138 --cuda --d_scheduler linear --scheduler_min #--n_workers 3

python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 36 --cuda --d_scheduler logarithmic --scheduler_min #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 68 --cuda --d_scheduler logarithmic --scheduler_min #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 132 --cuda --d_scheduler logarithmic --scheduler_min #--n_workers 3
