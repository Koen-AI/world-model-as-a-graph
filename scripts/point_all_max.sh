python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 991 --cuda --d_scheduler exponential  --scheduler_max  #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 992 --cuda --d_scheduler exponential  --scheduler_max  #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 993 --cuda --d_scheduler exponential  --scheduler_max #--n_workers 3

python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 678 --cuda --d_scheduler linear  --scheduler_max  #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 679 --cuda --d_scheduler linear  --scheduler_max  #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 680 --cuda --d_scheduler linear  --scheduler_max  #--n_workers 3

python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 345 --cuda --d_scheduler logarithmic  --scheduler_max  #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 456 --cuda --d_scheduler logarithmic  --scheduler_max  #--n_workers 3
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_epochs 360 --buffer_size 1000000 --batch_size 512 --grad_value_clipping -1.0 --seed 567 --cuda --d_scheduler logarithmic  --scheduler_max  #--n_workers 3
