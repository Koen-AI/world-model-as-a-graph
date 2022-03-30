#baseline
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_workers 1 --n_epochs 360 --batch_size 512 --grad_value_clipping -1.0 --seed 626262 
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_workers 1 --n_epochs 360 --batch_size 512 --grad_value_clipping -1.0 --seed 646464 
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_workers 1 --n_epochs 360 --batch_size 512 --grad_value_clipping -1.0 --seed 636363 

#no dist clipping
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_workers 1 --n_epochs 360 --batch_size 512 --grad_value_clipping -1.0 --seed 6262 --dist_clip -1000000 
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_workers 1 --n_epochs 360 --batch_size 512 --grad_value_clipping -1.0 --seed 6464 --dist_clip -1000000 
python -m rl.main_latent --env_name PointMaze-v1 --test_env_name PointMazeTest-v1 --n_workers 1 --n_epochs 360 --batch_size 512 --grad_value_clipping -1.0 --seed 6363 --dist_clip -1000000 



