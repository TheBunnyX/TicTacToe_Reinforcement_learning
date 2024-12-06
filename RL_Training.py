import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from Environment import TicTacToeEnv

if __name__ == "__main__":
    env = make_vec_env(TicTacToeEnv, n_envs=1)
    
    # Initialize DQN agent
    model = DQN("MlpPolicy", env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=50000)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/tic_tac_toe_dqn")
    print("Model saved!")
