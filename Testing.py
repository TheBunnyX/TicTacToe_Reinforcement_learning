import numpy as np
from stable_baselines3 import DQN
from Environment import TicTacToeEnv

def human_play(env):
    print("Human plays as X (1).")
    obs, _ = env.reset()
    env.render()

    while True:
        # Human input
        try:
            action = int(input("Enter position (0-8) : "))
            if action < 0 or action > 8:
                raise ValueError("Invalid input. Please enter a number between 0 and 8.")
        except ValueError as e:
            print(e)
            continue
        
        obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            if reward > 0:
                print("Human wins!")
            elif reward == 0:
                print("It's a draw!")
            else:
                print("Invalid move! RL Agent wins!")
            break

        # RL agent's turn
        print("RL Agent's turn...")
        action, _ = model.predict(obs, deterministic=True)  # No flattening needed
        obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            if reward > 0:
                print("RL Agent wins!")
            elif reward == 0:
                print("It's a draw!")
            break

if __name__ == "__main__":
    # Load trained model
    model = DQN.load("models/tic_tac_toe_dqn")

    # Initialize the environment
    env = TicTacToeEnv()

    # Play against the human
    human_play(env)
