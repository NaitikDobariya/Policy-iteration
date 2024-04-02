import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, grid_size, objects, step_reward=-1, gamma=0.9, theta=1e-2):
        self.ROWS = grid_size[0]
        self.COLS = grid_size[1]
        self.objects = objects
        self.gamma = gamma
        self.theta = theta

        self.action_space = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.action_space_name = ["U", "D", "R", "L"]

        self.reward_matrix = self.get_reward_matrix(step_reward)
        self.value_function = np.zeros((self.ROWS, self.COLS))
        self.policy = np.random.choice([0, 1, 2, 3], size=(self.ROWS, self.COLS))  # Initialize with random policy

    def get_reward_matrix(self, step_reward):
        reward_matrix = np.full((self.ROWS, self.COLS), step_reward)
        goal_loc = self.objects["goal"]["location"]
        obstacle_locs = self.objects["obstacle"]["location"]

        reward_matrix[goal_loc[0], goal_loc[1]] = self.objects["goal"]["reward"]
        for obstacle_loc in obstacle_locs:
            reward_matrix[obstacle_loc[0], obstacle_loc[1]] = self.objects["obstacle"]["reward"]

        return reward_matrix

    def check_state_valid(self, state):
        row, col = state
        return 0 <= row < self.ROWS and 0 <= col < self.COLS

    def policy_evaluation(self):
        while True:
            delta = 0
            for row in range(self.ROWS):
                for col in range(self.COLS):
                    present_state = (row, col)

                    if present_state == self.objects["goal"]["location"] or present_state in self.objects["obstacle"]["location"]:
                        self.value_function[row][col] = 0
                    else:
                        old_value = self.value_function[row][col]
                        action = self.policy[row][col]
                        new_state = (present_state[0] + self.action_space[action][0], present_state[1] + self.action_space[action][1])

                        if self.check_state_valid(new_state):
                            new_value = self.reward_matrix[new_state[0]][new_state[1]] + self.gamma * self.value_function[new_state[0]][new_state[1]]
                            self.value_function[row][col] = new_value
                            delta = max(delta, abs(old_value - new_value))

            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for row in range(self.ROWS):
            for col in range(self.COLS):
                present_state = (row, col)

                if present_state == self.objects["goal"]["location"] or present_state == self.objects["obstacle"]["location"]:
                    continue

                old_action = self.policy[row][col]
                max_value = float("-inf")
                best_action = None

                for action in range(len(self.action_space)):
                    new_state = (present_state[0] + self.action_space[action][0], present_state[1] + self.action_space[action][1])

                    if self.check_state_valid(new_state):
                        new_value = self.reward_matrix[new_state[0]][new_state[1]] + self.gamma * self.value_function[new_state[0]][new_state[1]]
                        if new_value > max_value:
                            max_value = new_value
                            best_action = action

                self.policy[row][col] = best_action

                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()

            if policy_stable:
                break
    
    def plot_value_function(self):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.value_function, cmap = 'Greens_r', interpolation='nearest')

        for i in range(self.ROWS):
            for j in range(self.COLS):
                point = (i, j)
                matrix_value = self.value_function[i, j]

                if point == items["goal"]['location']:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='#0066cc')) #, edgecolor='black'
                    ax.text(j, i, 'G', color='black', ha='center', va='center')
                elif point in items["obstacle"]['location']:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='orange')) #, edgecolor='black'
                    ax.text(j, i, 'O', color='black', ha='center', va='center')
                else:
                    ax.text(j, i, f'{matrix_value:.2f}', color='black', ha='center', va='center')

        ax.axis('off')
        plt.show()

    def plot_policy(self):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.value_function, cmap = 'Greens_r', interpolation='nearest')

        for i in range(self.ROWS):
            for j in range(self.COLS):
                point = (i, j)
                step = self.action_space_name[self.policy[i][j]]

                if point == items["goal"]['location']:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='#0066cc')) #, edgecolor='black'
                    ax.text(j, i, 'G', color='black', ha='center', va='center')
                elif point in items["obstacle"]['location']:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='orange')) #, edgecolor='black'
                    ax.text(j, i, 'O', color='black', ha='center', va='center')
                else:
                    ax.text(j, i, step, color='black', ha='center', va='center')

        ax.axis('off')
        plt.show()       

if __name__ == "__main__":
    items = {'obstacle': {'reward': -10, 'location': {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
                                                      (1, 0), (1, 7), 
                                                      (2, 2), (2, 3), (2, 5), (2, 7),
                                                      (3, 0), (3, 3), (3, 4), (3, 7), 
                                                      (4, 0), (4, 1), (4, 4), (4, 6), (4, 7), 
                                                      (5, 0), (5, 2), (5, 4), (5, 7),
                                                      (6, 0), (6, 5),
                                                      (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)}},
             'goal': {'reward': 10, 'location': (6, 7)}}

    grid_size = (8, 8)

    # items = {'obstacle': {'reward': -1  , 'location': {(0, 0)}},
    #          'goal': {'reward': 0, 'location': (8, 8)}}

    # grid_size = (9, 9)

    grid_world = GridWorld(grid_size, objects=items)
    grid_world.policy_iteration()

    grid_world.plot_policy()
    # grid_world.plot_value_function()

