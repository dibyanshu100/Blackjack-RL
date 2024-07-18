import numpy as np
from environment import BlackjackEnv, BlackjackEnv_CardCounting
from algorithms import Q_learning, SARSA, MonteCarlo
import logging
import matplotlib.pyplot as plt


## Functions to run Q Learning
def run_episodes_Qlearning(agent, observation, actions_count, episodes, cc_flag=False):
    rewards_episodes = np.zeros(episodes)
    i = 0
    while i < episodes:
        round_rewards = 0  # Reset round_rewards for each new episode
        while True:  # Continue until the episode is done
            action = agent.choose_action(observation, cc_flag)
            next_observation, reward, is_done = env.step(action)
            actions_count[action] += 1
            round_rewards += reward
            agent.learn(observation, action, reward, next_observation, cc_flag)
            #logging.info(f'Episode: {i} || Observation: {observation} || Action: {action} || Reward: {reward} || Q: {agent.Q[observation]}')
            observation = next_observation
            if is_done:
                rewards_episodes[i] = round_rewards
                observation = env.reset()
                i += 1
                break  # Break the while loop and start a new episode

    return agent, actions_count, rewards_episodes



def train_Q(env, epochs, episodes, epsilon, learning_rate, gamma, cc_flag):
    """This function starts the training and evaluation with the Q-Learning method.
        Input:
        env: Blackjack Environment
        epochs: Number of epochs to be trained
        episodes: Number of players
    """

    episodes = episodes  # Reward calculated over every episode
    re = []
    avg_rewards = np.zeros(episodes)
    dic_actions_count = {}
    dic_actions = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT", 4: "INSURANCE"}
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Number of actions performed in each category

    agent = Q_learning(env, epsilon, learning_rate, gamma)

    print("Start learning with Q-Learning and the calculation of the profit or loss ...")
    for epc in range(epochs):
        if (epc % 100 == 0):
            print(f"------------------Epoch: {epc}")
        observation = env.reset()
        agent, actions_count, rewards_episodes = run_episodes_Qlearning(agent, observation, actions_count, episodes, cc_flag)
        avg_rewards += (rewards_episodes - avg_rewards) / (epc + 1)
        re.append(np.sum(avg_rewards)/episodes)
        #logging.info(f'\n\n******End of an epoch : Average rewards = {avg_rewards} \n\n')
        logging.info(f'Total average: {np.sum(avg_rewards)/episodes}')

    # Create an understandable dictionary
    for key, value in dic_actions.items():
        dic_actions_count.update({value: actions_count[key]})


    return agent, dic_actions_count, re


## Functions to run SARSA
def run_episodes_SARSA(agent, observation, actions_count, episodes, cc_flag=False):
    rewards_episodes = np.zeros(episodes)
    i = 0
    while i < episodes:
        round_rewards = 0  # Reset round_rewards for each new episode
        action = agent.choose_action(observation, cc_flag)  # Choose the first action
        while True:  # Continue until the episode is done
            next_observation, reward, is_done = env.step(action)
            actions_count[action] += 1
            round_rewards += reward
            next_action = agent.choose_action(next_observation, cc_flag)  # Choose the next action
            agent.learn(observation, action, reward, next_observation, next_action, cc_flag)
            observation = next_observation
            action = next_action
            if is_done:
                rewards_episodes[i] = round_rewards
                observation = env.reset()
                i += 1
                break  # Break the while loop and start a new episode

    return agent, actions_count, rewards_episodes



def train_SARSA(env, epochs, episodes, epsilon, learning_rate, gamma, cc_flag):
    """This function starts the training and evaluation with the SARSA method.
       Input:
       env: Blackjack Environment
       epochs: Number of epochs to be trained
       episodes: Number of players
    """
    re = []
    avg_rewards = np.zeros(episodes)
    dic_actions_count = {}
    dic_actions = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT", 4: "INSURANCE"}
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Number of actions performed in each category

    agent = SARSA(env, epsilon, learning_rate, gamma)

    print("Start learning with SARSA and the calculation of the profit or loss ...")
    for epc in range(epochs):
        if (epc % 100 == 0):
            print(f"------------------Epoch: {epc}")
        observation = env.reset()
        agent, actions_count, rewards_episodes = run_episodes_SARSA(agent, observation, actions_count, episodes, cc_flag)
        avg_rewards += (rewards_episodes - avg_rewards) / (epc + 1)
        re.append(np.sum(avg_rewards) / episodes)
        logging.info(f'Total average: {np.sum(avg_rewards) / episodes}')

    # Create an understandable dictionary
    for key, value in dic_actions.items():
        dic_actions_count.update({value: actions_count[key]})

    return agent, dic_actions_count, re


## Functions to run Monte Carlo (On Policy)
def run_episodes_MC(agent, observation, actions_count, episodes, cc_flag):
    rewards_episodes = np.zeros(episodes)
    i = 0
    while i < episodes:
        episode = []
        round_rewards = 0  # Reset round_rewards for each new episode

        while True:  # Continue until the episode is done
            action = agent.choose_action(observation, cc_flag)
            next_observation, reward, is_done = env.step(action)
            actions_count[action] += 1
            round_rewards += reward

            episode.append((observation, action, reward))
            observation = next_observation

            if is_done:
                agent.learn(episode)
                rewards_episodes[i] = round_rewards
                observation = agent.env.reset()
                i += 1
                break  # Break the while loop and start a new episode

    return agent, actions_count, rewards_episodes

def train_MonteCarlo(env, epochs, episodes, epsilon, gamma, cc_flag):
    """
    This function starts the training and evaluation with the Monte Carlo on-policy method.
    """
    re = []
    avg_rewards = np.zeros(episodes)
    dic_actions_count = {}
    dic_actions = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT", 4: "INSURANCE"}
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Number of actions performed in each category

    agent = MonteCarlo(env, epsilon, gamma)

    print("Start learning with Monte Carlo on-policy and the calculation of the profit or loss ...")
    for epc in range(epochs):
        if (epc % 100 == 0):
            print(f"------------------Epoch: {epc}")
        observation = env.reset()
        agent, actions_count, rewards_episodes = run_episodes_MC(agent, observation, actions_count, episodes, cc_flag)
        avg_rewards += (rewards_episodes - avg_rewards) / (epc + 1)
        re.append(np.sum(avg_rewards) / episodes)
        #logging.info(f'Total average: {np.sum(avg_rewards) / episodes}')

    # Create an understandable dictionary
    for key, value in dic_actions.items():
        dic_actions_count.update({value: actions_count[key]})

    return agent, dic_actions_count, re


if (__name__ == "__main__"):
    np.random.seed(0)
    logging.basicConfig(filename='train.log', level=logging.INFO, format='%(levelname)s - %(message)s', filemode='w')

    epochs = 1000
    episodes = 100

    # Select Environment
    #env = BlackjackEnv()

    #with card counting technique: 'hi-lo' or 'omega_2' or 'zen_count'
    cc_technique = 'zen_count'
    env = BlackjackEnv_CardCounting( number_of_decks = 1, cc_technique=cc_technique)
    cc_flag = True

    epsilon = [0.2]
    learning_rate = [0.1]
    gamma = 0.5

    QL = True
    Sarsa = True
    MC = True

    plot_learning = True
    plot_action = True

    results_QL = {}
    if QL:
        for ep in epsilon:
            results_QL[ep] = {}
            act_counts = {}
            for lr in learning_rate:
                agent, dic_actions_count_QL, avg_rewards = train_Q(env, epochs, episodes, ep, lr, gamma, cc_flag)
                results_QL[ep][lr] = avg_rewards


    # SARSA
    results_SARSA = {}
    if Sarsa:
        for ep in epsilon:
            results_SARSA[ep] = {}
            act_counts = {}
            for lr in learning_rate:
                agent, dic_actions_count_SARSA, avg_rewards = train_SARSA(env, epochs, episodes, ep, lr, gamma, cc_flag)
                results_SARSA[ep][lr] = avg_rewards


    # Monte Carlo
    results_MC = {}
    if MC:
        i=0
        for ep in epsilon:
            agent, dic_actions_count_MC, avg_rewards = train_MonteCarlo(env, epochs, episodes, ep, gamma, cc_flag)
            results_MC[i] = avg_rewards
            i += 1

    ### Plotting for all three Agents
    if QL and SARSA and MC and plot_learning:
        plt.figure(figsize=(11, 7))
        plt.plot(results_QL[epsilon[0]][learning_rate[0]], label=f'Q_Learning')
        plt.plot(results_SARSA[epsilon[0]][learning_rate[0]], label=f'SARSA')
        plt.plot(results_MC[0], label=f'Monte Carlo')
        plt.xlabel("Number of epochs", fontsize=17)
        plt.ylabel(f"Average Reward for {episodes} episodes", fontsize=17)
        plt.title(f"Rewards when epsilon = {epsilon[0]}, gamma = {gamma}, card-counting {cc_technique} and 2x rewards on Natural Blackjack", fontsize=12)#, card-counting {cc_technique}, 2x rewards on Natural Blackjack")
        plt.grid()
        plt.legend(fontsize = 14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        #plt.ylim(bottom=-0.5, top=0.1)
        plt.savefig(f"Rewards when epsilon = {epsilon[0]}, gamma = {gamma}, card-counting {cc_technique}  and 2x rewards on Natural Blackjack.png")#, gamma = {gamma}, card-counting {cc_technique}, 2x rewards on Natural Blackjack.png")
        plt.show()


    ### Plotting action taken
    if QL and SARSA and MC and plot_action:
        ## Actions Count
        print(f"QL actions count: {dic_actions_count_QL}")
        print(f"SARSA actions count: {dic_actions_count_SARSA}")
        print(f"MC actions count: {dic_actions_count_MC}")

        def calculate_percentages(action_counts):
            total = sum(action_counts.values())
            return {action: (count / total) * 100 for action, count in action_counts.items()}


        # Calculate percentages for QL
        total_QL = sum(dic_actions_count_QL.values())
        percentages_QL = {action: (count / total_QL) * 100 for action, count in dic_actions_count_QL.items()}

        # Calculate percentages for SARSA
        total_SARSA = sum(dic_actions_count_SARSA.values())
        percentages_SARSA = {action: (count / total_SARSA) * 100 for action, count in dic_actions_count_SARSA.items()}

        # Calculate percentages for MC
        total_MC = sum(dic_actions_count_MC.values())
        percentages_MC = {action: (count / total_MC) * 100 for action, count in dic_actions_count_MC.items()}

        # Actions
        actions = list(dic_actions_count_QL.keys())

        # Percentages for plotting
        percentages_QL_list = [round(percentages_QL[action], 1) for action in actions]
        percentages_SARSA_list = [round(percentages_SARSA[action], 1) for action in actions]
        percentages_MC_list = [round(percentages_MC[action], 1) for action in actions]

        # Plotting
        x = np.arange(len(actions))  # Label locations
        width = 0.2  # Width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, percentages_QL_list, width, label='QL')
        bars2 = ax.bar(x, percentages_SARSA_list, width, label='SARSA')
        bars3 = ax.bar(x + width, percentages_MC_list, width, label='MC')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Actions',fontsize=17)
        ax.set_ylabel('Percentage',fontsize=17)
        ax.set_title('Percentage of Each Action when 2x reward to player on Natural Blackjack',fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(actions)
        ax.legend()

        # Add percentages on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=45)

        for bar in bars3:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=45)

        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig('Percentage of Each Action taken when 2x reward to player on Natural Blackjack.png')
        plt.show()




