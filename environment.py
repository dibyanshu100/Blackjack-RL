import random
from typing import Optional
from utils import BlackjackUtils
from Deck import Deck

class BlackjackEnv:

    def __init__(self):
        """Initialisation of the BlackjackEnv class (constructor)."""
        self.action_space = [0,1,2,3,4]
        self.actionstaken = 0
        self.insurance_bet = 0

    def step(self, action):

        assert action in self.action_space

        # Force stand if the sum is 21
        if BlackjackUtils.calculate_hand_sum(self.player) == 21:
            action = 0

        """Perform a learning step."""
        if action == 0:  # stick
            terminated = True
            while BlackjackUtils.calculate_hand_sum(self.dealer) < 17:
                self.dealer.append(BlackjackUtils.draw_random_card())
            reward = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(self.player), BlackjackUtils.calculate_score(self.dealer))
            if BlackjackUtils.check_natural_blackjack(self.player) and not BlackjackUtils.check_natural_blackjack(self.dealer):
                reward = 1.0
            if self.insurance_bet > 0:
                if BlackjackUtils.check_natural_blackjack(self.dealer):
                    reward += 2 * self.insurance_bet
                else:
                    reward -= self.insurance_bet
            self.actionstaken += 1

        elif action == 1:  # hit
            self.player.append(BlackjackUtils.draw_random_card())
            #print(self.player)
            if BlackjackUtils.check_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
            self.actionstaken += 1

        elif action == 2:  # double down
            self.player.append(BlackjackUtils.draw_random_card())
            if BlackjackUtils.check_bust(self.player):
                terminated = True
                reward = -2.0
            else:
                terminated = True
                while BlackjackUtils.calculate_hand_sum(self.dealer) < 17:
                    self.dealer.append(BlackjackUtils.draw_random_card())
                reward = 2.0 * BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(self.player), BlackjackUtils.calculate_score(self.dealer))
            self.actionstaken += 1


        elif action == 3:  # split
            if BlackjackUtils.can_split(self.player, self.actionstaken):
                card1 = self.player[0]
                card2 = self.player[1]

                # Splitting the hand
                hand1 = [card1, BlackjackUtils.draw_random_card()]
                hand2 = [card2, BlackjackUtils.draw_random_card()]

                if card1 == 1:  # Ace splitting rule
                    hand1 = [card1, BlackjackUtils.draw_random_card()]
                    hand2 = [card2, BlackjackUtils.draw_random_card()]
                    terminated = True
                    reward1 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand1),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward2 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand2),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2
                else:
                    # Play hand 1
                    while not BlackjackUtils.check_bust(hand1) and not BlackjackUtils.check_natural_blackjack(hand1):
                        hand1.append(BlackjackUtils.draw_random_card())
                    # Play hand 2
                    while not BlackjackUtils.check_bust(hand2) and not BlackjackUtils.check_natural_blackjack(hand2):
                        hand2.append(BlackjackUtils.draw_random_card())
                    terminated = True
                    reward1 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand1),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward2 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand2),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2

            else:
                terminated = False
                reward = 0.0
            self.actionstaken += 1


        elif action == 4:  # insurance
            if self.dealer[0] == 1:  # If dealer's face-up card is an Ace
                self.insurance_bet = 0.5  # Assume half of the initial bet as insurance
                reward = 0.0  # No immediate reward, calculated later based on dealer's hand
                terminated = False
            else:
                reward = 0.0
                terminated = False

        #print(f"Reward = {reward}")
        #print(f"Terminated = {terminated}")
        return self._get_obs(), reward, terminated

    def _get_obs(self):
        """Get the current observations."""
        return (
            BlackjackUtils.calculate_hand_sum(self.player), self.dealer[0], BlackjackUtils.has_usable_ace(self.player),
            BlackjackUtils.can_double_down(self.player, self.actionstaken),
            BlackjackUtils.can_split(self.player, self.actionstaken),
            BlackjackUtils.can_insurance(self.dealer[0], self.actionstaken)
        )

    def reset(self, seed: Optional[int] = None):
        """Reset the environment."""

        if seed is not None:
            random.seed(seed)
        self.dealer = BlackjackUtils.draw_initial_hand()    # eg. [10,2]
        self.player = BlackjackUtils.draw_initial_hand()    # eg. [3, 8]
        self.actionstaken = 0
        self.insurance_bet = 0
        return self._get_obs()




## BlackJack Environment with card counting
class BlackjackEnv_CardCounting:

    def __init__(self, number_of_decks, cc_technique):
        """Initialisation of the BlackjackEnv class (constructor)."""
        self.action_space = [0,1,2,3,4]
        self.actionstaken = 0
        self.insurance_bet = 0
        self.deck = Deck(seed=0, number_of_decks=number_of_decks, technique = cc_technique)

    def step(self, action):

        assert action in self.action_space

        # Force stand if the sum is 21
        if BlackjackUtils.calculate_hand_sum(self.player) == 21:
            action = 0

        """Perform a learning step."""
        if action == 0:  # stick
            terminated = True
            while BlackjackUtils.calculate_hand_sum(self.dealer) < 17:
                self.dealer.append(self.deck.draw_card())
            reward = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(self.player), BlackjackUtils.calculate_score(self.dealer))
            if BlackjackUtils.check_natural_blackjack(self.player) and not BlackjackUtils.check_natural_blackjack(self.dealer):
                reward = 1.0
            if self.insurance_bet > 0:
                if BlackjackUtils.check_natural_blackjack(self.dealer):
                    reward += 2 * self.insurance_bet
                else:
                    reward -= self.insurance_bet
            self.actionstaken += 1

        elif action == 1:  # hit
            self.player.append(self.deck.draw_card())
            #print(self.player)
            if BlackjackUtils.check_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
            self.actionstaken += 1

        elif action == 2:  # double down
            self.player.append(self.deck.draw_card())
            if BlackjackUtils.check_bust(self.player):
                terminated = True
                reward = -2.0
            else:
                terminated = True
                while BlackjackUtils.calculate_hand_sum(self.dealer) < 17:
                    self.dealer.append(self.deck.draw_card())
                reward = 2.0 * BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(self.player), BlackjackUtils.calculate_score(self.dealer))
            self.actionstaken += 1


        elif action == 3:  # split
            if BlackjackUtils.can_split(self.player, self.actionstaken):
                card1 = self.player[0]
                card2 = self.player[1]

                # Splitting the hand
                hand1 = [card1, self.deck.draw_card()]
                hand2 = [card2, self.deck.draw_card()]

                if card1 == 1:  # Ace splitting rule
                    hand1 = [card1, self.deck.draw_card()]
                    hand2 = [card2, self.deck.draw_card()]
                    terminated = True
                    reward1 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand1),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward2 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand2),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2
                else:
                    # Play hand 1
                    while not BlackjackUtils.check_bust(hand1) and not BlackjackUtils.check_natural_blackjack(hand1):
                        hand1.append(self.deck.draw_card())
                    # Play hand 2
                    while not BlackjackUtils.check_bust(hand2) and not BlackjackUtils.check_natural_blackjack(hand2):
                        hand2.append(self.deck.draw_card())
                    terminated = True
                    reward1 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand1),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward2 = BlackjackUtils.compare_scores(BlackjackUtils.calculate_score(hand2),
                                                            BlackjackUtils.calculate_score(self.dealer))
                    reward = (reward1 + reward2) / 2

            else:
                terminated = False
                reward = 0.0
            self.actionstaken += 1


        elif action == 4:  # insurance
            if self.dealer[0] == 1:  # If dealer's face-up card is an Ace
                self.insurance_bet = 0.5  # Assume half of the initial bet as insurance
                reward = 0.0  # No immediate reward, calculated later based on dealer's hand
                terminated = False
            else:
                reward = 0.0
                terminated = False

        #print(f"Reward = {reward}")
        #print(f"Terminated = {terminated}")
        return self._get_obs(), reward, terminated

    def _get_obs(self):
        """Get the current observations."""
        return (
            BlackjackUtils.calculate_hand_sum(self.player), self.dealer[0], BlackjackUtils.has_usable_ace(self.player),
            BlackjackUtils.can_double_down(self.player, self.actionstaken),
            BlackjackUtils.can_split(self.player, self.actionstaken),
            BlackjackUtils.can_insurance(self.dealer[0], self.actionstaken),
            self.deck.total_points,
            self.deck.unseen_cards
        )

    def reset(self, seed: Optional[int] = None):
        """Reset the environment."""

        if seed is not None:
            random.seed(seed)

        self.deck.init_deck()
        self.dealer = self.deck.draw_hand()
        self.player = self.deck.draw_hand()
        self.actionstaken = 0
        self.insurance_bet = 0

        return self._get_obs()





