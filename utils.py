import random

class BlackjackUtils:
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    @staticmethod
    def compare_scores(player_score, dealer_score):
        """Compare the player's score with the dealer's score. Returns 1.0 if player wins else -1.0 """
        return float(player_score > dealer_score) - float(player_score < dealer_score)

    @staticmethod
    def draw_random_card():
        """Draw a random card from the deck."""
        return random.choice(BlackjackUtils.deck)

    @staticmethod
    def draw_initial_hand():
        """Draw an initial hand consisting of two random cards."""
        return [BlackjackUtils.draw_random_card(), BlackjackUtils.draw_random_card()]

    @staticmethod
    def has_usable_ace(hand):
        """Check if the hand contains a usable ace."""
        return 1 in hand and sum(hand) + 10 <= 21

    @staticmethod
    def calculate_hand_sum(hand):
        """Calculate the sum of the hand, considering usable aces."""
        if BlackjackUtils.has_usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    @staticmethod
    def check_bust(hand):
        """Check if the hand is a bust (sum exceeds 21)."""
        return BlackjackUtils.calculate_hand_sum(hand) > 21

    @staticmethod
    def calculate_score(hand):
        """Calculate the score of the hand, with zero indicating a bust."""
        return 0 if BlackjackUtils.check_bust(hand) else BlackjackUtils.calculate_hand_sum(hand)

    @staticmethod
    def check_natural_blackjack(hand):
        """Check if the hand is a natural blackjack."""
        return sorted(hand) == [1, 10]

    @staticmethod
    def can_double_down(hand, actions_taken):
        """Check if double down can be played."""
        return len(hand) == 2 and actions_taken == 0

    @staticmethod
    def can_split(hand, actions_taken):
        """Check if splitting the hand is possible."""
        return len(hand) == 2 and actions_taken == 0 and hand[0] == hand[1]

    @staticmethod
    def can_insurance(dealer_up_card, actions_taken):
        """Returns true if insurance is possible. (dealers up card =1)"""
        return dealer_up_card == 1 and actions_taken == 0



# obj = BlackjackUtils()
# x = obj.draw_initial_hand()
# print(x)

