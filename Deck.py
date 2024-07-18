import random


class Deck():
    """This class provides a deck of cards."""

    def __init__(self, seed=0, number_of_decks=6, low_limit=6, high_limit=10, technique = 'hi_lo') -> None:
        """Initialisation of the Deck class (constructor)."""

        self.random = random.Random(seed)
        self.number_of_decks = number_of_decks
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.technique = technique
        self.init_deck()

    def init_deck(self):
        """This method initialises a deck of cards."""

        # 1 = Ace, 2-10 = Number cards, Jack / Queen / King = 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * self.number_of_decks
        self.random.shuffle(self.deck)
        self.unseen_cards = len(self.deck)
        self.total_points = 0
        self.card_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

    def draw_card(self):
        """This methode draws a random card and makes card counting possible."""

        card = self.deck.pop(0)
        self.unseen_cards -= 1

        if self.technique == 'hi_lo':
            if (card >= self.high_limit or card == 1):
                # normally: 10, Jack, Queen, King, Ace
                self.total_points -= 1

            elif (card <= self.low_limit):
                # normally: 2 - 6
                self.total_points += 1

        if self.technique == 'omega_2':
            if card in [2, 3, 7]:
                self.total_points += 1
            elif card in [4, 5, 6]:
                self.total_points += 2
            elif card == 9:
                self.total_points -= 1
            elif card in [10]:  # 10, Jack, Queen, King
                self.total_points -= 2

        if self.technique == 'zen_count':
            if card in [2, 3, 7]:
                self.total_points += 1
            elif card in [4, 5, 6]:
                self.total_points += 2
            elif card in [10]:  # 10, Jack, Queen, King
                self.total_points -= 2
            elif card == 1:  # Ace
                self.total_points -= 1


        self.card_counter[card] += 1

        return card

    def draw_hand(self):
        """This methde makes a hand from two random cards."""

        return [self.draw_card(), self.draw_card()]

