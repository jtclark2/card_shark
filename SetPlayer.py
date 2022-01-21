from Card import *

class SetPlayer:
    def find_sets(self, cards):
        """
        Update this when the improved version(s) work
        :param cards: all cards being evaluated
        :return:
            List of sets of 3 cards.
        """
        return self._find_sets_brute_force(cards)

    def _find_sets_brute_force(self, cards):
        """
        Start brute force: This algorithm is very inefficient...but it still takes almost no time compared to the
        computer vision processing.
        """
        i = 0
        Sets = []
        for idx1, card1 in enumerate(cards, start=0):
            for idx2, card2 in enumerate(cards[idx1 + 1:], start=idx1 + 1):
                for idx3, card3 in enumerate(cards[idx2 + 1:], start=idx2 + 1):
                    i += 1
                    if (self._check_set(card1, card2, card3)):
                        Sets.append(set([card1, card2, card3]))
                        valid = True
                    else:
                        valid = False
        return Sets

    def _find_sets_efficient(self):
        """
        - for this description, I'll use capital 'Set' to indicate a winning combination of cards, and lowercase 'set'
            just means the data structure
        - We have 4 properties: color, shape, number, and fill
        - Each property has 3 possible values.

        - Start by grouping cards by the first property (which we will arbitrarily choose as color)
            - Now we have 3 groups
                - for each group, 'Set' candidates include all permutations of three cards, representing 3 values being
                 the same
                    - on average, each group will have 4 cards, producing 4 potential sets * 3 groups = 12 possibilities
                - across groups, 'Set' candidates include all permutations with 1 card in each group, representing all
                  three values being different
                    - on average each of 3 groups will have 4 cards, so we would need to check 4**3=64 possibilities
                        - I think we might also be able to do a similar trick to the "2 sum problem", and using
                          the first 2 cards, figure out what the third must be, and check if it's in the set of the
                          third group
                - On average (more precisely the mode, not the mean, but we're keeping it simple for now), we'll have
                  around 12 + 64 = 72 possibilities ... Is there a way to do better by subgrouping in another way before
                  checking (this is plenty fast enough already, but I enjoy this game, and my clock speed is a bit slower)
        """

    def _check_set_efficient(self):
        """
        - We need to check that each property meets 1 of the following conditions:
            - all same: 3 different values in the set
            - all different: 1 unique values in the set
            - For any 3 cards in a potential set, we can just checking how many unique elements are in a set formed
              by a particular property.
                - if len(set(potential_triplet)) != 2: it's a set
        """
        raise NotImplementedError

    def _check_set(self, card1, card2, card3):

        # This conditional is intentionally awkward to intuit (that's why the game is fun)
        attributes = ['shape', 'color', 'count', 'fill']

        # Any unidentified attribute is assumed to be false
        if( None in [getattr(card1, attribute) for attribute in attributes] ):
            return False

        # Check for set (each card must match or all 3 must be unique). Check each attribute independently
        for attribute in attributes:
            all_same_condition =   (getattr(card1, attribute) == getattr(card2, attribute) and
                                    getattr(card1, attribute) == getattr(card3, attribute) )
            all_different_condition =  (getattr(card1, attribute) != getattr(card2, attribute) and
                                        getattr(card1, attribute) != getattr(card3, attribute) and
                                        getattr(card2, attribute) != getattr(card3, attribute) )
            if(all_same_condition or all_different_condition):
                pass    #need all attributes to be True for a set
            else:
                return False

        #All attributes are True
        return True

# TODO: Consider a better place for this function, but this will do for now
def deal(n):
    """
    Deals n cards from a set deck, without repeats.
    :return:
        n set cards, or None, if n is not valid
    """
    # Create cards
    if 0 > n > 81:
        raise Exception(f"{n} is not a valid argument for deal(n). Select 0 > n > 81.")

    deck = []
    for shape in Shape:
        for color in Color:
            for count in Count:
                for fill in Fill:
                    # TODO: might want to move image to the end of the params
                    deck.append(Card(None, shape, color, count, fill))
    random.shuffle(deck)
    return deck[0:n]


if __name__ == "__main__":
    n = 12
    cards = deal(n)
    assert(len(cards) == n) # n cards dealt
    assert(len(set(cards)) == n) # no repeats

    # This isn't much of a unit test, but it's a good sanity check
    player = SetPlayer()
    sets = player.find_sets(cards)
    for s in sets:
        print(s)

    # this is purely because I'm curious about the odds
    set_histogram = {i:0 for i in range(20)}
    n = 15
    for i in range(10000):
        cards = deal(n)
        sets = player.find_sets(cards)
        sets_discovered = 0
        for s in sets:
            sets_discovered += 1
        set_histogram[sets_discovered] += 1
    for num_sets, num_games in set_histogram.items():
        print(f"Found {num_sets} in {num_games} games.")