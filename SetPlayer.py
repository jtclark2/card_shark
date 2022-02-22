from Card import *
import random

class SetPlayer:
    def find_sets(self, cards):
        """
        Update this when the improved version(s) work
        :param cards: all cards being evaluated
        :return:
            List of sets of 3 cards.
        """
        return self._find_sets_efficient(cards)

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
                    potential_set =  set([card1, card2, card3])
                    if (self._check_set(potential_set)):
                        Sets.append(potential_set)
                        valid = True
                    else:
                        valid = False
        return Sets

    def find_third_card(self, card1, card2):
        """
        ### Untested ###

        Any pair of cards forces the identity of the last card...find it!
        :param card1: A card in the set
        :param card2: A card in the set
        :return: The missing card to complete the set
        """
        attributes = ['shape', 'color', 'count', 'fill']
        enums = [Shape, Color, Count, Fill]

        final_card = Card()
        for attribute, EnumObj in zip(attributes, enums):
            if getattr(card1, attribute) == getattr(card2, attribute):
                value = getattr(card1, attribute)
            else:
                values = [val for val in EnumObj]
                values.remove(getattr(card1, attribute))
                values.remove(getattr(card2, attribute))
                value = values[0]
            setattr(final_card, attribute, value)

        return final_card

    def combinations_of_three(self, cards):
        """
        Purpose: Find all combinations of 3 items in a list.
        :param cards: A list
        :return: list(set([item1, item2, item3])): A list of sets of 3 cards, exploring all permutations of the "cards" input
        """
        combinations = []
        length = len(cards)
        for i in range(length-2):
            for j in range(i+1,length-1):
                for k in range(j+1,length):
                    combinations.append(set([cards[i], cards[j], cards[k]]))
        return combinations

    def _find_sets_efficient(self, cards):
        """
        Purpose: Find all the sets.

        Inputs:
            -  cards: list(Card) - A list of Cards.
        Returns:
            - list(set(Card): Any sets found among those cards.

        Definition of a SET:
            Playing the game SET, not to be confused with the data type, any 3 cards can form a set if each of the
            4 properties independantly fulfills one of two conditions:
                1) That property matches for all 3 cards
                2) The properties are all unqique between the 3 cards

        ### How it works:
        - Start by grouping cards by the first property (which we will arbitrarily choose as color)
            - Now we have 3 groups (we'll have to check for: "same" and "different"
                - Same: Just check the remaining properties within this subset
                - Different: All permutations with 1 card from each group
                    - Use the two-sum trick to loop through the first two groups, and infer the third
                - On average (more precisely the mode, not the mean, but we're keeping it simple for now), we'll have
                  around 12 + 64 = 72 possibilities ... Is there a way to do better by subgrouping in another way before
                  checking (this is plenty fast enough already, but I enjoy this game, and my clock speed is a bit slower)


        ### Big O Complexity Analysis in time:
            In practice this take the processing time from ~ 2.5ms to .7ms on my machine. It's tiny compared to the time
            it takes to perform the image process, but every bit helps.

            Just for kicks, we'll do a little time complexity analysis:
            Selecting cards is considered a combination/choose without replacement problem, which is governed by:
            choose(n=number of things to choose from, r=number of things being chosen) = O(n! / (r!(n−r)!))
            Let m be the number of cards showing:
            1) Brute Force Method: We just try all combinations of 3 cards.
                O(n! / (3!*(n-3)!)) = O((n)*(n-1)*(n-2)/6) ~ n^3/6
               - Standard board is 12 cards: 12! / (3!9!) = 220
               - Extended board is 15 cards: 15! / (3!12!) = 455
               - Full Deck:                  81! / (3!*78!) = 85320

            2) Efficient:
                  Top Level grouping into ~n/3 (on average, some will be larger, and some smaller)
                      Same: For each of 3 groups choose(n/3,r) = 3*(n/3)! / (r!((n/3)−r)!)
                      Different: Across 2 of the 3 groups, try all combinations with replacement: n/3*n/3 (and check for answer in dict)
                  O( 3*(n/3)! / (r!(n/3-r!)) + (n/3)^2 )
                  Considering cards chosen(r) is always 3, and letting m = n/3, this reduces to:
                  O( 3*(m)! / (3!(m-3!)) + (m)^2 )
                  = O( 3*m*(m-1)*(m-2)/6 + m^2) )
                  = O(m^3/2+ m^2) = m^3/2
                  = O(n^3/(2*3^3) ~ n^3 / 54
                  Comparing to the brute force solution...we haven't saved much, maybe a constant factor of ~9 .
                  This doesn't matter to the computer, but it might help win a game with your friends :)
               - Standard board is 12 cards: 3 * (12/3)! / (3! * (12/3-3)!) + (12/3)^2 = 3*4! / (3!*1!) + 4^2 = 28
               - Extended board is 15 cards: 3 * (15/3)! / (3! * (15/3-3)!) + (15/3)^2 = 55
               - Full Deck:                  3 * (81/3)! / (3! * (81/3-3)!) + (81/3)^2 = 9504

        """
        # TODO: Should we fix this here, or just in the enumerations?
        if len(cards) != len(set(cards)):
            return [] # a duplicate card implies our identification failed

        red_cards = [card for card in cards if card.color == Color.red]
        green_cards = [card for card in cards if card.color == Color.green]
        purple_cards = [card for card in cards if card.color == Color.purple]

        valid_sets = []

        # same
        for color_matched_cards in [red_cards, green_cards, purple_cards]:
            potential_sets = self.combinations_of_three(color_matched_cards)
            for potential_set in potential_sets:
                if self._check_set(potential_set):
                    valid_sets.append(potential_set)

        # different
        purple_cards = {card: card for card in purple_cards}# set(purple_cards) # index lookup is much faster than looping through a list
        for red_card in red_cards:
            for green_card in green_cards:
                missing_card = self.find_third_card(red_card, green_card)
                if missing_card in purple_cards:
                    missing_card = purple_cards[missing_card] # TODO: INDEX??? for display purposes...use the real card, not the one you create!!!!
                    valid_sets.append(set([red_card, green_card, missing_card]))

        return valid_sets

    def _check_set(self, set_of_cards):
        """
        - Each property needs to meet 1 of the following conditions:
            1) all same: 3 unique values in the set
            2) all different: 1 unique values in the set
            -> 1 & 2 together imply we just need to check len(unique_attributes) == 2
        """
        attributes = ['shape', 'color', 'count', 'fill']

        # Check for set (each card must match or all 3 must be unique). Check each attribute independently
        try:
            card1, card2, card3 = [card for card in set_of_cards]
        except:
            # TODO: This catch only here for testing, after seeing that only 2 values were unpacked instead of 3
            print([card for card in set_of_cards])
            raise
        for attribute in attributes:
            attributes = set([getattr(card1, attribute), getattr(card2, attribute), getattr(card3, attribute)])
            if len(attributes) == 2:
                return False
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

    ### Setup
    n = 12
    cards = deal(n)
    assert(len(cards) == n) # n cards dealt
    assert(len(set(cards)) == n) # no repeats

    # This isn't much of a unit test, but it's a good sanity check
    player = SetPlayer()
    sets = player.find_sets(cards)


    ### Test for set finder (human sanity check - not a unit test)
    for s in sets:
        print(s)


    ### Test combinations_of_three
    combinations = player.combinations_of_three([1,2,3,4,5])
    assert(combinations[0] == set([1,2,3]))
    assert(combinations[1] == set([1,2,4]))
    assert(combinations[2] == set([1,2,5]))
    assert(combinations[3] == set([1,3,4]))
    assert(combinations[4] == set([1,3,5]))
    assert(combinations[5] == set([1,4,5]))
    assert(combinations[6] == set([2,3,4]))
    assert(combinations[7] == set([2,3,5]))
    assert(combinations[8] == set([2,4,5]))
    assert(combinations[9] == set([3,4,5]))

    ### Test _find_set_efficient
    # Brute force works (I've tested it manually quite a bit)...Let's use it to test the faster methods
    n = 12
    for i in range(10):
        cards = deal(n)
        brute = player._find_sets_brute_force(cards)
        efficient = player._find_sets_efficient(cards)

        assert(len(efficient) ==  len(brute))
        for s in efficient:
            if not s in brute:
                print(brute)
                print(efficient)
                raise ValueError(f"Brute and Efficient solutions do not match...\nBrute: {brute}\n Efficient: {efficient}\n")



    import time
    tic = time.perf_counter()
    # this is purely because I'm curious about the odds
    histogram_of_sets = {i:0 for i in range(50)}
    n = 12
    for i in range(1000):
        cards = deal(n)
        sets = player._find_sets_efficient(cards)
        # sets = player._find_sets_brute_force(cards)
        sets_discovered = 0
        for s in sets:
            sets_discovered += 1
        histogram_of_sets[sets_discovered] += 1

    toc = time.perf_counter()
    print(toc-tic)

    patience = 2
    zero_count = 0
    for num_sets, num_games in histogram_of_sets.items():
        print(f"Found {num_sets} in {num_games} games.")

        if num_games == 0:
            zero_count += 1
        else:
            zero_count = 0

        if zero_count >= patience:
            break