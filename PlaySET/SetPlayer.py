from Card import *

class SetPlayer:
    def find_sets(self, cards):
        # Start brute force (this game is intentionally computationally complex...but for humans, not computers)
        # Todo: Future optimization-we can get a bit clever with our patterns if this proves computationally straining...
        #   Certain cards can be eliminated from the pool, by virtue of having certain attributes, without
        #   having to check all possibilities
        i = 0
        Sets = []
        for idx1, card1 in enumerate(cards, start=0):
            for idx2, card2 in enumerate(cards[idx1 + 1:], start=idx1 + 1):
                for idx3, card3 in enumerate(cards[idx2 + 1:], start=idx2 + 1):
                    i += 1
                    if (self.check_set(card1, card2, card3)):
                        Sets.append(set([card1, card2, card3]))
                        valid = True
                    else:
                        valid = False
        return Sets

    def check_set(self, card1, card2, card3):

        # This conditional is intentionally awkward to intuit (that's why the game is fun)
        attributes = ['shape', 'color', 'count', 'fill']

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
