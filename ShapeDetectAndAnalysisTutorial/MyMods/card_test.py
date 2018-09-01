from enum import Enum

class Shape(Enum):
    diamond = 1
    stadium = 2 # I looked it up, and it's a a proper name (also discorectanble, and obround
    wisp    = 3

class Color(Enum):
    red     = 1
    green   = 2
    purple  = 3

class Count(Enum):
    one     = 1
    two     = 2
    three   = 3

class Fill(Enum):
    solid   = 1
    striped = 2
    empty   = 3

class Card:
    def __init__(self, shape, color, count, fill):
        self.shape = shape
        self.color = color
        self.count = count
        self.fill = fill

    def __repr__(self):
        return ( repr(self.shape) + repr(self.color) + repr(self.count) + repr(self.fill) + '\n')

    
def check_set(card1, card2, card3):

    # This conditional is intentionally awkward to intuit (that's why the game is fun)
    attributes = ['shape', 'color', 'count', 'fill']

    for attribute in attributes:
        all_same_condition =   (getattr(card1, attribute) == getattr(card2, attribute) and
                                getattr(card1, attribute) == getattr(card3, attribute) )
        all_different_condition =  (getattr(card1, attribute) != getattr(card2, attribute) and
                                    getattr(card1, attribute) != getattr(card3, attribute) and
                                    getattr(card2, attribute) != getattr(card3, attribute) )
        #print('Attribute: %s\t\tSame: %r \t\t Diff: %r' % (attribute, all_same_condition, all_different_condition) )
        #print("1: %r \t2: %r \t3: \t%r" % (getattr(card1, attribute), getattr(card2, attribute), getattr(card3, attribute)) )
        if(all_same_condition or all_different_condition):
            pass    #need all attributes to be True for a set
        else:
            return False

    #All attributes are True
    return True



card1 = Card(Shape.wisp,     Color.purple, Count.two,   Fill.striped )
card2 = Card(Shape.stadium,  Color.red,    Count.two,   Fill.empty )
card3 = Card(Shape.wisp,     Color.red,    Count.one,   Fill.empty )
card4 = Card(Shape.wisp,     Color.green,  Count.three, Fill.striped )
card5 = Card(Shape.diamond,  Color.red,    Count.two,   Fill.striped )
card6 = Card(Shape.wisp,     Color.purple, Count.one,   Fill.empty )
card7 = Card(Shape.stadium,  Color.green,  Count.two,   Fill.solid )
card8 = Card(Shape.diamond,  Color.purple, Count.three, Fill.solid )
card9 = Card(Shape.wisp,     Color.green,  Count.one,   Fill.empty )
card10 = Card(Shape.diamond, Color.green,  Count.one,   Fill.empty )
card11 = Card(Shape.stadium, Color.purple, Count.one,   Fill.empty )
card12 = Card(Shape.wisp,    Color.red,    Count.two,   Fill.solid )

cards = [card1, card2, card3, card4, card5, card6, card7, card8, card9, card10, card11, card12]

i=0
sets_found = 0
for idx1, card1 in enumerate(cards, start=0):
    for idx2, card2 in enumerate(cards[idx1+1:], start=idx1+1):
        for idx3, card3 in enumerate(cards[idx2+1:], start=idx2+1):
            i += 1
            if(check_set(card1, card2, card3)):
                sets_found += 1
                valid = True
                print("%d: %d, %d, %d, %r" % (i, idx1, idx2, idx3, valid) )
            else:
                valid = False
            
print(sets_found)
