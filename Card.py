from enum import Enum


class Shape(Enum):
    diamond = 'diamond'
    stadium = 'stadium' # I looked it up, and it's a a proper name (also discorectanble, and obround)
    wisp    = 'wisp'

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"


class Color(Enum):
    red     = 'red'
    green   = 'green'
    purple  = 'purple'

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"


class Count(Enum):
    one     = 'one'
    two     = 'two'
    three   = 'three'

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"


class Fill(Enum):
    solid   = 'solid'
    striped = 'striped'
    empty   = 'empty'

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[5:]+"-"


class Card:
    def __init__(self, image=None, shape=None, color=None, count=None, fill=None, index=None):
        self.image = image
        self.shape = shape
        self.color = color
        self.count = count
        self.fill = fill
        self.index = index

    def __repr__(self):
        return (repr(self.shape) + repr(self.color) + repr(self.count) + repr(self.fill))


if __name__ == "__main__":
    myCard = Card()
