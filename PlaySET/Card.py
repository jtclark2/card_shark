from enum import Enum

class Shape(Enum):
    diamond = 1
    stadium = 2 # I looked it up, and it's a a proper name (also discorectanble, and obround)
    wisp    = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"

class Color(Enum):
    red     = 1
    green   = 2
    purple  = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"

class Count(Enum):
    one     = 1
    two     = 2
    three   = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[6:]+"-"

class Fill(Enum):
    solid   = 1
    striped = 2
    empty   = 3

    def __repr__(self):
        """Make Printing less noisy"""
        return str(self)[5:]+"-"

class Card:
    def __init__(self, shape, color, count, fill, index = None):
        self.shape = shape
        self.color = color
        self.count = count
        self.fill = fill
        self.index = index

    def __repr__(self):
        return (repr(self.shape) + repr(self.color) + repr(self.count) + repr(self.fill))