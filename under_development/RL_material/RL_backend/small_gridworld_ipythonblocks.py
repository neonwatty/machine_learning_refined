"""
BlockGrid is a class that displays a colored grid in the
IPython Notebook. The colors can be manipulated, making it useful for
practicing control flow stuctures and quickly seeing the results.

See the IPython Notebook at https://gist.github.com/4499453 for a demo.

"""

import copy
import itertools
import numbers

from operator import iadd

from IPython.display import HTML, display

__all__ = ['Block', 'BlockGrid', 'InvalidColorSpec']


_TABLE = '<table><tbody>{0}</tbody></table>'
_TR = '<tr>{0}</tr>'
_TD = ('<td style="width: 35px; height: 35px;'
       ' border: 5px solid white; background-color: {0};"></td>')


_SINGLE_ITEM = 'single item'
_SINGLE_ROW = 'single row'
_ROW_SLICE = 'row slice'
_DOUBLE_SLICE = 'double slice'


class InvalidColorSpec(Exception):
    """
    Error for a color value that is not a number.

    """
    pass


class Block(object):
    """
    A class with .red, .green, and .blue attributes.

    """

    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue

    @staticmethod
    def check_value(value):
        """
        Check that a value is a number and constrain it to [0 - 255].

        """
        if not isinstance(value, numbers.Number):
            s = 'value must be a number. got {0}.'.format(value)
            raise InvalidColorSpec(s)

        return min(255, max(0, value))
    
    @property
    def red(self):
        return self._red

    @red.setter
    def red(self, value):
        value = self.check_value(value)
        self._red = value

    @property
    def green(self):
        return self._green

    @green.setter
    def green(self, value):
        value = self.check_value(value)
        self._green = value

    @property
    def blue(self):
        return self._blue

    @blue.setter
    def blue(self, value):
        value = self.check_value(value)
        self._blue = value

    def set_colors(self, color_tuple):
        """
        Updated colors from a tuple of RGB integers.

        Parameters
        ----------
        color_tuple : tuple of int
            Tuple containing integers of (red, green, blue) values.

        """
        if len(color_tuple) != 3:
            s = 'color_tuple must have xthree integers. got {0}.'
            raise ValueError(s.format(color_tuple))

        self.red = color_tuple[0]
        self.green = color_tuple[1]
        self.blue = color_tuple[2]

    @property
    def td(self):
        """
        Return the HTML of a table cell with the background
        color of this Block.

        """
        rgb = 'rgb({0}, {1}, {2})'.format(self._red, self._green, self._blue)
        return _TD.format(rgb)


class BlockGrid(object):
    """
    A grid of squares whose colors can be individually controlled.

    Individual squares have a width and height of 10 screen pixels.
    To get the second Block in the third row use block = grid[1, 2].

    Parameters
    ----------
    width : int
        Number of squares wide to make the grid.
    height : int
        Number of squares high to make the grid.
    fill : tuple of int, optional
        An optional initial color for the grid, defaults to black.
        Specified as a tuple of (red, green, blue). E.g.: (10, 234, 198)

    Attributes
    ----------
    width : int
        Number of squares wide to make the grid.
    height : int
        Number of squares high to make the grid.

    """

    def __init__(self, width, height, fill=(0, 0, 0)):
        self.width = width
        self.height = height
        self._initialize_grid(fill)

    def _initialize_grid(self, fill):
        grid = [[Block(*fill) for _ in xrange(self.width)]
                for _ in xrange(self.height)]

        self._grid = grid

    @classmethod
    def _view_from_grid(cls, grid):
        """
        Make a new BlockGrid from a list of lists of Block objects.

        """
        new_width = len(grid[0])
        new_height = len(grid)

        new_BG = cls(new_width, new_height)
        new_BG._grid = grid

        return new_BG

    @staticmethod
    def _categorize_index(index):
        """
        Used by __getitem__ and __setitem__ to determine whether the user
        is asking for a single item, single row, or some kind of slice.

        """
        if isinstance(index, int):
            return _SINGLE_ROW

        elif isinstance(index, slice):
            return _ROW_SLICE

        elif isinstance(index, tuple):
            if len(index) not in (1, 2):
                s = 'Invalid index, too many dimensions.'
                raise IndexError(s)

            if isinstance(index[0], slice):
                if isinstance(index[1], (int, slice)):
                    return _DOUBLE_SLICE

            if isinstance(index[1], slice):
                if isinstance(index[0], (int, slice)):
                    return _DOUBLE_SLICE

            elif isinstance(index[0], int) and isinstance(index[0], int):
                return _SINGLE_ITEM

        raise IndexError('Invalid index.')

    def __getitem__(self, index):
        ind_cat = self._categorize_index(index)

        if ind_cat == _SINGLE_ROW:
            return self._grid[index]

        elif ind_cat == _SINGLE_ITEM:
            return self._grid[index[0]][index[1]]

        elif ind_cat == _ROW_SLICE:
            return BlockGrid._view_from_grid(self._grid[index])

        elif ind_cat == _DOUBLE_SLICE:
            new_grid = self._get_double_slice(index)
            return BlockGrid._view_from_grid(new_grid)

    def __setitem__(self, index, value):
        ind_cat = self._categorize_index(index)

        if ind_cat == _SINGLE_ROW:
            map(Block.set_colors, self._grid[index],
                itertools.repeat(value, len(self._grid[index])))

        elif ind_cat == _SINGLE_ITEM:
            self._grid[index[0]][index[1]].set_colors(value)

        else:
            if ind_cat == _ROW_SLICE:
                sub_grid = self._grid[index]

            elif ind_cat == _DOUBLE_SLICE:
                sub_grid = self._get_double_slice(index)

            nblocks = len(sub_grid) * len(sub_grid[0])
            map(Block.set_colors, itertools.chain(*sub_grid),
                itertools.repeat(value, nblocks))

    def _get_double_slice(self, index):
        sl_height = index[0]
        sl_width = index[1]

        if isinstance(sl_width, int):
            sl_width = slice(sl_width, sl_width + 1)

        if isinstance(sl_height, int):
            sl_height = slice(sl_height, sl_height + 1)

        rows = self._grid[sl_height]
        grid = [r[sl_width] for r in rows]

        return grid

    def _repr_html_(self):
        html = reduce(iadd,
                      (_TR.format(reduce(iadd, (block.td for block in row)))
                       for row in self._grid))

        return _TABLE.format(html)

    def copy(self):
        """
        Returns an independent copy of this BlockGrid.

        """
        return copy.deepcopy(self)

    def show(self):
        """
        Display colored grid as an HTML table.

        """
        return display(HTML(self._repr_html_()))
