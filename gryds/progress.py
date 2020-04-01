import math
import sys
import time


class ProgressBar:
    """ A simple customisable progress """

    def __init__(self, n_total, length=20, complete_ch='#', name='Progress'):
        """
        Args:
            n_total (int): The total number of tasks
            length (int, optional): The number of characters, defaults to 20
            complete_ch (str, optional): The character of the complete portion
            name (str, optional): The name to be before the bar


        Example:
            >>> pbar = ProgressBar(n_total=20, length=10, name='Tunning')
            >>> # After 10 updates...
            Tunning: [#####-----]   50.00%
        """
        self._lenght = length
        self._n_done = 0
        self._n_total = n_total
        self._complete_ch = complete_ch
        self._remaining_ch = '-'
        self._name = name

    def update(self):
        """ Increase the number of tasks done and show the bar on console """
        self._n_done += 1
        self._show()
        if self._is_complete():
            print()

    def _show(self):
        """ Shows the bar on console and clear it """
        thebar = self._build_bar()
        print(thebar)
        sys.stdout.write('\033[F')

    def _build_bar(self):
        """ Computes how much work is done proportionally to the bar length and
        builds the bar string
        """
        percent_done = self._n_done / self._n_total
        prop_done = math.ceil(percent_done * self._lenght)
        todo = self._lenght - prop_done
        dones = prop_done * self._complete_ch
        todos = todo * self._remaining_ch
        return f"{self._name}: [{dones}{todos}]\t{100 * percent_done:3.2f}%"

    def _is_complete(self):
        return self._n_done == self._n_total

