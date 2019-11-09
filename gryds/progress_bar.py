import math
import sys
import time


class ProgressBar:
    """
    Args:
        n_total
    """

    def __init__(self, n_total, length=20, complete_ch='#', name='Progress'):
        self._lenght = length
        self._n_done = 0
        self._n_total = n_total
        self._complete_ch = complete_ch
        self._remaining_ch = '-'
        self._name = name

    def update(self):
        self._n_done += 1
        self._show()
        if self._is_complete():
            print()

    def _show(self):
        thebar = self._build_bar()
        print(thebar)
        sys.stdout.write('\033[F')

    def _build_bar(self):
        percent_done = self._n_done / self._n_total
        prop_done = math.ceil(percent_done * self._lenght)
        todo = self._lenght - prop_done
        dones = prop_done * self._complete_ch
        todos = todo * self._remaining_ch
        return f"{self._name}: [{dones}{todos}]\t{100 * percent_done:3.2f}%"

    def _is_complete(self):
        return self._n_done == self._n_total


if __name__ == '__main__':
    pbar = ProgressBar(40, length=30)
    for _ in range(40):
        pbar.update()
        time.sleep(0.5)
