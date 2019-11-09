import math
import sys
import time


class ProgressBar:

    def __init__(self, n_total, length=20, complete_ch='#', name='Progress'):
        self.length = length
        self.n_done = 0
        self.n_total = n_total
        self.complete_ch = complete_ch
        self.remaining_ch = '-'
        self.name = name

    def update(self):
        self.n_done += 1
        self.show()

    def show(self):
        thebar = self._build_bar()
        print(thebar)
        sys.stdout.write('\033[F')

    def _build_bar(self):
        percent_done = self.n_done / self.n_total
        prop_done = math.ceil(percent_done * self.length)
        todo = self.length - prop_done
        done_chs = prop_done * self.complete_ch
        todo_chs = todo * self.remaining_ch
        return f"{self.name}: [{done_chs}{todo_chs}]\t{100 * percent_done:3.2f}%"


if __name__ == '__main__':
    pbar = ProgressBar(40, length=30)
    for _ in range(40):
        pbar.update()
        time.sleep(0.5)
