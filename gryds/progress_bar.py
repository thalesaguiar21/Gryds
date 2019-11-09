import math
import sys
import time


class ProgressBar:

    def __init__(self, n_total, length=20, complete_ch='#'):
        self.length = length
        self.n_done = 0
        self.n_total = n_total
        self.complete_ch = complete_ch
        self.remaining_ch = '-'

    def update(self):
        self.n_done += 1
        self.show()

    def show(self):
        percent_done = self.n_done / self.n_total
        prop_done = math.ceil(percent_done * self.length)
        todo = self.length - prop_done
        thebar = f"{self.complete_ch * prop_done}{self.remaining_ch * todo}"
        print(thebar)
        sys.stdout.write('\033[F')

if __name__ == '__main__':
    pbar = ProgressBar(40, length=30)
    for _ in range(40):
        pbar.update()
        time.sleep(0.5)
