#!/usr/bin/env python

import sys

def read_iput(file):
    return list(str(file))



data = read_iput(sys.stdin)
for number in data:
    for num in number:
        print('%s\t%d' % (num, 1))
