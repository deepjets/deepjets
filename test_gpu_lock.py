#!/usr/bin/python

import time
from deepjets import gpu_lock

id = gpu_lock.obtain_lock_id(block=False)

print 'now imagine a long run using id %s' % id

while True:
    time.sleep(0.1)
