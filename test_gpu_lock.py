#!/usr/bin/python

from deepjets import gpu_lock

id = gpu_lock.obtain_lock_id()

print 'now imagine a long run using id %s' % id

while True:
    pass
