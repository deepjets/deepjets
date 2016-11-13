#!/usr/bin/env python

import sys
from deepjets.tasksystem import AsyncTask

def start():
    # Create a real subprocess, not just a fork.
    asyncTask = AsyncTask(func=process, name="My sub process", must_exec=True)
    asyncTask.start()

    # We have a connection (duplex pipe) to communicate.
    asyncTask.conn.send(print_action)
    assert asyncTask.conn.recv() == 42
    asyncTask.conn.send(sys.exit)

    asyncTask.join()

def print_action():
    print "Hello"
    return 42

def process(asyncTask):
    while True:
        action = asyncTask.conn.recv()
        res = action()
        asyncTask.conn.send(res)

if __name__ == "__main__":
    start()
