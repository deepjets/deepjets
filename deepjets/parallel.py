import multiprocessing
import time


class Worker(multiprocessing.Process):
    def __init__(self):
        super(Worker, self).__init__()
        self.result = multiprocessing.Queue()

    @property
    def output(self):
        return self.result.get()

    def run(self):
        self.result.put(self.work())


class FuncWorker(Worker):
    def __init__(self, func, *args, **kwargs):
        super(FuncWorker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def work(self):
        return self.func(*self.args, **self.kwargs)


def run_pool(workers, n_jobs=-1, sleep=0.1):
    # defensive copy
    workers = workers[:]
    if n_jobs < 1:
        n_jobs = multiprocessing.cpu_count()
    processes = []
    p = None
    try:
        while True:
            active = multiprocessing.active_children()
            while len(active) < n_jobs and len(workers) > 0:
                p = workers.pop(0)
                p.start()
                processes.append(p)
                active = multiprocessing.active_children()
            if len(workers) == 0 and len(active) == 0:
                break
            time.sleep(sleep)
    except KeyboardInterrupt, SystemExit:
        if p is not None:
            p.terminate()
        for p in processes:
            p.terminate()
        raise


def map_pool(process, args, n_jobs=-1, **kwargs):
    procs = [process(*arg, **kwargs) for arg in args]
    run_pool(procs, n_jobs=n_jobs)
    return [p.output for p in procs]


def map_pool_kwargs(process, kwargs, n_jobs=-1):
    procs = [process(**args) for args in kwargs]
    run_pool(procs, n_jobs=n_jobs)
    return [p.output for p in procs]
