import multiprocessing
import time
from .tasksystem import AsyncTask
from .gpu_lock import obtain_lock_id_to_hog, launch_reaper


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


class GPUWorker(AsyncTask):
    def __init__(self, func, *args, **kwargs):
        self.gpu_id = None
        self.args = args
        super(GPUWorker, self).__init__(func, must_exec=True, **kwargs)

    def start(self):
        if self.gpu_id is None:
            raise RuntimeError(
                "attempted to start GPUWorker without first setting gpu_id")
        ret = super(GPUWorker, self).start()
        self.conn.send(self.gpu_id)
        for arg in self.args:
            self.conn.send(arg)
        return ret


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


def run_gpu_pool(workers, n_gpus=-1, sleep=0.1):
    # defensive copy
    workers = workers[:]
    if n_gpus < 1:
        # determine number of GPUs available
        from pycuda import driver
        driver.init()
        n_gpus = driver.Device.count()
    processes = []
    p = None
    try:
        while True:
            # collect gpu_ids from finished workers
            finished = []
            for p in processes:
                if not p.is_alive():
                    finished.append(p)
            processes = [p for p in processes if p not in finished]
            while len(workers) > 0:
                # get available gpu_id
                gpu_id = obtain_lock_id_to_hog(block=False)
                if gpu_id == -1:
                    # all GPUs currently being used. Will try again next time.
                    break
                p = workers.pop(0)
                p.gpu_id = gpu_id
                p.start()
                launch_reaper(gpu_id, p.child_pid)
                processes.append(p)
            if len(workers) == 0 and len(processes) == 0:
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


def map_gpu_pool(process, args, n_gpus=-1, **kwargs):
    procs = [process(*arg, **kwargs) for arg in args]
    run_gpu_pool(procs, n_gpus=n_gpus)
    return [p.conn.recv() for p in procs]


def map_pool_kwargs(process, kwargs, n_jobs=-1):
    procs = [process(**args) for args in kwargs]
    run_pool(procs, n_jobs=n_jobs)
    return [p.output for p in procs]
