

def train_one_point(task):
    gpu_id = task.conn.recv()
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu{0}'.format(gpu_id))

    model_name = task.conn.recv()
    files = task.conn.recv()
    epochs = task.conn.recv()
    learning_rate = task.conn.recv()
    batch_size = task.conn.recv()

    from .learning import cross_validate_model
    from .models import get_maxout
    import numpy as np
    import uuid
    model = get_maxout(25**2)
    batch_size = int(batch_size)
    # do not run CV in parallel
    vals = cross_validate_model(
        model, files,
        model_name=model_name + "_{0}_lr{1}_bs{2}".format(uuid.uuid4().hex, learning_rate, batch_size),
        batch_size=batch_size, epochs=epochs,
        val_frac=0.1, patience=10,
        lr_init=learning_rate, lr_scale_factor=1.0,
        log_to_file=True, read_into_ram=True, max_jobs=1)
    task.conn.send(-1 * np.array(vals['AUC']).mean())


class ObjectiveFunction(object):
    def __init__(self, model_name, train_files, epochs):
        self.model_name = model_name
        self.train_files = train_files
        self.epochs = epochs

    def __call__(self, args):
        from .parallel import map_gpu_pool, GPUWorker
        import numpy as np
        return np.array(map_gpu_pool(
            GPUWorker,
            [(train_one_point, self.model_name, self.train_files,
                self.epochs, learning_rate, batch_size)
                for learning_rate, batch_size in args],
            n_gpus=-1))[:,np.newaxis]


def bayesian_optimization(model_name, train_files, epochs):
    import GPyOpt
    from numpy.random import seed
    seed(12345)
    bounds = [(0.0001, 0.001), (32, 1024)]
    objective = ObjectiveFunction(model_name, train_files, epochs)
    bo = GPyOpt.methods.BayesianOptimization(f=objective, bounds=bounds)
    bo.run_optimization(max_iter=50, eps=1e-5)
    print bo.x_opt
    print bo.fx_opt

