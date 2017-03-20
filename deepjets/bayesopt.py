from .extern.six import string_types


def _train_one_point_helper(model_name, files, epochs, val_frac,
                            learning_rate, batch_size):
    from .learning import train_model, cross_validate_model
    from .models import get_maxout
    import numpy as np
    import uuid

    model = get_maxout(25**2)
    batch_size = int(batch_size)

    model_name = model_name + "_{0}_lr{1}_bs{2}".format(
        uuid.uuid4().hex, learning_rate, batch_size)

    if isinstance(files, string_types):
        # only one file; no CV
        # return val_loss
        model, history = train_model(
            model, files,
            model_name=model_name,
            batch_size=batch_size, epochs=epochs,
            val_frac=val_frac, patience=10,
            lr_init=learning_rate, lr_scale_factor=0.98,
            log_to_file=True, read_into_ram=True)
        return min(history.history['val_loss'])

    # CV
    # do not run CV in parallel
    vals = cross_validate_model(
        model, files,
        model_name=model_name,
        batch_size=batch_size, epochs=epochs,
        val_frac=val_frac, patience=10,
        lr_init=learning_rate, lr_scale_factor=0.98,
        log_to_file=True, read_into_ram=True, max_jobs=1)
    return -1 * np.array(vals['AUC']).mean()


def train_one_point(task):
    gpu_id = task.conn.recv()
    model_name = task.conn.recv()
    files = task.conn.recv()
    epochs = task.conn.recv()
    val_frac = task.conn.recv()
    learning_rate = task.conn.recv()
    batch_size = task.conn.recv()

    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu{0}'.format(gpu_id))

    result = _train_one_point_helper(
        model_name=model_name, files=files, epochs=epochs, val_frac=val_frac,
        learning_rate=learning_rate, batch_size=batch_size)
    task.conn.send(result)


class ObjectiveFunction(object):
    def __init__(self, model_name, train_files, epochs, val_frac):
        self.model_name = model_name
        self.train_files = train_files
        self.epochs = epochs
        self.val_frac = val_frac

    def __call__(self, args):
        from .parallel import map_gpu_pool, GPUWorker
        import numpy as np
        return np.array(map_gpu_pool(
            GPUWorker,
            [(train_one_point, self.model_name, self.train_files,
                self.epochs, self.val_frac, learning_rate, batch_size)
                for learning_rate, batch_size in args],
            n_gpus=-1))[:,np.newaxis]


def bayesian_optimization(model_name, train_files, max_iter, epochs, val_frac):
    import GPyOpt
    from numpy.random import seed
    seed(12345)
    bounds = [(0.0001, 0.001), (30, 1000)]
    objective = ObjectiveFunction(model_name, train_files, epochs, val_frac)
    bo = GPyOpt.methods.BayesianOptimization(
        f=objective, bounds=bounds,)
        #exact_feval=True,
        #acquisition='LCB', # Lower confidence bound method
        #acquisition_par=2, # Set parameter psi=2
        #normalize=True)    # Normalize the acquisition funtction
    bo.run_optimization(max_iter=max_iter, eps=1e-5, n_inbatch=4, n_procs=1,
                        batch_method='predictive',
                        acqu_optimize_method='fast_random',
                        verbose=True)
    print(bo.x_opt)
    print(bo.fx_opt)
    bo.plot_acquisition(filename='bo_acquisition.pdf')
    bo.plot_convergence(filename='bo_convergence.pdf')
