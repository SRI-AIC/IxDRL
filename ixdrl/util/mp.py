import multiprocessing as mp
import os
from multiprocessing.managers import NamespaceProxy, BaseManager
from types import MethodType
from typing import Callable, Optional, List, Any

import tqdm
from joblib import Parallel, delayed, parallel_backend

from .logging import MultiProcessLogger, create_mp_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class _ProgressParallel(Parallel):
    """
    See: https://stackoverflow.com/a/61900501/16031961
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_num_workers(processes: int) -> int:
    """
    Gets the number of parallel workers according to the available CPUs.
    :param int processes: the number of parallel processes to use according to the `joblib` convention.
    `-1` means all CPUs are used, `1` means no parallel computing, `<-1` means `(n_cpus + 1 + n_jobs)` are used,
    `None` is equivalent to `n_jobs=1`.
    :rtype: int
    :return: the number of available workers.
    """
    if processes is None or processes == 0:
        return 1
    cpus = os.cpu_count()
    if processes >= 1:
        return min(processes, cpus)
    return max(1, cpus + 1 + processes)


def run_parallel(func: Callable,
                 args: List,
                 processes: Optional[int] = None,
                 use_tqdm: bool = True,
                 mp_logging: bool = False) -> List:
    """
    Run the given function for each of the given arguments in parallel and returns  alist with the results.
    :param func: the function to be executed.
    :param list args: the list of arguments for the function to be processed in parallel. If the function has multiple
    arguments, this should be a list of tuples, the length of each should match the function's arity.
    :param int processes: the number of parallel processes to use.  `-1` means all CPUs are used, `1` means no parallel
    computing, `<-1` means `(n_cpus + 1 + n_jobs)` are used, `None` is equivalent to `n_jobs=1`.
    :param bool use_tqdm: whether to show a progress bar during parallel execution.
    :param  bool mp_logging: whether to use multiprocess logging.
    :rtype: list
    :return: a list with the results of executing the given function over each of the arguments. Indices will be
    aligned with the input arguments.
    """

    # check processes
    processes = get_num_workers(processes)
    processes = min(processes, len(args))

    # check multiprocess logging
    if mp_logging and processes > 1:
        # redirects function to _log_processor to assign log handler
        assert MultiProcessLogger.queue is not None, 'MultiProcessLogger has not been created'
        args = [(func, _a, MultiProcessLogger.queue) for _a in args]
        func = _log_processor

    if processes == 1:
        # if single-process just call the function in a for loop
        if use_tqdm:
            args = tqdm.tqdm(args)
        return [func(*_args) for _args in args]

    star = isinstance(args[0], tuple)  # star if function is multi-argument

    with parallel_backend('loky', inner_max_num_threads=os.cpu_count() // processes):  # spread cpus per job
        return _ProgressParallel(n_jobs=processes, use_tqdm=use_tqdm, total=len(args))(
            delayed(func)(*(arg if star else [arg])) for arg in args)


def _log_processor(func: Callable, args, queue: mp.Queue) -> Any:
    if MultiProcessLogger.queue is None:
        # if not already set, set root logger of this process to redirect to queue
        create_mp_log_handler(queue)

        # override singleton queue in this process to allow for nested calls to `run_parallel`
        MultiProcessLogger.queue = queue

    # execute function
    star = isinstance(args, tuple)  # star if function is multi-argument
    return func(*(args if star else [args]))


class SharedContainer(object):
    """
    An object used to share data among multiple processes when used together with a `SharedContainerManager`.
    Use `manager.container.data` to retrieve the shared data/object.
    """
    def __init__(self, data):
        """
        Creates a new sharable container with the given data.
        :param data: the data to store in the container.
        """
        self.data = data


class _ContainerProxy(NamespaceProxy):
    """
    See: https://stackoverflow.com/a/68123850/16031961
    """
    _exposed_ = tuple(dir(SharedContainer))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, MethodType):
            def wrapper(*args, **kwargs):
                return self._callmethod(name, args, kwargs)

            return wrapper
        return result


class SharedContainerManager(BaseManager):
    """
    A multiprocessing manager with a shareable container.
    Use `manager.container.data` to retrieve the shared data/object.
    """

    def __init__(self, data):
        """
        Creates a new shared container manager.
        :param data: the data to place into the container to be shared among processes.
        """
        SharedContainerManager.register('SharedContainer', SharedContainer, _ContainerProxy)
        super().__init__()
        self._data = data
        self._container: SharedContainer = None

    @property
    def container(self) -> SharedContainer:
        """
        Gets the shared container managed by this manager.
        :rtype: SharedContainer
        :return: the shared container.
        """
        if self._container is None:
            self._container = self.SharedContainer(self._data)
        return self._container
