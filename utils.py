import warnings
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
import traceback


def parallel_worker(worker, args, kwargs, queue, proc_idx):
    result = worker(*args, **kwargs)
    queue.put([result, proc_idx])


def parallel_execute(worker, args, num_proc, kwargs=None, show_progress=True, desc=None, terminate_func=None, raise_exception=True):
    '''
    Tool for parallel execution
    '''
    if show_progress:
        pbar = tqdm(total=len(args), desc=desc)

    queue = mp.Queue()
    procs = {}
    n_active_proc = 0
    if kwargs is None: kwargs = [{}] * len(args)

    try:

        # loop over arguments for all processes
        for proc_idx, (arg, kwarg) in enumerate(zip(args,kwargs)):

            if num_proc > 1:
                proc = mp.Process(target=parallel_worker, args=(worker, arg, kwarg, queue, proc_idx))
                proc.start()
                procs[proc_idx] = proc
                n_active_proc += 1
                if n_active_proc >= num_proc: # launch a new process after an existing one finishes
                    result, proc_idx = queue.get()
                    procs.pop(proc_idx)
                    yield result

                    if terminate_func and terminate_func(result): # terminate condition meets
                        for p in procs.values(): # terminate all running processes
                            p.terminate()
                        if show_progress:
                            pbar.update(pbar.total - pbar.last_print_n)
                            pbar.close()
                        return
                    
                    n_active_proc -= 1

                    if show_progress:
                        pbar.update(1)
            else:
                parallel_worker(worker, arg, kwarg, queue, proc_idx) # no need to use mp.Process when serial
                result, _ = queue.get()
                yield result

                if terminate_func and terminate_func(result): # terminate condition meets
                    if show_progress:
                        pbar.update(pbar.total - pbar.last_print_n)
                        pbar.close()
                    return

                if show_progress:
                    pbar.update(1)

        for _ in range(n_active_proc): # wait for existing processes to finish
            result, proc_idx = queue.get()
            procs.pop(proc_idx)
            yield result

            if terminate_func and terminate_func(result): # terminate condition meets
                for p in procs.values(): # terminate all running processes
                    p.terminate()
                if show_progress:
                    pbar.update(pbar.total - pbar.last_print_n)
                    pbar.close()
                return

            if show_progress:
                pbar.update(1)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[parallel_execute] interrupt')
        else:
            print('[parallel_execute] exception:', e)
            print(traceback.format_exc())
        for proc in procs.values():
            proc.terminate()
        if raise_exception:
            raise e

    if show_progress:
        pbar.close()
