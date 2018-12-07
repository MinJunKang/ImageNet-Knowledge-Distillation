
import threading
import multiprocessing
import queue
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import Pool
from multiprocessing import Queue
from multiprocessing import Lock

class Multitask(object):
    def __init__(self,num_max_thread,num_max_process):

        self.lock_thread = threading.Lock()
        self.manager = multiprocessing.Manager()
        self.lock_process = self.manager.Lock()

        self.threads = []
        self.processes = []
        self.pools = []

        self.result_thread = queue.Queue()
        self.result_process = self.manager.Queue()
        self.result_pool = queue.Queue()

        self.num_maxthread = num_max_thread
        self.num_maxprocess = num_max_process

        self.pool = None
        self.process = None

    def begin_thread(self,func,input = []):
        if(len(self.threads) >= self.num_maxthread):
            self.join_thread(timeout = 0.5)
        t = threading.Thread(target=func, name='thread_'+str(len(self.threads)), args=(self.lock_thread,self.result_thread,input))
        t.start()
        self.threads.append(t)

    def join_thread(self,timeout = -1,ignore_unfinished = False):

        if(timeout == -1):
            for th in self.threads:
                th.join()
            self.threads.clear()
        else:
            delete_count = 0
            for idx in range(len(self.threads)):
                th = self.threads[idx-delete_count]
                th.join(timeout=timeout)
                if(th.isAlive() == False):
                    del self.threads[idx-delete_count]
                    delete_count += 1

        if(len(self.threads) == 0):
            return True
        else:
            if(ignore_unfinished):
                self.threads.clear()
            return False

    def begin_pool(self,func,input=[]):
        if self.pool is None:
            self.pool = ThreadPool(processes=self.num_maxprocess)
        p = self.pool.apply_async(func,(self.lock_thread,input))
        self.pools.append(p)

    def join_pool(self,timeout=-1,terminate_all = False):
        if(timeout == -1):
            for p in self.pools:
                self.result_pool.put(p.get())
            self.pools.clear()
        else:
            delete_count = 0
            for idx in range(len(self.pools)):
                p = self.pools[idx-delete_count]
                try:
                    self.result_pool.put(p.get(timeout = timeout))
                except multiprocessing.context.TimeoutError:
                    continue
                del self.pools[idx-delete_count]
                delete_count += 1
        if(len(self.pools) == 0):
            if(self.pool != None):
                self.pool.close()
            self.pool = None
            return True
        else:
            if(terminate_all):
                if(self.pool != None):
                    self.pools.clear()
                    self.pool.terminate()
                    self.pool.close()
                self.pool = None
            return False

    def begin_process(self,func,input=[]):
        if self.process is None:
            self.process = Pool(processes=self.num_maxprocess)
        p = self.process.apply_async(func,(self.lock_process,input))
        self.processes.append(p)

    def join_process(self,timeout=-1,terminate_all = False):
        if(timeout == -1):
            for p in self.processes:
                self.result_process.put(p.get())
            self.processes.clear()
        else:
            delete_count = 0
            for idx in range(len(self.processes)):
                p = self.processes[idx-delete_count]
                try:
                    self.result_process.put(p.get(timeout = timeout))
                except multiprocessing.context.TimeoutError:
                    continue
                del self.processes[idx-delete_count]
                delete_count += 1
        if(len(self.processes) == 0):
            self.process.close()
            self.process = None
            return True
        else:
            if(terminate_all):
                self.processes.clear()
                self.process.terminate()
                self.process.close()
                self.process = None
            return False


    def Set_Queue_Thread_Size(self,size = 0):
        self.result_thread = queue.Queue(size)

    def Set_Queue_Process_Size(self,size = 0):
        self.result_process = self.manager.Queue(size)

    def Set_Queue_Pool_Size(self,size = 0):
        self.result_pool = queue.Queue(size)

    def Get_Thread_Result(self):
        result = []
        for i in range(self.result_thread.qsize()):
            result.append(self.result_thread.get())
        return result

    def Get_Process_Result(self):
        result = []
        for i in range(self.result_process.qsize()):
            result.append(self.result_process.get())
        return result

    def Get_Pool_Result(self):
        result = []
        for i in range(self.result_pool.qsize()):
            result.append(self.result_pool.get())
        return result

    def Set_Thread_Max(self,max_num):
        self.num_maxthread = max_num

    def Set_Process_Max(self,max_num):
        self.num_maxprocess = max_num




