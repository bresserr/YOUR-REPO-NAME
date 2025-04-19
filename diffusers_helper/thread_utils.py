import time

from threading import Thread, Lock


class Listener:
    task_queue = []
    lock = Lock()
    thread = None
    
    @classmethod
    def _process_tasks(cls):
        while True:
            task = None
            with cls.lock:
                if cls.task_queue:
                    task = cls.task_queue.pop(0)
                    
            if task is None:
                time.sleep(0.001)
                continue
                
            func, args, kwargs = task
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Error in listener thread: {e}")
    
    @classmethod
    def add_task(cls, func, *args, **kwargs):
        with cls.lock:
            cls.task_queue.append((func, args, kwargs))

        if cls.thread is None:
            cls.thread = Thread(target=cls._process_tasks, daemon=True)
            cls.thread.start()


def async_run(func, *args, **kwargs):
    Listener.add_task(func, *args, **kwargs)


class FIFOQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()
        print("【调试】创建新的FIFOQueue")

    def push(self, item):
        print(f"【调试】FIFOQueue.push: 准备添加项目: {item}")
        with self.lock:
            self.queue.append(item)
            print(f"【调试】FIFOQueue.push: 成功添加项目: {item}, 当前队列长度: {len(self.queue)}")

    def pop(self):
        print("【调试】FIFOQueue.pop: 准备弹出队列首项")
        with self.lock:
            if self.queue:
                item = self.queue.pop(0)
                print(f"【调试】FIFOQueue.pop: 成功弹出项目: {item}, 剩余队列长度: {len(self.queue)}")
                return item
            print("【调试】FIFOQueue.pop: 队列为空，返回None")
            return None

    def top(self):
        print("【调试】FIFOQueue.top: 准备查看队列首项")
        with self.lock:
            if self.queue:
                item = self.queue[0]
                print(f"【调试】FIFOQueue.top: 队列首项为: {item}, 当前队列长度: {len(self.queue)}")
                return item
            print("【调试】FIFOQueue.top: 队列为空，返回None")
            return None

    def next(self):
        print("【调试】FIFOQueue.next: 等待弹出队列首项")
        while True:
            with self.lock:
                if self.queue:
                    item = self.queue.pop(0)
                    print(f"【调试】FIFOQueue.next: 成功弹出项目: {item}, 剩余队列长度: {len(self.queue)}")
                    return item

            time.sleep(0.001)


class AsyncStream:
    def __init__(self):
        self.input_queue = FIFOQueue()
        self.output_queue = FIFOQueue()


class InterruptibleStreamData:
    def __init__(self):
        self.input_queue = FIFOQueue()
        self.output_queue = FIFOQueue()
        print("【调试】创建新的InterruptibleStreamData，初始化输入输出队列")
        
    # 推送数据至输出队列
    def push_output(self, item):
        print(f"【调试】InterruptibleStreamData.push_output: 准备推送输出: {type(item)}")
        self.output_queue.push(item)
        print(f"【调试】InterruptibleStreamData.push_output: 成功推送输出")
        
    # 获取下一个输出数据
    def get_output(self):
        print("【调试】InterruptibleStreamData.get_output: 准备获取下一个输出数据")
        item = self.output_queue.next()
        print(f"【调试】InterruptibleStreamData.get_output: 获取到输出数据: {type(item)}")
        return item
    
    # 推送数据至输入队列
    def push_input(self, item):
        print(f"【调试】InterruptibleStreamData.push_input: 准备推送输入: {type(item)}")
        self.input_queue.push(item)
        print(f"【调试】InterruptibleStreamData.push_input: 成功推送输入")
    
    # 获取下一个输入数据
    def get_input(self):
        print("【调试】InterruptibleStreamData.get_input: 准备获取下一个输入数据")
        item = self.input_queue.next()
        print(f"【调试】InterruptibleStreamData.get_input: 获取到输入数据: {type(item)}")
        return item
