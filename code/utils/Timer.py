"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import time


class Timer:
    """
    Time context manager for code block
        with Timer():
            do something
        Timer.get()
    """
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    def __init__(self, tape=None, **kwargs):
        if kwargs.get("name"):
            Timer.NAMED_TAPE[kwargs["name"]] = Timer.NAMED_TAPE[
                kwargs["name"]] if Timer.NAMED_TAPE.get(kwargs["name"]) else 0.
            self.named = kwargs["name"]
        else:
            self.named = False
            self.tape = tape or Timer.TAPE

    @staticmethod
    def get():
        if len(Timer.TAPE) > 1:
            return Timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"

        if select_keys is None:
            for key, value in Timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = Timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"

        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key in Timer.NAMED_TAPE.keys():
                Timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                Timer.NAMED_TAPE[key] = 0

    def __enter__(self):
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            Timer.NAMED_TAPE[self.named] += time.time() - self.start
        else:
            self.tape.append(time.time() - self.start)
