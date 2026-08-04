import os
import dill
import pandas as pd
from pathlib import Path
from functools import wraps

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memory = {}
    def __call__(self, *args, **kwargs):
        key = str(args) + str({k:v for k,v in kwargs.items() if type(v) != pd.DataFrame})
        if not key in self.memory:
            self.memory[key] = self.f(*args, **kwargs)
        return self.memory[key]
    

class DiskMemoize:
    def __init__(self, f, memory_path="../simulation/memos"):
        self.f = f
        self.memory = {}
        if not os.path.exists(Path(memory_path)):
            raise ValueError(f"Path {memory_path} does not exist.")
        self.memory_path = Path(memory_path) / f"{f.__name__}_dict.pkl"
        self.log_path = Path(memory_path) / f"{f.__name__}_logs.log"
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "wb") as j:
                dill.dump(self.memory, j)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as log:
                log.write(f"--- {self.f.__name__} ---\n")
    def __call__(self, *args, **kwargs):
        with open(self.memory_path, "rb") as j:
            self.memory = dill.load(j)
        key = str(args) + str({k:v for k,v in kwargs.items() if type(v) != pd.DataFrame})
        if not key in self.memory.keys():
            self.memory[key] = self.f(*args, **kwargs)
            with open(self.memory_path, "wb") as j:
                dill.dump(self.memory, j)
            with open(self.log_path, "a") as log:
                log.write(f"- ENTRY for key : {key} \n")
        else:
            with open(self.log_path, "a") as log:
                log.write(f"- ACCESS for key : {key} \n")
        return self.memory[key]
    

def memoclean(func, memory_path="../simulation/memos"):
    @wraps(func)
    def memory_cleaning(*args, memory_path=memory_path, **kwargs):
        out = func(*args, **kwargs)
        memory_path = Path(memory_path)
        for fn in [x for x in os.listdir(memory_path) if ((".pkl" in x) or (".log" in x))]:
            os.remove(memory_path / fn)
            print(f"\nLOG: memory_cleaning: file {memory_path / fn} was removed")
        return out
    return memory_cleaning