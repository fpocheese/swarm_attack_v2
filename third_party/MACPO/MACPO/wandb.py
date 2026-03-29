"""Wandb stub — no-op for offline training"""
class _Run:
    def log(self, *a, **kw): pass
    def finish(self, *a, **kw): pass
    def __setattr__(self, k, v): object.__setattr__(self, k, v)

_run = _Run()

def init(*a, **kw): return _run
def log(*a, **kw): pass
def finish(*a, **kw): pass
