import asyncio
import os

from milabench.executors import PackExecutor, VoirExecutor, NJobs, TimeOutExec
from milabench.pack import Package
from milabench.cli import _get_multipack
from milabench.alt_async import proceed

class MockPack(Package):
    pass


TEST_FOLDER = os.path.dirname(__file__)

def benchio():
    packs = _get_multipack(
        os.path.join(TEST_FOLDER, 'config', 'benchio.yaml'),
        base='/tmp',
        use_current_env=True,
    )

    _, pack = packs.packs.popitem()
    return pack



def test_pack_executor():
    # voir is not setup so we are not receiving anything
    exec = PackExecutor(benchio(), "--start", "2", "--end", "20")
    
    acc = 0
    for r in proceed(exec.execute()):
        print(r)
        acc += 1
        
    assert acc == 2, "Only two message received"
        
    
def test_voir_executor():
    exec = PackExecutor(benchio(), "--start", "2", "--end", "20")
    voir = VoirExecutor(exec)
    
    acc = 0
    for r in proceed(voir.execute()):
        print(r)
        acc += 1
        
    assert acc > 2 and acc < 70
        

def test_timeout():
    exec = PackExecutor(benchio(), "--start", "2", "--end", "20", '--sleep', 20)
    voir = VoirExecutor(exec)
    timed = TimeOutExec(voir, delay=1)
    
    acc = 0
    for r in proceed(timed.execute()):
        print(r)
        acc += 1
        
    assert acc == 70 * 5


def test_njobs_executor():
    exec = PackExecutor(benchio(), "--start", "2", "--end", "20")
    voir = VoirExecutor(exec)
    njobs = NJobs(voir, 5)
    
    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1
        
    assert acc == 70 * 5


def test_njobs_novoir_executor():
    exec = PackExecutor(benchio(), "--start", "2", "--end", "20")
    njobs = NJobs(exec, 5)
    
    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1
        
    assert acc == 2 * 5
