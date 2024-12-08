import numpy as np

import math
from sympy import primerange

from multiprocessing import Pool, Manager

thread_count = 16


def generate_squarefree_semiprimes(n):
    semiprimes = []
    primes = list(primerange(2, 1000))
    
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            semiprime = primes[i] * primes[j]
            semiprimes.append(semiprime)
            
    if(len(semiprimes) < n):
        raise NameError("Not enough semiprimes!")
    
    return sorted(semiprimes)[:n]

def generate_square_semiprimes(n):
    semiprimes = []
    primes = list(primerange(2, 1000))
    
    for i in range(len(primes)):
        semiprime = primes[i] * primes[i]
        semiprimes.append(semiprime)
        
    if(len(semiprimes) < n):
        raise NameError("Not enough semiprimes!")
    
    return sorted(semiprimes)[:n]


def generate_semiprimes(n):
    semiprimes = []
    primes = list(primerange(2, 1000))
    
    for i in range(len(primes)):
        for j in range(i, len(primes)):
            semiprime = primes[i] * primes[j]
            semiprimes.append(semiprime)
            
    if(len(semiprimes) < n):
        raise NameError("Not enough semiprimes!")
    
    return sorted(semiprimes)[:n]



def get_period(a, mod):
    iteration = 0
    temp = a
    started = False
    while not started or (temp != a and temp != 0):
        temp = (temp * a) % mod
        iteration += 1
        started = True
    return iteration



manager = Manager()
probs = manager.dict()


#semiprimes = generate_semiprimes(1000):
#semiprimes = generate_square_semiprimes(100):
semiprimes = generate_squarefree_semiprimes(3000)

def process(semiprime):
    found = False
    found_count = 0.0
    for a in range(semiprime):
        period = get_period(a, semiprime)
        g = math.gcd(semiprime, a ** int(period / 2) + 1)
        if period & (period-1) == 0 and period != 1 and g != 1 and g != semiprime:
            print(f"mod = {semiprime}, a = {a}, period = {period}, factors = ({g}, {int(semiprime / g)})")
            found = True
            found_count += 1
            #break
    if not found:
        print("!!!!!!!!!!!!!!!! NOT FOUND !!!!!!!!!!!!!!!!")
    probs[semiprime] = found_count / semiprime
    print("----------------------------------------------------------------------")

pool = Pool(thread_count)
pool.map(process, semiprimes)

print(probs)

keys = np.fromiter(probs.keys(), dtype=int)
vals = np.fromiter(probs.values(), dtype=float)



print(vals)
print(np.mean(vals))


folder = "Results"
suffix = ""


np.save(f"{folder}/semiprimes{suffix}", keys)
np.save(f"{folder}/a-probabilities{suffix}", vals)
