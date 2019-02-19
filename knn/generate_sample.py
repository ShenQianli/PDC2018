import random
n = int(input("numbers of v: "))
dim = int(input("dimensions: "))
k = int(input("top k: "))

with open('coordinates.txt', 'w') as f:
    f.write(str(n) + '\n')
    f.write(str(dim) + '\n')
    f.write(str(k) + '\n')
    for i in range(n):
        for k in range(dim):
            f.write(str(random.random()) + '\n');
