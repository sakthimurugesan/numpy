import numpy as np
from numpy import char as nc

a = np.array([1, 2, 3, 4, 5])
print(f'\nsingle dimension array {a}')

a = np.array(([1, 2, 3, 4]
              , [11, 22, 33, 44]))

print(f'2d array\n{a}')

a = np.arange(12)
print(f'\nnp.arange(12)\t{a}')

a = np.ones(5)
print(f'\nnp.ones(5)\t{a}')

a = np.zeros(5)

print(f'\nnp.zeros(5)\t{a}')

print('\n---------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4, 5])
print(f'single dimension array\t {a}')
print('array.ndim\t', a.ndim)
a = np.array(([1, 2, 3, 4]
              , [11, 22, 33, 44]))
print(f'2d array\n{a}')
print('array.ndim\t', a.ndim)

print('\n---------------------------------------------------\n\n')
a = np.array([1, 2, 3, 4, 5])
print('a\t', a)
print('len(a)\t', len(a))
print('a.size\t', a.size)

print('\n---------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4, 5])

print('a\t', a)
print('a.itemsize\t', a.itemsize)
print('a.dtype', a.dtype)

print('\n---------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4, 5])
print('a\t', a)
print('a.shape\t', a.shape)
a = np.array(([1, 2, 3, 4]
              , [11, 22, 33, 44]))
print('a\n', a)
print('a.shape', a.shape)
print('\n\na.reshape(4,2)\n', a.reshape(4, 2))

print('\n---------------------------------------------------\n\n')

print(f'\n\n{a}')

print('\na.max()\t', a.max())

print(f'\na.min()\t{a.min()}')

print('\na.max(axis=0)\t', a.max(axis=0))

print(f'\na.min(axis=0)\t{a.min(axis=0)}')

print('\na.max(axis=1)\t', a.max(axis=1))

print(f'\na.min(axis=1)\t{a.min(axis=1)}')

print('a.sum()\t', a.sum())
print('a.sum(axis=0)', a.sum(axis=0))
print('a.sum(axis=1)', a.sum(axis=1))

print('\n-----------------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4, 5])
print('a\t', a)

print('\na.sqrt', np.sqrt(a))

print('\na.std()\t', a.std())

a = np.array(([1, 2, 3, 4]
              , [11, 22, 33, 44]))
print('a\n', a)

print('\na.std(axis=0)\t', a.std(axis=0))
print('\na.std(axis=1)\t', a.std(axis=1))

print('\na.sqrt\t', np.sqrt(a))

print('\n-----------------------------------------------------------\n')

b = np.array(([1, 2, 3, 4],
              [5, 6, 7, 8]))
print('b\n', b)

c = np.array(([11, 12, 13, 14],
              [15, 16, 17, 18]))
print('\nc\n', c)

print(f'\nc+b\n{c + b}\nc-b\n{c - b}\nc*b\n{c * b}\n')

print('\nnp.hstack((b,c))\n', np.hstack((b, c)))

print('\nnp.vstack((b,c))\n', np.vstack((b, c)))

print(f'\na.transpose()\n{a.transpose()}')

print('\n-----------------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4, 5])
b = np.array([x for x in range(11, 16)])
print('a\t', a)

print('\nnp.log(a)\n', np.log(a))
print('\nnp.log10(a)\n', np.log10(a))
print('\nnp.log2(a)\n', np.log2(a))
print('\nnp.lcm(a, b)\n', np.lcm(a, b))

print('\n-----------------------------------------------------------\n\n')

c = np.array(([11, 12, 13, 14],
              [15, 16, 17, 18]))

d = np.array(([11, 12, 13, 14],
              [15, 16, 17, 18]))
print('\nc\n', c)

print('\nnp.log(c)\n', np.log(c))
print('\nnp.log10(c)\n', np.log10(c))
print('\nnp.log2(c)\n', np.log2(c))

print('\n-----------------------------------------------------------\n\n')
print('\nc.ravel()\n', c.ravel())
print('\nc.ravel(order="f")\n', c.ravel(order="f"))
print('\nc.flatten()\n', c.flatten())
print('\nc.flatten(order="f")\n', c.flatten(order="f"))
print('\nnp.add(c,d)\n', np.add(c, d))
print('\nnp.sin(a)\n', np.sin(a))

print('\n-----------------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4 - 1j, 5 + 4j])
print('\na\n', a)
print('\nnp.conj(a)\n', np.conj(a))  # doubt
print('\nnp.conj(a)\n', np.conj(a))
print('\nnp.real(a)\n', np.real(a))
print('\nnp.absolute(a)\n', np.absolute(a))

print('\n-----------------------------------------------------------\n\n')

a = np.array([1, 2, 3, 4, 5])
print(a)
e = np.array(a, dtype=np.float32)

print(f'\nnp.array(a, dtype=np.float32)\t{np.array(a, dtype=np.float32)}')

print('\n-----------------------------------------------------------\n\n')

d = np.arange(1, 13).reshape(3, 4)

print(f'\n{d}\n')

for i in np.nditer(d):
    print(i, end='\t')
print()
for i in np.nditer(d, order='f'):
    print(i, end='\t')

print('\n-----------------------------------------------------------\n\n')

print(f"np.char.add('hello',' world')\t{np.char.add('hello', ' world')}\n")

print(f"np.char.add(('hello','world'),('hello','world'))\t{np.char.add(('hello', 'world'), ('hello', 'world'))}\n")

print(f"np.char.center('hello',20,fillchar='*')\t\t{np.char.center('hello', 20, fillchar='*')}\n")

print(f"np.char.capitalize('hello')\t\t{np.char.capitalize('hello')}\n")

print(f"np.char.upper(['lower','lower'])\t\t{np.char.upper(['lower', 'lower'])}\n")

print(f"np.char.lower(['UPPER','UPPER'])\t\t{np.char.lower(['UPPER', 'UPPER'])}\n")

print(f"np.char.split('how are you')\t\t{np.char.split('how are you')}\n")
print("np.char.splitlines('i am \\n fine')\t\t{}\n".format(np.char.splitlines('i am \n fine')))

print(np.char.strip(['how are', 'you'], 'o'))

print(np.char.join([':', '-'], ['dmy', 'ymd']))
