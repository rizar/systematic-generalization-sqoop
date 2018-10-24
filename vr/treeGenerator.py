
class TreeGenerator:
    def __init__(self):
        pass

    def gen(self, tree_type='complete_binary'):
        if tree_type.startswith('complete_binary'):
            depth = tree_type[len('complete_binary'):]
            if depth == '': depth = 3
            else: depth = int(depth)
            return completeBinaryTree(depth)
        elif tree_type.startswith('chainTree'):
            depth = tree_type[len('chainTree'):]
            if depth == '': depth = 8
            else: depth = int(depth)
            return chainTree(depth)
        elif tree_type.startswith('pairChainTree'):
            depth = tree_type[len('pairChainTree'):]
            if depth == '': depth = 8
            else: depth = int(depth)
            return pairChainTree(depth)
        else:
            raise(NotImplemented)

    def genHeap(self, tree_type='complete_binary'):
        if tree_type.startswith('complete_binary'):
            depth = tree_type[len('complete_binary'):]
            if depth == '': depth = 3
            else: depth = int(depth)
            return heapCompleteBinaryTree(depth)
        elif tree_type.startswith('pairChainTree'):
            depth = tree_type[len('pairChainTree'):]
            if depth == '': depth = 7
            else: depth = int(depth)
            return heapPairChainTree(depth)
        else:
            raise(NotImplemented)

def heapCompleteBinaryTree(depth=3):
    childrens = []
    num = 2 ** depth
    for i in range(num-1):
        childrens.append([i*2+1,i*2+2])
    for _ in range(num):
        childrens.append([])
    return childrens

def heapPairChainTree(depth=8):
    children = [[1,2]]
    for i in range(3,2*depth+1):
        children.append([i])
    children.append([])
    children.append([])
    return children

def completeBinaryTree(depth=3):
    arities = []
    def gen(idepth=0):
        if idepth == depth:
            arities.append(0)
            return
        else:
            arities.append(2)
            gen(idepth+1)
            gen(idepth+1)
    gen(0)
    return arities

def chainTree(depth=8):
    if depth == 0: return [0]
    arities = []
    while (len(arities) < depth-1): arities.append(1)
    arities.append(0)
    return arities

def pairChainTree(depth=8):
    if depth < 2: raise Exception('Depth has to be at least 2')
    half = int(depth / 2)
    above = [1] * half
    above[-1] = 2
    below = [1] * half
    below[-1] = 0
    return above + below + below
