
class TreeGenerator:
  def __init__(self):
    pass
    
  def gen(self, tree_type='complete_binary'):
    if tree_type.startswith('complete_binary'):
      depth = tree_type[15:]
      if depth == '': depth = 3
      else: depth = int(depth)
      return completeBinaryTree(depth)
    else:
      raise(NotImplemented)

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
