
class node:
    def __init__(self,x,y):
        self.l=x
        self.r=y
        self.key=0
        if x<y:
            mid=int((x+y)/2)
            self.left=node(x,mid)
            self.right=node(mid+1,y)

    def set(p,x,y):
        if p.l==p.r:
            p.key=y
            return
        mid=int((p.l+p.r)/2)
        if x<=mid:
            p.left.set(x,y)
        else:
            p.right.set(x,y)

    def calc(p,x,y):
        if p.l==p.r:
            return p.key
        mid=int((p.l+p.r)/2)
        if mid>=y:
            return p.left.calc(x,y)
        elif mid<x:
            return p.right.calc(x,y)
        else:
            return max(p.left.calc(x,mid),p.right.calc(mid+1,y))

A=[]
N=int(input("N:"))
Q=int(input("Q:"))
Tree=node(1,N)
for i in range(0,N):
    print ("A"+str(i)+":")
    x=int(input(""))
    A.append(x)
    Tree.set(i+1,x)
for i in range(0,Q):
    print( "X"+str(i)+":")
    x=int(input(""))
    print ("Y"+str(i)+":")
    y=int(input(""))
    print (Tree.calc(x,y))