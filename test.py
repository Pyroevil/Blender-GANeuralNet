import nnetga
nnetga.add_pop()
nnetga.add_agent(0,1)
nnetga.add_net(0,2,1,2,1)

a = (((1,1),),)
b = nnetga.update(a)
print("Init:",b)

c = [1,-1,1,-1,1,1,1,10,1]
nnetga.insert_chromo(0,c)

a = (((1,1),),)
b = nnetga.update(a)
print("1 1:",b)

a = (((0,0),),)
b = nnetga.update(a)
print("0 0:",b)

a = (((1,0),),)
b = nnetga.update(a)
print("1 0:",b)

a = (((0,1),),)
b = nnetga.update(a)
print("0 1:",b)
