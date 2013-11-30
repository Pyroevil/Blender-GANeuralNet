import nnetga
from time import clock

agentNum = 4000
stime = clock()
nnetga.add_pop()
print ("Pop added in: ",clock() - stime,"sec")
stime = clock()
nnetga.add_agent(0,agentNum)
print ("Agent added in: ",clock() - stime,"sec")
stime = clock()
nnetga.add_net(0,2,1,2,1)
print ("net added in: ",clock() - stime,"sec")
nnetga.info_net(0,0,2)

c = [6,-6,-3,-6,5,-3,10,10,-5]
stime = clock()
nnetga.insert_chromo(0,c)
print ("Insert Chromo in: ",clock() - stime,"sec")

a = [0,1]
a = [a] * agentNum
a = [a]
#print(a)
stime = clock()
b = nnetga.update(a)
#print("0 1:",b)
print ("net updated in: ",clock() - stime,"sec")

d = [0] * agentNum
for i in range(agentNum):
    d[i] = i
stime = clock()    
nnetga.next_gen(0,d)
print ("new generation in: ",clock() - stime,"sec")
nnetga.info_net(0,0,2)

#for i in range(agentNum):
    #nnetga.info_net(0,i,1)

