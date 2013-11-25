import nnetga
from time import clock

agentNum = 160
stime = clock()
nnetga.add_pop()
print ("Pop added in: ",clock() - stime,"sec")
stime = clock()
nnetga.add_agent(0,agentNum)
print ("Agent added in: ",clock() - stime,"sec")
stime = clock()
nnetga.add_net(0,40,2,60,16)
print ("net added in: ",clock() - stime,"sec")
nnetga.info_net(0,0,0)

#c = [6,-6,-3,-6,5,-3,10,10,-5]
c = [0.5] * 7096
stime = clock()
nnetga.insert_chromo(0,c)
print ("Insert Chromo in: ",clock() - stime,"sec")
print("Chromo inserted")

a = [0,1]
a = [a] * agentNum
a = [a]
#print(a)
stime = clock()
b = nnetga.update(a)
#print("0 1:",b)
print ("net updated in: ",clock() - stime,"sec")


#nnetga.info_net(0,0)

