import nnetga
from time import clock,sleep


agentNum = 20
nnetga.add_pop(1,0.7,0.005,0.3)
nnetga.add_agent(0,agentNum)
nnetga.add_net(0,2,1,2,1)

#chromo = [6,-6,-3,-6,5,-3,10,10,-5]
#chromo = [0.6,-0.6,-0.3,-0.6,0.5,-0.3,0.10,0.10,-0.5]
#nnetga.insert_chromo(0,10,chromo)
#nnetga.insert_chromo(0,11,chromo)
#nnetga.insert_chromo(0,12,chromo)
#nnetga.insert_chromo(0,13,chromo)
#nnetga.insert_chromo(0,14,chromo)
#nnetga.insert_chromo(0,15,chromo)
#nnetga.info_net(0,13,1)

Score = [0] * agentNum

test1 = [0,1]
test1 = [test1] * agentNum
test1 = [test1]

test2 = [0,0]
test2 = [test2] * agentNum
test2 = [test2]

test3 = [1,1]
test3 = [test3] * agentNum
test3 = [test3]

test4 = [1,0]
test4 = [test4] * agentNum
test4 = [test4]

for i in range(600):
    Score = [0] * agentNum
    a = nnetga.update(test1)
    b = nnetga.update(test2)
    c = nnetga.update(test3)
    d = nnetga.update(test4)
    
    #for i in range(agentNum):
        #print(a[0][i][0])
        #print(b[0][i][0])
        #print(c[0][i][0])
        #print(d[0][i][0])

    #'''
    for iAgent in range(agentNum):
        bonus = 0
        Score[iAgent] += (1 - (1 - a[0][iAgent][0]))*100
        Score[iAgent] += (1 - (b[0][iAgent][0]))*100
        Score[iAgent] += (1 - (c[0][iAgent][0]))*100
        Score[iAgent] += (1 - (1 - d[0][iAgent][0]))*100
        if Score[iAgent] > 320:
            #print(iAgent,"have",Score[iAgent],"at gen",i)
            bonus += 10
        if Score[iAgent] > 370:
            #print(iAgent,"have",Score[iAgent],"at gen",i)
            bonus += 30
        if Score[iAgent] > 390:
            #print(iAgent,"have",Score[iAgent],"at gen",i)
            bonus += 60
        if Score[iAgent] > 395:
            #print(iAgent,"have bonus of",bonus,"at gen",i)
            bonus += 100
        
    '''
    for iAgent in range(agentNum):
        if a[0][iAgent][0] > 0.5:
            Score[iAgent] += 1
        if b[0][iAgent][0] < 0.5:
            Score[iAgent] += 1
        if c[0][iAgent][0] < 0.5:
            Score[iAgent] += 1
        if d[0][iAgent][0] > 0.5:
            Score[iAgent] += 1
        if Score[iAgent] == 4:
            #print(iAgent,"have 4 at gen ",i)
            Score[iAgent] += 1
    '''   
    hiScore = max(Score)
    Average = sum(Score) / agentNum
    WinnerAgent = Score.index(hiScore)
    #print ("Agent:",Score.index(hiScore)," winner with:", hiScore)
    #print ("Average score:",Average)
    #nnetga.info_net(0,Score.index(hiScore),1)
    #print("Number 13 score:",Score[13])
    #print("----")
    nnetga.next_gen(0,Score)
    print(Average,end="")
    back = "\b" * len(str(Average))
    print(back,end="")
    
  
print ("Agent:",WinnerAgent," winner with:", hiScore)
print ("Average score:",Average)
nnetga.info_net(0,WinnerAgent,1)
print(a[0][WinnerAgent][0],"|",b[0][WinnerAgent][0],"|",c[0][WinnerAgent][0],"|",d[0][WinnerAgent][0])
print("----")

'''
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
'''
