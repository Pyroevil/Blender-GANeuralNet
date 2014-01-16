#cython: profile=False
#cython: boundscheck=False
#cython: cdivision=True

# NOTE: order of slow fonction to be optimize/multithreaded: kdtreesearching , kdtreecreating , linksolving 

cimport cython
from time import clock,time
from math import tanh
from cython.parallel import parallel , prange , threadid
from libc.stdlib cimport malloc , realloc, free , rand , srand, abs

cdef extern from *:
    int INT_MAX
    float FLT_MAX
    float RAND_MAX

cdef World *Worlds = <World *>malloc( 1 * cython.sizeof(World) )
Worlds.NumPops = 0
Worlds.Pops = NULL
seed = int(int(time()) % (RAND_MAX + 1.0))
#print("seed:",seed)
srand(seed)
cdef float randMax = 36
cdef float randMin = randMax / 2
'''
cdef int i
for i in range(64000):
    if sigmoid(float(i)/1000) == 1.0:
        print"result:",(float(i)/1000),"=",sigmoid(float(i)/1000)
        break
    print (float(i)/1000),":",sigmoid(float(i)/1000)
'''

cpdef info():
    global Worlds
    
    print "Info:"
    print "  Numbers of population:",Worlds.NumPops
    print ""
    return

cpdef info_pop():
    global Worlds
    
    print "Population Info:"
    
    if Worlds.NumPops == 0:
        print "  No population created"
        return False
        
    print "  Numbers of population:",Worlds.NumPops
    for iPop in xrange(Worlds.NumPops):
        print "  Population index:",Worlds.Pops[iPop].Index
        print "    -number of agents:",Worlds.Pops[iPop].NumAgents
        print "    -number of generation:",Worlds.Pops[iPop].Generation
        print "    -crossover rate:",Worlds.Pops[iPop].CrossRate
        print "    -mutation rate:",Worlds.Pops[iPop].MutateRate
        print "    -mutation max range:",Worlds.Pops[iPop].MutateMax
        if Worlds.Pops[iPop].NumAgents > 0:
            if Worlds.Pops[iPop].Agents.Net != NULL:
                 print "    -neural network: Yes"
                 print "      -Numbers of Inputs:",Worlds.Pops[iPop].Agents.Net.NumInputs
                 print "      -Numbers of HiddenLayers:",Worlds.Pops[iPop].Agents.Net.NumLayers
                 print "      -Numbers of Neurons per Layers:",Worlds.Pops[iPop].Agents.Net.NeuronsPerLayers
                 print "      -Numbers of total Neurons:",Worlds.Pops[iPop].Agents.Net.NeuronsPerLayers * Worlds.Pops[iPop].Agents.Net.NumLayers + Worlds.Pops[iPop].Agents.Net.NumOutputs
                 print "      -Numbers of Outputs::",Worlds.Pops[iPop].Agents.Net.NumOutputs
            else:
                print "    -neural network: No"
        else:
            print "    -neural network: No"
        
    print ""
    return
    

cpdef info_net(int iPop,int iAgent,int level = 0):
    global Worlds
    
    cdef int iLayer = 0
    cdef int iNeuron = 0
    cdef int iWeight = 0
    cdef int iChromo = 0
    cdef int iInput = 0
    #cdef int increment = 0
    
    #print "level:",level
    
    pyChromo = [0] * Worlds.Pops[iPop].Agents[iAgent].NumChromo
    for iChromo in xrange(Worlds.Pops[iPop].Agents[iAgent].NumChromo):
        pyChromo[iChromo] = Worlds.Pops[iPop].Agents[iAgent].Chromo[iChromo]
    pyInput = [0] * Worlds.Pops[iPop].Agents[iAgent].Net.NumInputs
    for iInput in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.NumInputs):
        pyInput[iInput] = Worlds.Pops[iPop].Agents[iAgent].Net.Input[iInput]
    
    if level >= 0:
        print "Agent:",Worlds.Pops[iPop].Agents[iAgent].Index,"  NumChromo:",Worlds.Pops[iPop].Agents[iAgent].NumChromo
    if level >= 1:
        print " Chromo:", pyChromo
        print " Inputs", pyInput
    if level >= 2:
        for iLayer in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.NumLayers):
            print "  Layer:",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Index
            for iNeuron in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].NumNeurons):
                print "   Neuron:",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index
                for iWeight in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs):
                    print "    Weights:",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]


    
cpdef add_pop(int numPopAdd = 1,float crossover = 0.7,float mutation = 0.1,mutatemax = 0.3):
    global Worlds
    
    cdef int iPop = 0
    cdef int i = 0
    
    if numPopAdd == 0:
        return False
    
    if Worlds.NumPops == 0:
        Worlds.Pops = <Pop *>malloc( (Worlds.NumPops + numPopAdd) * cython.sizeof(Pop) )
    else:
        Worlds.Pops = <Pop *>realloc(Worlds.Pops, ( Worlds.NumPops + numPopAdd ) * cython.sizeof(Pop) )
        
    for i in xrange(numPopAdd):
        iPop = i + Worlds.NumPops
        #print(iPop)
        Worlds.Pops[iPop].Index = iPop
        Worlds.Pops[iPop].NumAgents = 0
        Worlds.Pops[iPop].Agents = NULL
        Worlds.Pops[iPop].Generation = 0
        Worlds.Pops[iPop].CrossRate = crossover
        Worlds.Pops[iPop].MutateRate = mutation
        Worlds.Pops[iPop].MutateMax = mutatemax
    
    Worlds.NumPops += numPopAdd
           
    return True  
    
    
cpdef add_agent(int popIndex = 0, int numAgents = 1):
    global Worlds
    
    cdef int iAgent = 0
    
    if popIndex > (Worlds.NumPops - 1):
        print('ERROR: popIndex out of range')
        return False
    
    Worlds.Pops[popIndex].Agents = <Agent *>malloc( numAgents * cython.sizeof(Agent) )
    Worlds.Pops[popIndex].NumAgents = numAgents
    
    for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
        Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent
        Worlds.Pops[popIndex].Agents[iAgent].Net = NULL
        Worlds.Pops[popIndex].Agents[iAgent].NumChromo = 0
        Worlds.Pops[popIndex].Agents[iAgent].Chromo = NULL
        Worlds.Pops[popIndex].Agents[iAgent].Score = 0
        Worlds.Pops[popIndex].Agents[iAgent].PosLow = 0
        Worlds.Pops[popIndex].Agents[iAgent].PosHi = 0

    return True
    
    
cpdef add_net(int popIndex,int num_input,int num_layers,int num_neurons,int num_output):
    global Worlds
    global srand,randMax,randMin
    
    cdef int iAgent = 0
    cdef int iLayer = 0
    cdef int iNeuron = 0
    cdef int iOutput = 0
    cdef int iWeight = 0
    cdef int LastLayerIndex = 0
    cdef int iChromo = 0
    cdef int iInput = 0
    #cdef int increment = 0
    
    if popIndex > (Worlds.NumPops - 1):
        print('ERROR: popIndex out of range')
        return False

    for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
        iChromo = 0
        Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent
        if num_layers >= 1:
            Worlds.Pops[popIndex].Agents[iAgent].NumChromo = ((num_input + 1) * num_neurons) + ((num_layers - 1) * ((num_neurons + 1) * num_neurons)) + (num_output * (num_neurons + 1))
        else:
            Worlds.Pops[popIndex].Agents[iAgent].NumChromo = num_input + 1
        Worlds.Pops[popIndex].Agents[iAgent].Chromo = <float *>malloc( Worlds.Pops[popIndex].Agents[iAgent].NumChromo * cython.sizeof(float) )
        Worlds.Pops[popIndex].Agents[iAgent].Net = <NeuralNet *>malloc( 1 * cython.sizeof(NeuralNet) )
        Worlds.Pops[popIndex].Agents[iAgent].Net.Index = 0
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs = num_input
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers = num_layers + 1
        Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers = num_neurons
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs = num_output
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers = <NeuronLayer *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers * cython.sizeof(NeuronLayer) )
        
        for iLayer in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1):
        
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Index = iLayer
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons = <Neuron *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers * cython.sizeof(Neuron) )
            
            for iNeuron in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers):
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index = iNeuron
                if iLayer == 0:
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs + 1
                else:
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1
                
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output = 0
                    
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights = <float *>malloc((Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs) * cython.sizeof(float) )
                
                for iWeight in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs):
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = clamped_rand(randMin)
                    Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]
                    iChromo += 1
        
        Worlds.Pops[popIndex].Agents[iAgent].Net.Input = <float *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs * cython.sizeof(float) )
        for iInput in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs):
            Worlds.Pops[popIndex].Agents[iAgent].Net.Input[iInput] = 333
        Worlds.Pops[popIndex].Agents[iAgent].Net.Output = <Neuron *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs * cython.sizeof(Neuron) )
        
        if Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers >= 2:
            LastLayerIndex = Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1
        else:
            LastLayerIndex = 0
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons = Worlds.Pops[popIndex].Agents[iAgent].Net.Output
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Index = Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].NumNeurons = Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs
        
        for iOutput in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs):
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Index = iOutput
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Output = 0
            if Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers >= 2:
                Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1
            else:
                Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs + 1
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Weights = <float *>malloc(Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs * cython.sizeof(float) )
            
            for iWeight in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].NumInputs):
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight] = clamped_rand(randMin)
                    Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight]
                    iChromo += 1

    
    return True


cpdef update(Data):
    global Worlds
    
    #print Data
    cdef int NumData = 0
    cdef int iData = 0
    cdef int iPop = 0
    cdef int iAgent = 0
    cdef int iLayer = 0
    cdef int iNeuron = 0
    cdef int iWeight = 0
    cdef float sum = 0.0
    cdef float Bias = -1.0
    
    for iPop in xrange(Worlds.NumPops):
        #print iPop
        for iAgent in xrange(Worlds.Pops[iPop].NumAgents):
            #print iAgent
            NumData = len(Data[iPop][iAgent])
            #print NumData
            for iData in xrange(NumData):
                #print iData,":",Data[iPop][iAgent][iData]
                Worlds.Pops[iPop].Agents[iAgent].Net.Input[iData] = Data[iPop][iAgent][iData]
    #with nogil:
    for iPop in xrange(Worlds.NumPops):
        for iAgent in xrange(Worlds.Pops[iPop].NumAgents): #,schedule='dynamic',chunksize=5):
            #print "Agent loop", iAgent
            for iLayer in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.NumLayers):
                #print "Layer loop", iLayer
                for iNeuron in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].NumNeurons):
                    #print "Neuron loop", iNeuron
                    sum = 0.0
                    for iWeight in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs - 1):
                        #print "Weight loop", iWeight
                        if iLayer == 0:
                            #print "(1)Layer",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Index,"  Neurons",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index,"  ",Worlds.Pops[iPop].Agents[iAgent].Net.Input[iWeight]," x ",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]," = ",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Worlds.Pops[iPop].Agents[iAgent].Net.Input[iWeight]
                            sum = sum + ( Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Worlds.Pops[iPop].Agents[iAgent].Net.Input[iWeight] )
                        else:
                            #print "(2)Layer",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Index,"  Neurons",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Neurons[iWeight].Index,"  ",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Neurons[iWeight].Output," x ",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]," = ",( Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Neurons[iWeight].Output )
                            sum = sum + ( Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] * Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer - 1].Neurons[iWeight].Output )
                            
                    sum = sum + ( Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight + 1] * Bias)
                    Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output = sigmoid(sum)
                    #print"sum neuron",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index,":",sum," sigmoid:",Worlds.Pops[iPop].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Output
    
    
    
    ExportData = [1] * Worlds.NumPops
    OutputData = []
    for iPop in xrange(Worlds.NumPops):
        ExportData[iPop] = [1] * Worlds.Pops[iPop].NumAgents
        for iAgent in xrange(Worlds.Pops[iPop].NumAgents):
            for iOutput in xrange(Worlds.Pops[iPop].Agents[iAgent].Net.NumOutputs):
                OutputData.append(Worlds.Pops[iPop].Agents[iAgent].Net.Output[iOutput].Output)
                
            ExportData[iPop][iAgent] = OutputData
            OutputData = []
        
    return ExportData


cpdef insert_chromo(popIndex,agentIndex,Data):
    global Worlds
    
    cdef int NumData = len(Data)
    cdef int iData = 0
    cdef int iAgent = 0
    cdef int iLayer = 0
    cdef int iNeuron = 0
    cdef int iWeight = 0
    cdef int iChromo = 0
    
    if agentIndex == -1:
        for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
            iChromo = 0
            for iLayer in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers):
                for iNeuron in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons):
                    for iWeight in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs):
                        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Data[iChromo]
                        Worlds.Pops[popIndex].Agents[iAgent].Chromo[iChromo] = Data[iChromo]
                        iChromo += 1
    else:
        iChromo = 0
        for iLayer in xrange(Worlds.Pops[popIndex].Agents[agentIndex].Net.NumLayers):
            for iNeuron in xrange(Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].NumNeurons):
                for iWeight in xrange(Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].Neurons[iNeuron].NumInputs):
                    Worlds.Pops[popIndex].Agents[agentIndex].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Data[iChromo]
                    Worlds.Pops[popIndex].Agents[agentIndex].Chromo[iChromo] = Data[iChromo]
                    iChromo += 1
                        
    return True
    
    
cpdef get_chromo(int PopIndex,int agentIndex):
    pyChromo = []
    for iChromo in xrange(Worlds.Pops[PopIndex].Agents[agentIndex].NumChromo):
        pyChromo.append(Worlds.Pops[PopIndex].Agents[agentIndex].Chromo[iChromo])
    
    return pyChromo
    
    
cpdef float next_gen(int PopIndex,Data,int crossOption = 1,int EliteNum = 0,int WorstNum = 0):
    global Worlds
    global srand,randMin,randMax
    
    cdef float TotalScore = 0
    cdef float PreviousPos = 0
    cdef int iAgent = 0
    cdef int iNeuron = 0
    cdef int iWeight = 0
    cdef int iChromo = 0
    cdef int iLovers = 0
    cdef int iElite = 0
    cdef int Mom = 0
    cdef int Dad = 0
    cdef Agent *Baby = <Agent *>malloc((Worlds.Pops[PopIndex].NumAgents + 1) * cython.sizeof(Agent) )
    cdef int *Elite = <int *>malloc( EliteNum * cython.sizeof(int) )
    cdef int *Worst = <int *>malloc( WorstNum * cython.sizeof(int) )
    cdef float CrossRate = 0
    cdef float Roulette = 0
    cdef float CrossPoint = 0
    cdef int iBaby = 0
    cdef float MutateChromo = 0
    cdef float MutateChance = 0
    
    for iBaby in xrange(Worlds.Pops[PopIndex].NumAgents):
        Baby[iBaby].Chromo = <float *>malloc(Worlds.Pops[PopIndex].Agents[0].NumChromo * cython.sizeof(float) )
        for iChromo in xrange(Worlds.Pops[PopIndex].Agents[0].NumChromo):
            Baby[iBaby].Chromo[iChromo] = 666.666
    
    CopyData = list(Data)
    DataSorted = []
    ScoreElite = []
    DataSorted = sorted(CopyData)
    ScoreElite = DataSorted[-EliteNum:]
    ScoreWorst = DataSorted[:WorstNum]
    #print(ScoreElite)
    for iElite in xrange(EliteNum):
        Elite[iElite] = CopyData.index(ScoreElite[iElite])
        CopyData[Elite[iElite]] += (float(rand()) / (RAND_MAX + 1.0)) * 0.0000001
        #print("Elite:",Elite[iElite],":",Data[Elite[iElite]])
    
    for iWorst in xrange(WorstNum):
        Worst[iWorst] = CopyData.index(ScoreWorst[iWorst])
        CopyData[Worst[iWorst]] += (float(rand()) / (RAND_MAX + 1.0)) * 0.0000001
        Worlds.Pops[PopIndex].Agents[Worst[iWorst]].Score = 0.0
        #print("Worst:",Worst[iWorst],":",Data[Worst[iWorst]])
    
    for iAgent in xrange(Worlds.Pops[PopIndex].NumAgents):
        if arraysearch(iAgent,Worst,WorstNum) == -1:
            Worlds.Pops[PopIndex].Agents[iAgent].Score = Data[iAgent] + 0.0000001
        TotalScore += Worlds.Pops[PopIndex].Agents[iAgent].Score
        
    for iAgent in xrange(Worlds.Pops[PopIndex].NumAgents):
        Worlds.Pops[PopIndex].Agents[iAgent].PosLow = PreviousPos
        Worlds.Pops[PopIndex].Agents[iAgent].PosHi = (Worlds.Pops[PopIndex].Agents[iAgent].Score / TotalScore) + PreviousPos
        PreviousPos = Worlds.Pops[PopIndex].Agents[iAgent].PosHi
        #print "Agent:",Worlds.Pops[PopIndex].Agents[iAgent].Index," Low:",Worlds.Pops[PopIndex].Agents[iAgent].PosLow," Hi:",Worlds.Pops[PopIndex].Agents[iAgent].PosHi
    
    iBaby = 0
    while iBaby < (Worlds.Pops[PopIndex].NumAgents):
        iLovers = 0
        while iLovers < 2:
            Roulette = (float(rand()) / (RAND_MAX + 1.0))
            #print "Roulette:",Roulette
            for iAgent in xrange(Worlds.Pops[PopIndex].NumAgents):
                if Roulette > Worlds.Pops[PopIndex].Agents[iAgent].PosLow and Roulette <= Worlds.Pops[PopIndex].Agents[iAgent].PosHi:
                    #print "hit:",Worlds.Pops[PopIndex].Agents[iAgent].Index
                    if arraysearch(Worlds.Pops[PopIndex].Agents[iAgent].Index,Worst,WorstNum) == 1:
                        print "Hit Worse:",Worlds.Pops[PopIndex].Agents[iAgent].Index
                    if iLovers == 0:
                        Mom = Worlds.Pops[PopIndex].Agents[iAgent].Index
                        #iLovers += 1
                    if iLovers == 1:# and Worlds.Pops[PopIndex].Agents[iAgent].Index != Mom:
                        Dad = Worlds.Pops[PopIndex].Agents[iAgent].Index
                        #iLovers += 1
                    iLovers += 1
        
        CrossRate = (float(rand()) / (RAND_MAX + 1.0))
        CrossPoint = rand() % (Worlds.Pops[PopIndex].Agents[Mom].NumChromo)
        #print "CrossPoint Rand:",CrossPoint
        #print "CrossRate Rand:",CrossRate
        #print "CrossRate:",Worlds.Pops[PopIndex].CrossRate
        if crossOption == 0:
            if CrossRate <= Worlds.Pops[PopIndex].CrossRate:
                #print "Crossing"
                for iChromo in xrange(Worlds.Pops[PopIndex].Agents[Mom].NumChromo):
                    if iChromo < CrossPoint:
                        Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo]
                        Baby[iBaby+1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo]
                    else:
                        Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo]
                        Baby[iBaby+1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo]
            else:
                #print "Stay the same"
                for iChromo in xrange(Worlds.Pops[PopIndex].Agents[Mom].NumChromo):
                    Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo]
                    Baby[iBaby+1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo]
        elif crossOption >= 1:
            for iChromo in xrange(Worlds.Pops[PopIndex].Agents[Mom].NumChromo):
                CrossPoint = (float(rand()) / (RAND_MAX + 1.0))
                if CrossPoint <= Worlds.Pops[PopIndex].CrossRate:
                    Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo]
                    Baby[iBaby+1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo]
                else:
                    Baby[iBaby].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Dad].Chromo[iChromo]
                    Baby[iBaby+1].Chromo[iChromo] = Worlds.Pops[PopIndex].Agents[Mom].Chromo[iChromo]
           
        iBaby += 2
    
    for iAgent in xrange(Worlds.Pops[PopIndex].NumAgents):
        Worlds.Pops[PopIndex].Agents[iAgent].PosLow = 0
        Worlds.Pops[PopIndex].Agents[iAgent].PosHi = 0
        Worlds.Pops[PopIndex].Agents[iAgent].Score = 0
        #if arraysearch(iAgent,Elite,EliteNum) >= 0:
            #print("Elite exist:",arraysearch(iAgent,Elite,EliteNum))
        if arraysearch(iAgent,Elite,EliteNum) == -1:
            for iChromo in xrange(Worlds.Pops[PopIndex].Agents[iAgent].NumChromo):
                Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = Baby[iAgent].Chromo[iChromo]
                MutateChance = (float(rand()) / (RAND_MAX + 1.0))
                if MutateChance < Worlds.Pops[PopIndex].MutateRate:
                    #if mutateOption == 0:
                    #MutateChromo = ((float(rand()) / (RAND_MAX + 1.0) * randMax) - randMin)
                    #Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] = MutateChromo
                    #else:
                    MutateChromo = clamped_rand(randMin) * Worlds.Pops[PopIndex].MutateMax
                    Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo] += MutateChromo
                    #print "Agent:",iAgent,"Mutate Chromo ",iChromo," at:",MutateChromo
            iChromo = 0
            for iLayer in xrange(Worlds.Pops[PopIndex].Agents[iAgent].Net.NumLayers):
                for iNeuron in xrange(Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons):
                    for iWeight in xrange(Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs):
                        Worlds.Pops[PopIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = Worlds.Pops[PopIndex].Agents[iAgent].Chromo[iChromo]
                        iChromo += 1
                        
        free(Baby[iAgent].Chromo)
    free(Baby)
    free(Elite)
    Worlds.Pops[PopIndex].Generation += 1
    return Worlds.Pops[PopIndex].Generation
    
cdef float clamped_rand(float range)nogil:
    return ((float(rand()) / (RAND_MAX + 1.0)) * range) - ((float(rand()) / (RAND_MAX + 1.0)) * range)

cdef float sigmoid(float input)nogil:
    #print "sigmoid function"
    return 1 / (1 + ( 2.7186**(-input/1.0)))
    #return tanh(input)
 
 
cdef int arraysearch(int element,int *array,int len)nogil:
    cdef int i = 0
    #printdb(939)
    for i in xrange(len):
        if element == array[i]:
            return i
    #printdb(943)
    return -1
    
    
cdef struct World:
    int NumPops
    Pop *Pops
        
        
cdef struct Pop:
    int Index
    int NumAgents
    Agent *Agents
    int Generation
    float CrossRate
    float MutateRate
    float MutateMax

    
cdef struct Agent:
    int Index
    NeuralNet *Net
    int NumChromo
    float *Chromo
    float Score
    float PosLow
    float PosHi

cdef struct NeuralNet:
    int Index
    int NumInputs
    int NumLayers
    int NeuronsPerLayers
    int NumOutputs
    NeuronLayer *Layers
    float *Input
    Neuron *Output

    
cdef struct NeuronLayer:
    int Index
    int NumNeurons
    Neuron *Neurons
 
 
cdef struct Neuron:
    int Index
    int NumInputs
    float Output
    float *Weights
    
