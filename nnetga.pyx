#cython: profile=False
#cython: boundscheck=False
#cython: cdivision=True

# NOTE: order of slow fonction to be optimize/multithreaded: kdtreesearching , kdtreecreating , linksolving 

cimport cython
from time import clock,time
from cython.parallel import parallel , prange , threadid
from libc.stdlib cimport malloc , realloc, free , rand , srand, abs

cdef extern from *:
    int INT_MAX
    float FLT_MAX
    float RAND_MAX

cdef World *Worlds = <World *>malloc( 1 * cython.sizeof(World) )
Worlds.NumPops = 0
Worlds.Pops = NULL
srand(int(time()))

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
    
cpdef add_pop(int numPopAdd = 1,float crossover = 0.7,float mutation = 0.05):
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
        iPop += i + Worlds.NumPops
        Worlds.Pops[iPop].Index = iPop
        Worlds.Pops[iPop].NumAgents = 0
        Worlds.Pops[iPop].Agents = NULL
        Worlds.Pops[iPop].Generation = 0
        Worlds.Pops[iPop].CrossRate = crossover
        Worlds.Pops[iPop].MutateRate = mutation
    
    Worlds.NumPops += numPopAdd
           
    return True  
    
    
cpdef add_agent(int popIndex = 0, int numAgents = 1):
    global Worlds
    
    cdef int iAgent = 0
    
    if popIndex > (Worlds.NumPops - 1):
        print('ERROR: popIndex out of range')
        return False
    
    Worlds.Pops[popIndex].Agents = <Agent *>malloc( numAgents * cython.sizeof(Agent) )
    Worlds.Pops[popIndex].NumAgents += numAgents
    
    for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
        Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent
        Worlds.Pops[popIndex].Agents[iAgent].Net = NULL
        Worlds.Pops[popIndex].Agents[iAgent].NumChromo = 0
        Worlds.Pops[popIndex].Agents[iAgent].Chromo = NULL

    return True
    
    
cpdef add_net(int popIndex,int num_input,int num_layers,int num_neurons,int num_output):
    global Worlds
    
    cdef int iAgent = 0
    cdef int iLayer = 0
    cdef int iNeuron = 0
    cdef int iOutput = 0
    cdef int iWeight = 0
    cdef int LastLayerIndex = 0
    #cdef int increment = 0
    srand(1)
    
    if popIndex > (Worlds.NumPops - 1):
        print('ERROR: popIndex out of range')
        return False

    for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
        
        Worlds.Pops[popIndex].Agents[iAgent].Index = iAgent
        Worlds.Pops[popIndex].Agents[iAgent].NumChromo = (num_layers * (num_neurons + 1)) + (num_output + 1)
        Worlds.Pops[popIndex].Agents[iAgent].Net = <NeuralNet *>malloc( 1 * cython.sizeof(NeuralNet) )
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
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight] = (float(rand()) / RAND_MAX * 2) - 1
                    #increment += 1
        
        Worlds.Pops[popIndex].Agents[iAgent].Net.Input = <float *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs * cython.sizeof(float) )
        Worlds.Pops[popIndex].Agents[iAgent].Net.Output = <Neuron *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs * cython.sizeof(Neuron) )
        
        LastLayerIndex = Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons = Worlds.Pops[popIndex].Agents[iAgent].Net.Output
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Index = Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers - 1
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].NumNeurons = Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs
        
        for iOutput in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs):
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Index = iOutput
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Output = 0
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers + 1
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Weights = <float *>malloc(Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs * cython.sizeof(float) )
            
            for iWeight in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].NumInputs):
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[LastLayerIndex].Neurons[iOutput].Weights[iWeight] = (float(rand()) / RAND_MAX * 2) - 1
                    #increment += 1
                    
        
                    
                    
    for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
        print "A:",Worlds.Pops[popIndex].Agents[iAgent].Index,"  NumChromo:",Worlds.Pops[popIndex].Agents[iAgent].NumChromo
        for iLayer in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumLayers):
            print " L",Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Index
            for iNeuron in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons):
                print "  N",Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Index
                for iWeight in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs):
                    print "    ",Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights[iWeight]
    #print(RAND_MAX)
    #print(1)
    #srand(1)
    #for i in xrange(100):
        #print((float(rand()) / RAND_MAX * 2) - 1)
    #print(Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs)
        
    return True

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

    
cdef struct Agent:
    int Index
    NeuralNet *Net
    int NumChromo
    float *Chromo

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
