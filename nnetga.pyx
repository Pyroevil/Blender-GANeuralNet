#cython: profile=False
#cython: boundscheck=False
#cython: cdivision=True

# NOTE: order of slow fonction to be optimize/multithreaded: kdtreesearching , kdtreecreating , linksolving 

cimport cython
from time import clock
from cython.parallel import parallel , prange , threadid
from libc.stdlib cimport malloc , realloc, free , rand , srand, abs

cdef extern from *:
    int INT_MAX
    float FLT_MAX
    float RAND_MAX

cdef World *Worlds = <World *>malloc( 1 * cython.sizeof(World) )
Worlds.NumPops = 0
Worlds.Pops = NULL

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
                 print "      -Numbers of HiddenLayers:",Worlds.Pops[iPop].Agents.Net.NumHiddenLayers
                 print "      -Numbers of Neurons per Layers:",Worlds.Pops[iPop].Agents.Net.NeuronsPerLayers
                 print "      -Numbers of total Neurons:",Worlds.Pops[iPop].Agents.Net.NeuronsPerLayers * Worlds.Pops[iPop].Agents.Net.NumHiddenLayers
                 print "      -Numbers of Outputs::",Worlds.Pops[iPop].Agents.Net.NumOutputs
            else:
                print "    -neural network: No"
        else:
            print "    -neural network: No"
        
    print ""
    return
    
cpdef add_pop(numPopAdd = 1,crossover = 0.7, mutation = 0.05):
    global Worlds
    
    if numPopAdd == 0:
        return False
    
    if Worlds.NumPops == 0:
        Worlds.Pops = <Pop *>malloc( (Worlds.NumPops + numPopAdd) * cython.sizeof(Pop) )
    else:
        Worlds.Pops = <Pop *>realloc(Worlds.Pops, ( Worlds.NumPops + numPopAdd ) * cython.sizeof(Pop) )
        
    iPop = 0
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
    
    
cpdef add_agent(popIndex = 0, numAgents = 1):
    global Worlds
    
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
    
    
cpdef add_net(popIndex,num_input,num_layers,num_neurons,num_output):
    global Worlds
    
    if popIndex > (Worlds.NumPops - 1):
        print('ERROR: popIndex out of range')
        return False

    for iAgent in xrange(Worlds.Pops[popIndex].NumAgents):
        Worlds.Pops[popIndex].Agents[iAgent].Net = <NeuralNet *>malloc( 1 * cython.sizeof(NeuralNet) )
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs = num_input
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumHiddenLayers = num_layers
        Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers = num_neurons
        Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs = num_output
        Worlds.Pops[popIndex].Agents[iAgent].Net.Layers = <NeuronLayer *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumHiddenLayers * cython.sizeof(NeuronLayer) )
        
        for iLayer in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumHiddenLayers):
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].NumNeurons = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers
            Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons = <Neuron *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers * cython.sizeof(Neuron) )
            
            for iNeuron in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers):
                if iLayer == 0:
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs
                else:
                    Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers
                    
                Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].Weights = <float *>malloc((Worlds.Pops[popIndex].Agents[iAgent].Net.Layers[iLayer].Neurons[iNeuron].NumInputs + 1) * cython.sizeof(float) )
        
        Worlds.Pops[popIndex].Agents[iAgent].Net.Input = <float *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumInputs * cython.sizeof(float) )
        Worlds.Pops[popIndex].Agents[iAgent].Net.Output = <Neuron *>malloc( Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs * cython.sizeof(Neuron) )
        
        for iOutput in xrange(Worlds.Pops[popIndex].Agents[iAgent].Net.NumOutputs):
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs = Worlds.Pops[popIndex].Agents[iAgent].Net.NeuronsPerLayers
            Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].Weights = <float *>malloc(Worlds.Pops[popIndex].Agents[iAgent].Net.Output[iOutput].NumInputs * cython.sizeof(float) )
        
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
    int NumHiddenLayers
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
