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

cpdef create_nets(num_net,num_input,num_layers,num_neurons,num_output):
    Nets = <NeuralNet *>malloc( num_net * cython.sizeof(NeuralNet) )
    for iNet in xrange(num_net):
        Nets[iNet].NumInputs = num_input
        Nets[iNet].NumHiddenLayers = num_layers
        Nets[iNet].NeuronsPerLayers = num_neurons
        Nets[iNet].NumOutputs = num_output
        Nets.Layers = <NeuronLayer *>malloc( Nets[iNet].NumHiddenLayers * cython.sizeof(NeuronLayer) )
        
        for iLayer in xrange(Nets[iNet].NumHiddenLayers):
            Nets[iNet].Layers[iLayer].NumNeurons = Nets[iNet].NeuronsPerLayers
            Nets[iNet].Layers[iLayer].Neurons = <Neuron *>malloc( Nets[iNet].NeuronsPerLayers * cython.sizeof(Neuron) )
            
            for iNeuron in xrange(Nets[iNet].NeuronsPerLayers):
                if iLayer == 0:
                    Nets[iNet].Layers[iLayer].Neurons[iNeuron].NumInputs = Nets[iNet].NumInputs
                else:
                    Nets[iNet].Layers[iLayer].Neurons[iNeuron].NumInputs = Nets[iNet].NeuronsPerLayers
                    
                Nets[iNet].Layers[iLayer].Neurons[iNeuron].Weights = <float *>malloc((Nets[iNet].Layers[iLayer].Neurons[iNeuron].NumInputs + 1) * cython.sizeof(float) )
        
        Nets[iNet].Input = <float *>malloc( Nets[iNet].NumInputs * cython.sizeof(float) )
        Nets[iNet].Output = <Neuron *>malloc( Nets[iNet].NumOutputs * cython.sizeof(Neuron) )
        
        for iOutput in xrange(Nets[iNet].NumOutputs):
            Nets[iNet].Output[iOutput].NumInputs = Nets[iNet].NeuronsPerLayers
            Nets[iNet].Output[iOutput].Weights = <float *>malloc(Nets[iNet].Output[iOutput].NumInputs * cython.sizeof(float) )
        
        #print(RAND_MAX)
        #print(1)
        #srand(1)
        #for i in xrange(100):
            #print(float(rand()) / RAND_MAX)
        #print(Nets[iNet].NumInputs)
        


cdef struct NeuralNet:
    int NumInputs
    int NumHiddenLayers
    int NeuronsPerLayers
    int NumOutputs
    NeuronLayer *Layers
    float *Input
    Neuron *Output
    
cdef struct NeuronLayer:
    int NumNeurons
    Neuron *Neurons
    
cdef struct Neuron:
    int NumInputs
    float *Weights
