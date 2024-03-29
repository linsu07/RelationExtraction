# RelationExtraction
   RelationExtraction is a TensorFlow implementation for extracting relations between named entities.


## features
   + Implemented using TensorFlow's tf.estimator APIs, which reduces the need for tedious coding.
   + Utilizes the Dataset API for efficient data input.
   + Includes both PCNN (Piecewise Convolutional Neural Network) and LSTM (Long Short-Term Memory) models.
   + Supports both single relation and multi-relation extraction on a single example.
   + Introduces a novel method for handling "not available" relations.
        
## overview    
   The idea behind this program is inspired by the papers [pcnn](http://aclweb.org/anthology/D/D15/D15-1203.pdf) and [bilstm](./Attention-BasedBidirectionalLongShort-TermMemoryNetworksfor RelationClassiﬁcatio.pdf). The reference implementation used for this project is based on [OpenNRE](https://github.com/thunlp/OpenNRE). However, certain modifications were made to meet specific requirements:

   + Multi-relation extraction is supported, resulting in a slightly different multi-instance attention layer compared to OpenNRE.
   + The dominant situation of "not available" relations is addressed by treating them as non-relations (all 0 labels), preventing confusion in the neural network.
   + Through careful coding and optimization, this program has achieved great success in our knowledge base project and has demonstrated excellent performance.
   + Please note that the code for generating tfrecords is not provided, as it was done using Spark.    
## Requirements
  * Python>=3.5
  * TensorFlow>=1.10
   
## Usage
    I use a parameter to switch models 
    --ner_procedure birnn,mi_att  means bilstm model is working
    --ner_procedure pcnn,mi_att   means pcnn model is working    
    you can assemble both to get better performance
     
## todo

   1.add elmo to replace embedding layer
   2.add bert to replace embedding layer 

## Test Results
I also use NYT10 Dataset for comparation

|      Model     |steps   |        F1      | precision | recall          |
|:--------------:|:------:|--------------:|:---------:|:---------------:|
|       pcnn     |  8000  |      0.52     |   0.63    |    0.41         |
|       birnn    |  18000 |      0.59     |   0.606   |    0.575        |