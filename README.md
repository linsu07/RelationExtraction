# RelationExtraction
a tensorflow version for relation extraction of named entity  


## features
   * a tensorflow version under tf.estimator APIs, such a framework free me a lot from tedius codings, hope you also enjoy it
   * use DatasetApi to provide data input
   * both pcnn model and lstm model inside it, for both can share common data input, network ...
   * support both single relation and multi_relation extraction on one example
   * a new method of handling with "not available" relation
        
## overview    
   this program's idea came for paper [pcnn](http://aclweb.org/anthology/D/D15/D15-1203.pdf) and [bilstm](./Attention-BasedBidirectionalLongShort-TermMemoryNetworksfor RelationClassiﬁcatio.pdf)
   and refrenced implementation came from [OpenNRE](https://github.com/thunlp/OpenNRE);
   beyond these papers and implementation, I do some modifications needed.
   +  multi_relation extraction is needed, so multi_instance attetion layer is little diffrent from OpenNRE
   + "not available" relation is dominent situation, but  "not available" should not be considered as a relation, which will make neural network confused. instead, all 0 label for "not available" should be used
   + by careful coding, this program make a big success in our knowledage base project and have very good performance
   + sorry, I did not privide code for making tfrecords, in fact, I did it using spark. 
    
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