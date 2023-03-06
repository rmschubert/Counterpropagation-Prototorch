## About
This is the implementation of Counterpropagation Models based on [Prototorch](https://github.com/si-cim/prototorch) by [SICIM](https://www.institute.hs-mittweida.de/webs/sicim/).

The implemented model is mainly based on the report by [Villmann, Schubert and Kaden (2021)](https://www.techfak.uni-bielefeld.de/~fschleif/mlr/mlr_01_2021.pdf) and the paper [Kaden et. al. (2021)](https://www.researchgate.net/profile/Thomas-Villmann/publication/355250021_The_LVQ-based_Counter_Propagation_Network_--_an_Interpretable_Information_Bottleneck_Approach/links/61adfdd6ca2d401f27cd9b00/The-LVQ-based-Counter-Propagation-Network--an-Interpretable-Information-Bottleneck-Approach.pdf). However, the original Counterpropagation model can be found in [Hecht-Nielsen (1987)](https://www.semanticscholar.org/paper/Counterpropagation-networks.-Hecht-Nielsen/e3ef03fcea4a0a6cc1809d3cee98fbe6148f8714).

Note, that the original Model is not implemented here (i.e. non-differentiable SOM initialization + perceptron-layer), since this implementation is focused on influencing the response by the subsequent supervised layer.

However, a simple approach to implement the Hecht-Nielsen model is to initialize a SOM on the data and use the assignments to create the subsequent perceptron layer.

## Contact
bugs, fixes, ideas etc. can be send to [trebuhcsynnor@gmail.com](mailto:trebuhcsynnor@gmail.com) 