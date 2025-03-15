# ML-CFD
Here is the Abstract and title of the paper
Verification and validation of optimal self-supervised physics-informed neural

network for heat transfer in miniature heat sinks
Reza Pirayeshshirazinezhad1

, Mahyar Pourghasemi3

, Nima Fathi1,2*
1Department of Marine Engineering Technology, Texas A&M University, Galveston, TX 77553
2Department of J. Mike Walker’66 Mechanical Engineering, Texas A&M University, College Station, TX 77840
3Department of Mechanical Engineering, Western New England University, Springfield, MA 01119

Abstract
A surrogate model is developed to predict the convective heat transfer coefficient of liquid sodium
(Na) flow within rectangular miniature heat sinks. Initially, kernel-based machine learning
techniques and shallow neural network are applied to a dataset with 87 Nusselt numbers for liquid
sodium in rectangular miniature heat sinks. Subsequently, a self-supervised physics-informed
neural network and transfer learning approach are used to increase the estimation performance. In
the self-supervised physics-informed neural network, an additional layer determines the weight
the of physics in the loss function to balance data and physics based on their uncertainty for a
better estimation. For transfer learning, a shallow neural network trained on water is adapted for
use with Na. Validation results show that the self-supervised physics-informed neural network
successfully estimate the heat transfer rates of Na with an error margin of approximately +8%.
Using only physics for regression, the error remains between 5% to 10%. Other machine learning
methods specify the prediction mostly within +8%. High-fidelity modeling of turbulent forced
convection of liquid metals using computational fluid dynamics (CFD) is both time-consuming
and computationally expensive. Therefore, machine learning based models offer a powerful
alternative tool for the design and optimization of liquid-metal-cooled miniature heat sinks.
Keywords: Machine learning, Transfer learning, Metrics, Predictive model, Nusselt number, Liquid metal,
Miniature heat sinks.
