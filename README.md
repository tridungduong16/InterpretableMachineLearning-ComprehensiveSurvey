
# Interpretable Machine Learning:
# Table of Contents
1. [Intrinsic Interpretable Model](#intrinsic)
	* [Generalized linear regression](#generalized)
	* [Generalized addictive model](#addictive)
	* [Bayesian model](#bayes)
	* [Rule-based classifiers](#rule)
	* [Attention mechanism](#attention)
	* [Disentangled Representation Learning](#represent)
	* [Others](#other_model)
	
2. [Model-Specific Explanation Methods](#specify)
	* [Knowledge distillation](#distillation)

3. [Model-Agnostic](#agnostic)
	* [Explanation by simplification](#simple)
	* [Feature relevance explanation](#feature)
	* [Local Explanations](#local)
	* [Example-based explantions](#example_agnostic)
	* [Others](#other_agnostic)
4. [Causal Interpretability](#survey)
	* [Causal interpretable models](#model_causal)
	* [Model-based](#base_causal)
	* [Example-based Interpretation](#example_causal)
	* [Fairness](#fairness)
	* [Guarantee](#guarantee)
5. [State-of-the-art and recent research](#state)
6. [Useful Link](#useful)

## Intrinsic interpretable model <a name="intrinsic"></a>

### Generalized linear regression: <a name="generalized"></a>
The generalized linear model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution

* Poisson regression
* Negative binomial regression
* Beta regression
* Hierarchical Linear Regression
### Generalized addictive model: <a name="addictive"></a>
A generalized additive model (GAM) is a generalized linear model in which the linear predictor depends linearly on unknown smooth functions of some predictor variables, and interest focuses on inference about these smooth functions.

<a href="https://www.codecogs.com/eqnedit.php?latex=g(\operatorname&space;{E}(Y))=\beta&space;_{0}&plus;f_{1}(x_{1})&plus;f_{2}(x_{2})&plus;\cdots&space;&plus;f_{m}(x_{m})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(\operatorname&space;{E}(Y))=\beta&space;_{0}&plus;f_{1}(x_{1})&plus;f_{2}(x_{2})&plus;\cdots&space;&plus;f_{m}(x_{m})" title="g(\operatorname {E}(Y))=\beta _{0}+f_{1}(x_{1})+f_{2}(x_{2})+\cdots +f_{m}(x_{m})" /></a>

* ***Zhang, Xuezhou, et al. "Axiomatic Interpretability for Multiclass Additive Models." _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1810.09092.pdf)
	- [Source code](https://github.com/interpretml/interpret)
	- Generalize a state-of-the-art GAM learning algorithm based on boosted trees to the multiclass setting, which nearly outperforms existing GAM learning algorithm
	- Additive Post-Processing for Interpretability (API) that provably transforms a pretrained additive model to satisfy the interpretability axioms without sacrificing accuracy
	- Optimization procedure is cyclic gradient boosting
	- **Paper:** Zhang, Xuezhou, et al. propose the generalized additive model (GAM) based on boosted trees in the mulclass setting, and interpretability methods to explain the model's behaviors.
* ***Caruana, Rich, et al. "Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission." _Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining_. 2015.***
	- [Paper Link](http://people.dbmi.columbia.edu/noemie/papers/15kdd.pdf)
	- present two case studies where high-performance gener-
alized additive models with pairwise interactions (GA2Ms)
are applied to real healthcare problems yielding intelligible
models with state-of-the-art accuracy.
	- <a href="https://www.codecogs.com/eqnedit.php?latex=g(E[y])&space;=&space;\beta&space;_{0}&space;&plus;&space;\sum&space;f_j(x_j)&space;&plus;&space;\sum&space;f_{i,j}(x_i;&space;x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(E[y])&space;=&space;\beta&space;_{0}&space;&plus;&space;\sum&space;f_j(x_j)&space;&plus;&space;\sum&space;f_{i,j}(x_i;&space;x_j)" title="g(E[y]) = \beta _{0} + \sum f_j(x_j) + \sum f_{i,j}(x_i; x_j)" /></a>
	- GA2M builds the best GAM first and then detects and ranks all possible pairs of interactions in the residuals. The top k pairs are then included in the model (k is determined by cross-validation).
	- **Paper**: Caruana, Rich, et al. propose GA2M which based on traditional GAM combining with interactions between pairs of features. 

* ***Timonen, Juho, et al. "An interpretable probabilistic machine learning method for heterogeneous longitudinal studies." arXiv preprint arXiv:1912.03549 (2019).***
	- [Paper Link](https://arxiv.org/pdf/1912.03549.pdf)
	- Present a widely applicable and interpretable probabilistic machine learning method for nonparametric longitudinal data analysis using additive Gaussian process regression.
	- 

### Bayesian model: <a name="bayes"></a>
* ***Letham, Benjamin, et al. "Interpretable classifiers using rules and bayesian analysis: Building a better stroke prediction model." The Annals of Applied Statistics 9.3 (2015): 1350-1371.***
	- [Paper Link](https://arxiv.org/pdf/1511.01644.pdf)
*  ***Kim, Been, Cynthia Rudin, and Julie A. Shah. "The bayesian case model: A generative approach for case-based reasoning and prototype classification." Advances in Neural Information Processing Systems. 2014.***
	- [Paper Link](https://beenkim.github.io/papers/KimRudinShahNIPS2014.pdf)
* ***Darwen, Paul J. "Bayesian model averaging for river flow prediction." Applied Intelligence 49.1 (2019): 103-111.***
	- [Paper Link](https://link.springer.com/content/pdf/10.1007/s10489-018-1232-0.pdf)
	- for a problem with limited data, namely future flowof five intermittent rivers. This problem is a useful proxy for many others, as the limited amount of data only allows tuningof small, simple models.


### Rule-based classifiers: <a name="rule"></a>
* ***Wang, Tong. "Multi-value rule sets for interpretable classification with feature-efficient representations." _Advances in Neural Information Processing Systems_. 2018.***
	- [Paper Link](http://papers.nips.cc/paper/8281-multi-value-rule-sets-for-interpretable-classification-with-feature-efficient-representations.pdf)
### Attention mechanism: <a name="attention"></a>	
* ***Arik, Sercan O., and Tomas Pfister. "TabNet: Attentive Interpretable Tabular Learning." arXiv preprint arXiv:1908.07442 (2019).***
	- [Paper Link](https://arxiv.org/pdf/1908.07442.pdf)
	- [Source code](https://github.com/google-research/google-research/tree/master/tabnet)
	- TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning
	- TabNet inputs raw tabular data without any feature preprocessing and is trained using gradient
descent-based optimization to learn flexible representations and enable flexible integration into end-to-end learning.
	- TabNet uses sequential attention to choose which features to reason from at each decision step
	- **Paper:**  Instead of using traditional feature engineer techniques, Arik, Sercan O., and Tomas Pfister adopt raw data as inputs for attention neural network for the flexible representations. 

 
### Disentangled Representation Learning. <a name="represent"></a>
* ***Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).***
	- [Paper Link]()
* ***Chen, Xi, et al. "Infogan: Interpretable representation learning by information maximizing generative adversarial nets." _Advances in neural information processing systems_. 2016.***
	- [Paper Link]()
* ***Higgins, Irina, et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." _Iclr_ 2.5 (2017): 6.***
	- [Paper Link]()
### Others: <a name="other_model"></a>
* ***Ross, Andrew Slavin, Michael C. Hughes, and Finale Doshi-Velez. "Right for the right reasons: Training differentiable models by constraining their explanations." _arXiv preprint arXiv:1703.03717_ (2017).***
	- [Paper Link](https://arxiv.org/pdf/1703.03717.pdf)
	- Introduce a method for efficiently explaining and regularizing differentiable models by examining and selectively penalizing their input gradients.
* ***Erion, Gabriel, et al. "Learning explainable models using attribution priors." _arXiv preprint arXiv:1906.10670_ (2019).***
	- [Paper Link](https://arxiv.org/pdf/1906.10670.pdf)
	- [Source code](https://github.com/suinleelab/attributionpriors)
	- Model priors transfer information from humans to a model by constraining the model’s parameters
	- Demonstrate the broad applicability of attribution priors (Ω) by presentingthree distinct examples that regularize models to behave more intuitively in three different domains.
	- Encode meaningful domain knowledge as differentiable functions of feature attributions
	- Propose a feature attribution method, called expected gradients, and describe why it is anatural choice for attribution priors. Expected gradients is an extension of integrated gradients with fewer hyperparameter choices. Like several other attribution methods, integrated gradients aimsto explain the difference between a model’s current prediction and the prediction that model would make when given a baseline input.
	- **Paper:** Erion, Gabriel, et al. propose a way to transfer information from human to model.
* ***Wang, Tong, and Qihang Lin. "Hybrid predictive model: When an interpretable model collaborates with a black-box model." _arXiv preprint arXiv:1905.04241_ (2019).***
	- [Paper Link](https://arxiv.org/pdf/1905.04241.pdf)
* ***Pan, Danqing, Tong Wang, and Satoshi Hara. "Interpretable Companions for Black-Box Models." _arXiv preprint arXiv:2002.03494_ (2020).***
	- [Paper Link](https://arxiv.org/pdf/2002.03494.pdf)
	- The companion model is trained from data and the predictions of the black-box model, with the objective combining area under the transparency-accuracy curve and model complexity.
	- **Paper:** Pan, Danqing, Tong Wang, and Satoshi Hara 
## Model-Specific Explanation Methods <a name="specify"></a>:
### Knowledge distillation <a name="distillation"></a>:
* ***Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).***
	- [Paper Link]()
* ***Frosst, Nicholas, and Geoffrey Hinton. "Distilling a neural network into a soft decision tree." arXiv preprint arXiv:1711.09784 (2017).***
	- [Paper Link]()
## Model-Agnostic <a name="agnostic"></a> :
### Explanation by simplification: <a name="simple"></a>
* ***Bastani, Osbert, Carolyn Kim, and Hamsa Bastani. "Interpretability via model extraction." _arXiv preprint arXiv:1706.09773_ (2017).***
	- [Paper Link](https://arxiv.org/abs/1706.09773)
	- The authors formulate model simplification as a model extraction process by approximating a transparent model to the complex one.
	- Given amodel <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;:&space;X&space;\rightarrow&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;:&space;X&space;\rightarrow&space;Y" title="f : X \rightarrow Y" /></a>, the interpretation produced by our algorithm is an approximation <a href="https://www.codecogs.com/eqnedit.php?latex=T(x)&space;\approx&space;f&space;(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(x)&space;\approx&space;f&space;(x)" title="T(x) \approx f (x)" /></a>, where T is an interpretable model.



* ***Tan, Sarah, et al. "Distill-and-compare: Auditing black-box models using transparent model distillation." _Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society_. 2018***
	- [Paper Link](https://arxiv.org/abs/1710.06169)
* ***Lakkaraju, Himabindu, et al. "Interpretable & explorable approximations of black box models." _arXiv preprint arXiv:1707.01154_ (2017).***
	- [Paper Link](https://arxiv.org/pdf/1707.01154.pdf)
	- To this end, we develop a novel objective function which allows us to learn (with optimality guarantees), a small number of compact decision setseach of which explains the behavior of the black box model in unambiguous, well-defined regions offeature space
* ***Mishra, Saumitra, Bob L. Sturm, and Simon Dixon. "Local Interpretable Model-Agnostic Explanations for Music Content Analysis." _ISMIR_. 2017.***
	- [Paper Link](https://www.eecs.qmul.ac.uk/~simond/pub/2017/MishraSturmDixon-ISMIR2017.pdf)
* ***Deng, H. "Interpreting tree ensembles with intrees (2014)." _arXiv preprint arXiv:1408.5456_.***
	- [Paper Link](https://arxiv.org/pdf/1408.5456.pdf)
	- The inTrees framework extracts rules from tree ensembles by treating tradeoffs among the frequency of the rules appearing in the trees, the errors made by the predictions, and the length of the rules. The fundamental difficulty with inTrees is that its target is limited to the classification tree ensembles. Regression tree ensembles are first transformed into the classification ones by discretizing the output, and then inTrees is applied to extract the rules. The number of discretization levels remains as a tuning parameter, which severely affects the resulting rules.

* ***Hara, Satoshi, and Kohei Hayashi. "Making tree ensembles interpretable: A Bayesian model selection approach." _arXiv preprint arXiv:1606.09066_ (2016).***
	- [Paper link](http://proceedings.mlr.press/v84/hara18a.html)
	- Presents the usage of two models (simple and complex) being the former the one in charge of interpretation and the latter of prediction by means of Expectation-Maximization and Kullback-Leibler divergence.
	- Given a complex tree ensemble, the author try to obtain the simplest representation that is essentially equivalent to the original one.
	- Derive a Bayesian model selection algorithm that optimizes the simplified model while maintaining the prediction performance
		* Adopt the probabilistic model for representing ensemble trees
		* Bayesian model selection algorithm called factorized asymptotic Bayesian (FAB) inference for finding the parameters.
	-  **Paper**: Hara, Satoshi, and Kohei Hayashi (2016) try to combine the several regions created by ensembles tree by the simpler one. In order to deal with this problem, they represent ensemble model as a probabilistic model and the find the optimal simpler one with parameters optimizing by factorized asymptotic inference (FAB).
### Feature relevance explanation: <a name="feature"></a>
* ***Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions." _Proceedings of the 34th International Conference on Machine Learning-Volume 70_. JMLR. org, 2017.***
	- [Paper Link](https://arxiv.org/pdf/1703.04730.pdf)
	- influence function is a measure of how strongly the model parameters or predictions depend on a training instance. Instead of deleting the instance, the method upweights the instance in the loss by a very small step. This method involves approximating the loss around the current model parameters using the gradient and Hessian matrix.
	- **Paper**: Koh, Pang Wei, and Percy Liang try to answer the counterfactual question that how the model prediction will be changed without an training instance. Therefore, they use the influence function to approximate the paramaters when upweighting instance $z$
	- <a href="https://www.codecogs.com/eqnedit.php?latex=\Theta&space;_{-z}&space;-&space;\Theta&space;\approx&space;\frac{1}{n}&space;\Im&space;_{up,&space;param}(z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Theta&space;_{-z}&space;-&space;\Theta&space;\approx&space;\frac{1}{n}&space;\Im&space;_{up,&space;param}(z)" title="\Theta _{-z} - \Theta \approx \frac{1}{n} \Im _{up, param}(z)" /></a>
* ***Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." _Advances in neural information processing systems_. 2017.**
	- [Paper Link](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
	- [Source code](https://github.com/slundberg/shap)
	- SHAP Tree explainer: which focuses on polynomial time fast SHAP value estimation specific for tree and ensemble tree
	- SHAP Deep Explainer: is the high speed approximation SHAP value for deep learning model
	- **Paper:**: Based on Shapley value in classical cooperative games, Lundberg, Scott M., and Su-In Lee. propose a novel method to explain the individual prediction. Since calculating Shapley value is computationally expensive and slow, his method called Kernel Shap can approximate the value by using kernel-estimation method. 

* ***Kononenko, Igor. "An efficient explanation of individual classifications using game theory." _Journal of Machine Learning Research_ 11.Jan (2010): 1-18.***
	- [Paper Link](http://www.jmlr.org/papers/volume11/strumbelj10a/strumbelj10a.pdf)
* ***Chen, Hugh, Scott Lundberg, and Su-In Lee. "Explaining Models by Propagating Shapley Values of Local Components." _arXiv preprint arXiv:1911.11888_ (2019).***
	- [Paper Link](https://arxiv.org/pdf/1911.11888.pdf)
* ***Auret, Lidia, and Chris Aldrich. "Interpretation of nonlinear relationships between process variables by use of random forests." _Minerals Engineering_ 35 (2012): 27-42.***
	- [Paper Link]()
### Local Explanations: <a name="local"></a>
* ***Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "" Why should i trust you?" Explaining the predictions of any classifier." _Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining_. 2016.***
	- [Paper Link](https://arxiv.org/pdf/1602.04938.pdf)
	- [Source code](https://github.com/marcotcr/lime)
	- often use simple linear models as the student model
to produce a local interpretable approximation to the otherwise complex black-box model.
	- build a locally interpretable model (a linear model in LIME, and patterns in aLIME) in the newly generated neighborhood of a test instance;the local model is then used as the local explanation. In both LIME and aLIME, random perturbation is used to generate the neigh-bors
* ***Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Anchors: High-precision model-agnostic explanations." _Thirty-Second AAAI Conference on Artificial Intelligence_. 2018.***
	- [Paper Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16982/15850)
	- An extension of LIME using decision rules as local interpretable classifier cl is presented. The Anchor f uses a bandit algorithm that randomly constructs the anchors with the highest coverage and respecting a user-specified precision threshold. An anchor explanation is a decision rule that sufficiently tie a prediction locally such that changes to the rest of the features values do not matter, i.e., similar instances covered by the same anchor have the same prediction outcome. Anchor is applied on tabular, images and textual datasets. 

* ***Guidotti, Riccardo, et al. "Local rule-based explanations of black box decision systems." _arXiv preprint arXiv:1805.10820_ (2018).***
	- [Paper Link](https://arxiv.org/pdf/1805.10820.pdf)
	-  A recent proposal that overcomes both LIME and Anchor in terms of performance and clarityvof the explanations is LORE (LOcal Rule-based Explanations). LORE implements function f by learning a local interpretable predictor cl on a synthetic neighborhood generated through a genetic algorithm approach. Then, it derives from the logic of cl , represented by a decision tree, an explanation e consisting of: a decision rule explaining the reasons of the decision, and a set of counterfactual rules, suggesting the changes in the instance’s features that lead to a different outcome



* ***Lucic, Ana, Hinda Haned, and Maarten de Rijke. "Explaining Predictions from Tree-based Boosting Ensembles." _arXiv preprint arXiv:1907.02582_ (2019).***
	- [Paper Link](https://arxiv.org/pdf/1907.02582.pdf)
	- Most existing methods for counterfactual explanations are:
		* model-agnostic, so they do not take into account the structure of the original model
		* involve building a surrogate model on top of the original model, which is not guaranteed to represent the original model accurately.
	- 1) We first extend [13] to work for adaptive boosting [6],
which still trains on the original labels. is allows us to
focus specially on training trees in sequence and use this to narrow our search space.
	- (2) We extend our new method for adaptive boosting to gradient
boosting, where we not only train in sequence but
also train on the negative gradients of the previous tree.
	- In our case study, we examined how we can use the sequential training nature of boosting methods to narrow the search space for alternative examples when generating explanations.


* ***Jia, Yunzhe, et al. "Improving the Quality of Explanations with Local Embedding Perturbations." _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_. 2019.***
	- [Paper Link](https://people.eng.unimelb.edu.au/baileyj/papers/KDD2019.pdf)
	- propose a local intrinsic dimensionality (LID) based locality constraint. Based on this, we then propose a new neighborhood 	u,edjikk,generation method. Our method first fits a local embedding/subspace around a given instance using the LID of the test instance as the target dimensionality, then generates neighbors in the local embedding and projects them back to the original space. 
	- **Paper:** Jia, Yunzhe, et al popose a local model-agnostic methods which genrates a embedding around a given instance called a local instrinsic dimensionality.
### Example-based explantions: <a name="example_agnostic"></a>
* ***Wachter, Sandra, Brent Mittelstadt, and Chris Russell. "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." _Harv. JL & Tech._ 31 (2017): 841.***
	- [Paper Link](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
* ***Byrne, Ruth MJ. "Counterfactuals in explainable artificial intelligence (XAI): evidence from human reasoning." _Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19_. 2019.***
	- [Paper Link](https://pdfs.semanticscholar.org/0501/b0661057d745d6bf247b7e100b8c2eac6bb7.pdf)
* ***Kim, Been, Rajiv Khanna, and Oluwasanmi O. Koyejo. "Examples are not enough, learn to criticize! criticism for interpretability." Advances in neural information processing systems. 2016.***
	- [Paper Link](https://papers.nips.cc/paper/6300-examples-are-not-enough-learn-to-criticize-criticism-for-interpretability.pdf)
* ***Goudet, Olivier, et al. "Learning functional causal models with generative neural networks." Explainable and Interpretable Models in Computer Vision and Machine Learning. Springer, Cham, 2018. 39-80.***
	- [Paper Link]()
* ***Lopez-Paz, David, et al. "Discovering causal signals in images." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2017.***
	- [Paper Link]()
### Others: <a name="other_agnostic"></a>
* ***Afrabandpey, Homayun, et al. "Making Bayesian Predictive Models Interpretable: A Decision Theoretic Approach." _arXiv preprint arXiv:1910.09358_ (2019).***
	- [Paper Link](https://arxiv.org/pdf/1910.09358.pdf)

## Causal Interpretability <a name="survey"></a>:

### Some definition:
* Potential outcomes (counterfactuals)

### Causal interpretable models: <a name="model_causal"></a> 
* Structural Causal Models
* Causal Bayesian Network
* Average Causal Effect: 
	- because casual effects vary over individuals and are not observable, they can not be measured at the individual level
### Model-based: <a name="base_causal"></a>
* ***Alvarez-Melis, David, and Tommi S. Jaakkola. "A causal framework for explaining the predictions of black-box sequence-to-sequence models." _arXiv preprint arXiv:1707.01943_ (2017).***

* ***Artelt, André, and Barbara Hammer. "Convex Density Constraints for Computing Plausible Counterfactual Explanations." _arXiv preprint arXiv:2002.04862_ (2020).***
	- Proposed and studied a formal denition of plausible counterfactual explanations. In this denition we proposed to add density constraints to the optimization problem for computing counterfactual explanations to ensure that the resulting counterfactual is plausible in the given data domain.
	- **Kernel density estimation** (**KDE**) is a [non-parametric](https://en.wikipedia.org/wiki/Non-parametric_statistics "Non-parametric statistics") way to [estimate](https://en.wikipedia.org/wiki/Density_estimation "Density estimation") the [probability density function](https://en.wikipedia.org/wiki/Probability_density_function "Probability density function") of a [random variable](https://en.wikipedia.org/wiki/Random_variable "Random variable")
* ***Bertossi, Leopoldo, et al. "Causality-based Explanation of Classification Outcomes." _arXiv preprint arXiv:2003.06868_ (2020).***
	- Dene the explanation problem for a black-box classier.
	- Introduce RESP, a black-box explanation score based on causality. 
	- Describe the formal connection between RESP and SHAP, showing that, while they are somewhat correlated, they are quite dierent. 
	- Establish the computational complexity of RESP and SHAP over two simple probabilistic models of the population. 

* ***Guidotti, Riccardo, et al. "Factual and Counterfactual Explanations for Black Box Decision Making." _IEEE Intelligent Systems_ (2019).***
	- Blackbox model b
	- Interpretable model c
	- Instance x
	- Neighborhood Z is generated from c by using genetic algorithms
	- Z is then labeled by b and then analyze by c
	- We call <a href="https://www.codecogs.com/eqnedit.php?latex=\delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta" title="\delta" /></a> a counterfactual .  A counterfactual  describes
what features to change and how to change them to get an
outcome different from y.
* ***Narendra, Tanmayee, et al. "Explaining deep learning models using causal inference." _arXiv preprint arXiv:1811.04376_ (2018). [77]***
	- [Paper Link](https://arxiv.org/pdf/1811.04376.pdf)
	- Consider the DNN as an structural causal model, apply a function on each filter of the model to obtain the targeted value such as variance or expected value of each filter and reason on the obtained SCM.
	- **Paper**: The traditional way to analyze Deep Neural Network is to build different models with different architectures. However, training several models is extremely computationally expensive. Narendra, Tanmayee, et al (2018) thus propose a causal framework to analyze the elements of DNN, and provide a simple solution to validate its result. They finally rank each component of DNN base on their contribution and important to the final prediction. 

* ***Harradon, Michael, Jeff Druce, and Brian Ruttenberg. "Causal learning and explanation of deep neural networks via autoencoded activations." _arXiv preprint arXiv:1802.00541_ (2018).***
	- [Paper Link](https://arxiv.org/pdf/1802.00541.pdf)
	- suggest that in order to have an effective interpretability, having a human-understandable causal model of DNN, which allows different kinds of causal interventions, is necessary. 
	- Based on this hypothesis, the authors propose an interpretability framework, which extracts human understandable concepts such as eyes and ears of a cat from deep neural networks, learns the causal structure between the input, output and these concepts in an SCM and performs causal reasoning on it to gain more insights into the model.


* ***Besserve, Michel, Rémy Sun, and Bernhard Schölkopf. "Counterfactuals uncover the modular structure of deep generative models." _arXiv preprint arXiv:1812.03253_ (2018).***
	- [Paper Link](https://arxiv.org/pdf/1812.03253.pdf)
* ***Rathi, Shubham. "Generating counterfactual and contrastive explanations using SHAP." arXiv preprint arXiv:1906.09293 (2019).***
	- [Paper Link](https://arxiv.org/pdf/1906.09293.pdf)
	- Generates counterfactual explanations using shapely additive explanations (SHAP).
	
*  ***Zhao, Qingyuan, and Trevor Hastie. "Causal interpretations of black-box models." Journal of Business & Economic Statistics (2019): 1-10.***
	- [Paper Link](https://web.stanford.edu/~hastie/Papers/pdp_zhao.pdf)
	- state that to extract the causal interpretations from black-box models, one needs a model with good predictive performance, domain knowledge in the form of a causal graph, and an appropriate visualization tool.
*  ***Martínez, Álvaro Parafita, and Jordi Vitrià Marca. "Explaining Visual Models by Causal Attribution." 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW). IEEE, 2019.[81]***
	- [Paper Link](https://arxiv.org/abs/1909.08891)
	- introduce a causal attribution framework to explain decisions of a classifier based on the latent factors. The framework consists of three steps:
		+ constructing Distributional Causal Graph which allows us to sample and compute likelihoods of the samples
		+ generating a counterfactual image which is as similar as possible to the original image
		+ estimating the effect of the modified factor by estimating the causal effect.
	- **Paper:** 

* ***Chattopadhyay, Aditya, et al. "Neural network attributions: A causal perspective." _arXiv preprint arXiv:1902.02302_ (2019). [12]***
	- [Paper Link](https://arxiv.org/pdf/1902.02302.pdf)
	- "What is the impact of the n-th filter of the m-th layer of a deep neural network on the predictions of the model?"
	- These frameworks are mainly designed to explain the importance of each component of a deep neural network on its predictions by answering counterfactual questions such as "What would have happened to the output of the model had we had a different component in the model?".
	- These types of questions are answered by borrowing some concepts from the causal inference literature.

*  ***Bau, David, et al. "Gan dissection: Visualizing and understanding generative adversarial networks." _arXiv preprint arXiv:1811.10597_ (2018).***
	- [Paper Link](https://arxiv.org/pdf/1811.10597.pdf)
	* Causal interpretation has also gained a lot of attention in Generative Adversarial Networks (GANs) interpretability.
	* propose a causal framework to understand "How" and "Why" images are generated by Deep Convolutional GANs (DCGANs). This is achieved by a two-step framework which finds units, objects or scenes that cause specific classes in the data samples. 
		+ In the first step, dissection is performed, where classes with explicit representations in the units are obtained by measuring the spatial agreement between individual units of the region we are examining and classes using a dictionary of object classes
		+ In the second step, intervention is performed to estimate the causal effect of a set of units on the class.

### Example-based Interpretation: <a name="example_causal"></a>
They are designed based on a new type of conditional probability <a href="https://www.codecogs.com/eqnedit.php?latex=P(y_{x}|x^{'},y^{'})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y_{x}|x^{'},y^{'})" title="P(y_{x}|x^{'},y^{'})" /></a>. This probability indicates how likely the outcome (label) of an observed instance, i.e., y′, would change to yx if x′ is set to x.

* ***Grath, Rory Mc, et al. "Interpretable credit application predictions with counterfactual explanations." arXiv preprint arXiv:1811.05245 (2018).***
	- [Paper Link](https://arxiv.org/pdf/1811.05245.pdf)
	- Your application was denied because your annual income is $30,000 and your current balance is $200. If your income had instead been $35,000 and your current balance had been $400 and allother values remained constant, your application would have been approved.
	- Propose a method to generate counterfactual examples in a high dimensional setting.
	- For **credit application prediction** via off-the-shelf interchangeable black-box classifiers.
	- Propose to reweigh the distance between the features of an instance and its corresponding counterfactual with the inverse median absolute deviation.
	- <a href="https://www.codecogs.com/eqnedit.php?latex=L(x,x^\prime,y^\prime,\lambda)=\lambda\cdot(\hat{f}(x^\prime)-y^\prime)^2&plus;d(x,x^\prime)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(x,x^\prime,y^\prime,\lambda)=\lambda\cdot(\hat{f}(x^\prime)-y^\prime)^2&plus;d(x,x^\prime)" title="L(x,x^\prime,y^\prime,\lambda)=\lambda\cdot(\hat{f}(x^\prime)-y^\prime)^2+d(x,x^\prime)" /></a>
	- **Paper:** Grath, Rory Mc, et al. propose a weighted counterfactual explanation method for features importance. Their intuition is that there are some features that make little changes to generate counterfactual explanation. 
> Ideas: Using the high-performance backbox model, after that, we evalaute the features importance by using LIME or SHAP, after that we generate the counterfactual explantion based on the weighted features. 

* ***Kanehira, Atsushi, et al. "Multimodal explanations by predicting counterfactuality in videos." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1812.01263.pdf)
* ***Hendricks, Lisa Anne, et al. "Generating counterfactual explanations with natural language." arXiv preprint arXiv:1806.09809 (2018).***
	- [Paper Link](https://arxiv.org/pdf/1806.09809.pdf)
	- define a method to generate **natural language counterfactual explanations**. The framework checks for evidences of a counterfactual class in the text explanation generated for the original input. It then checks if those factors exist in the counterfactual image and returns the existing ones. 
	- Metrics. To measure the performance of counterfactual explanation models, we propose two metrics: **phrase error** and **accuracy** with counterfactual text. Counterfactual text
should not appear in the ground truth descriptions of an
image. We report if counterfactual text appears in ground
truth descriptions of an image and refer to this as phrase
error. Ideally, the **phrase error should be 0**.
	- **Take for example:** This is not A because A does not have feature C. 



* ***Russell, Chris. "Efficient search for diverse coherent explanations." In Proceedings of the Conference on Fairness, Accountability, and Transparency, pp. 20-28. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1901.04909.pdf)
	- proposes new search algorithms for counterfactual explanations
based upon mixed integer programming
	- Formalising Counterfactual Explanations:
		+ 
* ***Wachter, Sandra, Brent Mittelstadt, and Chris Russell. "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." Harv. JL & Tech. 31 (2017): 841.***
	- [Paper Link](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
	- Formula:
	<a href="https://www.codecogs.com/eqnedit.php?latex=L(x,x^\prime,y^\prime,\lambda)=\lambda\cdot(\hat{f}(x^\prime)-y^\prime)^2&plus;d(x,x^\prime)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(x,x^\prime,y^\prime,\lambda)=\lambda\cdot(\hat{f}(x^\prime)-y^\prime)^2&plus;d(x,x^\prime)" title="L(x,x^\prime,y^\prime,\lambda)=\lambda\cdot(\hat{f}(x^\prime)-y^\prime)^2+d(x,x^\prime)" /></a>
	- propose to minimize the mean squared error between the model’s predictions and counterfactual outcomes as well as the distance between the original instances and their corresponding counterfactuals in the feature space.


* ***Mothilal, Ramaravind K., Amit Sharma, and Chenhao Tan. "Explaining machine learning classifiers through diverse counterfactual explanations." _Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency_. 2020.***
	- [Paper Link](https://arxiv.org/pdf/1905.07697.pdf)
	- [Source code](https://github.com/microsoft/DiCE)	
	- Generating counterfactual examples which satisfy the following two criteria:
		+ generated examples must be feasible given users conditions and context such as range for the features or features to be changed; 	
		+ counterfactual examples generated for explanations should be as diverse as possible. 
	- Maximize the point-wise distance between examples in feature-space or leverage the concept from Determinantal point processes to select a subset of samples with the diversity constraint.
* ***Moore, Jonathan, Nils Hammerla, and Chris Watkins. "Explaining Deep Learning Models with Constrained Adversarial Examples." Pacific Rim International Conference on Artificial Intelligence. Springer, Cham, 2019.***
	- [Paper Link]()
* ***Van Looveren, Arnaud, and Janis Klaise. "Interpretable counterfactual explanations guided by prototypes." arXiv preprint arXiv:1907.02584 (2019).***
	- [Paper Link]()
* ***Goyal, Yash, Uri Shalit, and Been Kim. "Explaining Classifiers with Causal Concept Effect (CaCE)." arXiv preprint arXiv:1907.07165 (2019).***
	- [Paper Link](https://arxiv.org/pdf/1907.07165.pdf)
	- propose to explain classifiers’ decisions by measuring the Causal Concept Effect (CACE). 
	- CACE is defined as the causal effect of a concept (such as the brightness or an object in the image) on the prediction. In order to generate counterfactuals, authors leverage a VAE-based architecture. 
	- 
* ***Liu, Shusen, et al. "Generative counterfactual introspection for explainable deep learning." arXiv preprint arXiv:1907.03077 (2019).***
	- [Paper  Link](https://arxiv.org/pdf/1907.03077.pdf)
	- propose a generative model to generate counterfactual explanations for explaining a model’s decisions.
	- Given the query image I for which the classifier C predict class c, we seek to identify the key attribute changes in I to change their decision.
* ***Ghazimatin, Azin, et al. "PRINCE: Provider-side Interpretability with Counterfactual Explanations in Recommender Systems." _Proceedings of the 13th International Conference on Web Search and Data Mining_. 2020.***
	- [Paper Link](https://arxiv.org/pdf/1911.08378.pdf)
	- Explain the question "Why does the recommendation system give the items to users"
### Fairness: <a name="fairness"></a>
Counterfactual fairnessis a notion of fairness derived from Pearl’s causal model, which considers a model is fair if for a par-ticular individual or group its prediction in the realworld is the same as that in the counterfactual worldwhere the individual(s) had belonged to a differ-ent demographic group.
* ***Kusner, Matt J., et al. "Counterfactual fairness." _Advances in Neural Information Processing Systems_. 2017.***
	- [Paper Link](https://papers.nips.cc/paper/6995-counterfactual-fairness.pdf)
	- the actual world 
	- a counterfactual world where the individual belonged to a different demographic group.
* ***Kilbertus, Niki, et al. "Avoiding discrimination through causal reasoning." Advances in Neural Information Processing Systems. 2017.***
	- [Paper Link](https://arxiv.org/pdf/1706.02744.pdf)
	- address the problem from a data generation perspective by going beyond observational data. The authors propose to
utilize causal reasoning to address the fairness problem by asking the question "What do we need to assume about the
causal data generating process?" instead of "What should be the fairness criterion?"

* ***Zhang, Junzhe, and Elias Bareinboim. "Fairness in decision-making—the causal explanation formula." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.***
	- [Paper Link](https://www.cs.purdue.edu/homes/eb/r30.pdf)
	- propose a metric (i.e., causal explanations) to quantitatively
measure the fairness of an algorithm. This measure
is based on three measures of transmission from cause to
effect namely counterfactual direct (Ctf-DE), indirect (Ctf-
IE), and spurious (Ctf-SE) effects

* ***Chiappa, Silvia. "Path-specific counterfactual fairness." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 7801-7808. 2019.***
	- [Paper Link](https://www.aaai.org/ojs/index.php/AAAI/article/download/4777/4655)

* ***Russell, Chris, Matt J. Kusner, Joshua Loftus, and Ricardo Silva. "When worlds collide: integrating different counterfactual assumptions in fairness." In Advances in Neural Information Processing Systems, pp. 6414-6423. 2017.***
	- [Paper Link](http://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf)

* ***Garg, Sahaj, Vincent Perot, Nicole Limtiaco, Ankur Taly, Ed H. Chi, and Alex Beutel. "Counterfactual fairness in text classification through robustness." In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pp. 219-226. 2019.***
	- [Paper Link ](https://dl.acm.org/doi/pdf/10.1145/3306618.3317950)
	- How would the prediction change if the
sensitive attribute referenced in the example were different?




### Guarantee: <a name="guarantee"></a>
* ***Kim, Carolyn, and Osbert Bastani. "Learning Interpretable Models with Causal Guarantees." arXiv preprint arXiv:1901.08576 (2019).***
	- [Paper link](https://arxiv.org/pdf/1901.08576.pdf)	
	- propose a framework to bridge the gap between causal and interpretable models by transforming any algorithm into an interpretable individual treatment effect estimation framework.
	- propose a general framework for learning interpretable models with causal guarantees. In particular, given any supervised learning algorithm A for learning interpretable models, our framework converts A into an algorithm A~ for learning interpretable models ^ :
X ! Y that predict the ITE  (x) 2 Y of an individual with
covariates x 2 X. Furthermore, we provide guarantees on
the performance of the models learned using A~.

### Evaluation methods:
### Challenges:
### Others, state-of-the-art and recent research: <a name="state"></a>
* ***Rudin, Cynthia. "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead." _Nature Machine Intelligence_ 1.5 (2019): 206-215.***
	- [Paper Link](https://arxiv.org/pdf/1811.10154.pdf)
 * ***Kim, Wonjae, and Yoonho Lee. Learning Dynamics of Attention: Human Prior for Interpretable Machine Reasoning.  Advances in Neural Information Processing Systems 2019***
	 - [Paper Link](https://arxiv.org/pdf/1905.11666.pdf)
	 - [Source code](https://github.com/kakao/DAFT)
 *  ***Heo, Juyeon, Sunghwan Joo, and Taesup Moon. "Fooling neural network interpretations via adversarial model manipulation." _Advances in Neural Information Processing Systems. 2019.***
	 - [Paper Link](https://arxiv.org/pdf/1902.02041.pdf)
* ***Wu, Chieh, et al. "Solving Interpretable Kernel Dimensionality Reduction." _Advances in Neural Information Processing Systems. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1909.03093.pdf)
* ***Chen, Chaofan, et al. "This looks like that: deep learning for interpretable image recognition." _Advances in Neural Information Processing Systems. 2019.***
	- [Paper Link]()
*  ***Hooker, Sara, et al. "A benchmark for interpretability methods in deep neural networks." _Advances in Neural Information Processing Systems. 2019.***
	- [Paper Link]()
* ***Schwab, Patrick, and Walter Karlen. "CXPlain: Causal explanations for model interpretation under uncertainty." _Advances in Neural Information Processing Systems. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1910.12336.pdf)
	- [Source code](https://github.com/d909b/cxplain)
	- Train causal explanation (CXPlain) models that learn to estimate to what degree certain inputs cause outputs in another machine-learning model. 
	- CXPlain can, once trained, be used to explain the target model in little time, and enables the quantification of the uncertainty associated with its feature importance estimates via bootstrap ensembling.
	- The core component of CXPlain is the causal objective that enables us to optimise explanation models to learn to explain another predictive model.
*  ***Toneva, Mariya, and Leila Wehbe. "Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)." _Advances in Neural Information Processing Systems. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1905.11833.pdf)
* ***Ying, Zhitao, et al. "Gnnexplainer: Generating explanations for graph neural networks." _Advances in Neural Information Processing Systems. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1903.03894.pdf)
*  ***Ghorbani, Amirata, et al. "Towards automatic concept-based explanations." _Advances in Neural Information Processing Systems_. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1902.03129.pdf)
* ***Ignatiev, Alexey, Nina Narodytska, and Joao Marques-Silva. "On Relating Explanations and Adversarial Examples." _Advances in Neural Information Processing Systems_. 2019.***
	- [Paper Link](https://papers.nips.cc/paper/9717-on-relating-explanations-and-adversarial-examples.pdf)
* ***Ying, Zhitao, et al. "Gnnexplainer: Generating explanations for graph neural networks." _Advances in Neural Information Processing Systems_. 2019.***
	- [Paper Link](https://arxiv.org/pdf/1903.03894.pdf)
	- ***model-agnostic approach*** for providing inter-pretable explanations for predictions of any GNN-based model on any graph-basedmachine learning task.
	- GNNEXPLAINER as an optimization task that maximizes the mutual information between a GNN’s predic-tion and distribution of possible subgraph structures
### Ideas - Potential Research Gap
* Counterfactual in 
	- Recommendation System
	- Economic decision making
	- NLP
	- Images
* The link between econometric and machine learning model
* Focus on Counterfactual Explanation
* Try to answer that "Why" question
* Causal inference


## Useful Links <a name="useful"></a>:
1. On Model Explainability [Link](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html#7_explainable_boosting_machine)
2. http://www.cs.cornell.edu/~adith/B-exam.pdf
3. https://www.accenture.com/us-en/blogs/technology-innovation/costabello-mcgrath-ai-counterfactual-explanations
4. [Double Machine Learning Tutorial](https://nbviewer.jupyter.org/github/kinokoberuji/Causal_Inference/blob/master/EconML%20customer%20policy.ipynb?fbclid=IwAR2Qq77Pw2My-sMpY2QZuXeXkRE-yg99knYPF-z5yUvYAvjFJiQTAnvprjA)
5. [How to Explain the Prediction of a Machine Learning Model? ](https://lilianweng.github.io/lil-log/2017/08/01/how-to-explain-the-prediction-of-a-machine-learning-model.html)
6.  [Knowledge distillation](https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322)
7.  [Causal inference in Machine Learning](https://www.ericsson.com/en/blog/2020/2/causal-inference-machine-learning)
## Authors

**Dung Duong** - PhD Student at UTS 

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NzA5MDc4OTcsMTM1MjkxMzM4LC02MD
MzOTAyNywtNDk3NTM3MjQ0LC0xMzYyMDI1MjkwLDE2MzQ1NTg0
NzgsMjUxMDM1OTIyLC0yNDI4OTEwNTcsMTY5NjgxMzM2NiwxND
EwNDM2MjQ3LC0xNzI3MTk0OTAzLDEzMTc3MjU2MTksNzYzMjA1
OTAwLDIxMzg0NTUzMTksLTE2ODU2OTAxNzMsMTc5MDcwNjc0MS
wxODczMzM3OTcxLDg2Mjg4MjMyNSw0MDQwMzIzMTQsLTEyMjcy
MDM5NThdfQ==
-->