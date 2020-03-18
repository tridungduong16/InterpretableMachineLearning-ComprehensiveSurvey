# Interpretable Machine Learning:
# Table of Contents
1. [Intrinsic Interpretable Model](#intrinsic)
2. [Model-Specific Explanation Methods](#specify)
3. [Model-Agnostic](#agnostic)
4. [Causal Interpretability](#survey)
5. [Useful Link](#useful)

## Intrinsic interpretable model <a name="intrinsic"></a>
### Linear Regression: 
Linear Regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables)

### Generalized linear regression:
The generalized linear model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution

* Poisson regression
* Negative binomial regression
* Beta regression
* Hierarchical Linear Regression
### Generalized addictive model:
A generalized additive model (GAM) is a generalized linear model in which the linear predictor depends linearly on unknown smooth functions of some predictor variables, and interest focuses on inference about these smooth functions.

<a href="https://www.codecogs.com/eqnedit.php?latex=g(\operatorname&space;{E}(Y))=\beta&space;_{0}&plus;f_{1}(x_{1})&plus;f_{2}(x_{2})&plus;\cdots&space;&plus;f_{m}(x_{m})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(\operatorname&space;{E}(Y))=\beta&space;_{0}&plus;f_{1}(x_{1})&plus;f_{2}(x_{2})&plus;\cdots&space;&plus;f_{m}(x_{m})" title="g(\operatorname {E}(Y))=\beta _{0}+f_{1}(x_{1})+f_{2}(x_{2})+\cdots +f_{m}(x_{m})" /></a>

* Axiomatic Interpretability for Multiclass Additive Models (2019) - ***Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining  - KDD 19***
	- [Paper Link](https://arxiv.org/pdf/1810.09092.pdf)
	- [Source code](https://github.com/interpretml/interpret)
	- Generalize a state-of-the-art GAM learning algorithm based on boosted trees to the multiclass setting
	- Additive Post-Processing for Interpretability (API) that provably transforms a pretrained additive model to satisfy the interpretability axioms without sacrificing accuracy
	- Optimization procedure is cyclic gradient boosting
* Intelligible Models for HealthCare: Predicting PneumoniaRisk and Hospital 30-day Readmission (2015) - ***Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining**
	- [Paper Link](http://people.dbmi.columbia.edu/noemie/papers/15kdd.pdf)
* Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models (2019) 
	- [Paper Link](https://arxiv.org/pdf/1911.04974.pdf)
* An interpretable probabilistic machine learning method for heterogeneous longitudinal studies (2019):
	- [Paper Link](https://arxiv.org/pdf/1912.03549.pdf)
	- Present a widely applicable and interpretable probabilistic machine learning method for nonparametric longitudinal data analysis using additive Gaussian process regression.

### Bayesian model:
* INTERPRETABLE CLASSIFIERS USING RULES AND BAYESIANANALYSIS: BUILDING A BETTER STROKE PREDICTION MODEL ***The Annals of Applied Statistics***
	- [Paper Link](https://arxiv.org/pdf/1511.01644.pdf)
*  The Bayesian Case Model: A Generative Approachfor Case-Based Reasoning and Prototype Classification 
	- [Paper Link](https://beenkim.github.io/papers/KimRudinShahNIPS2014.pdf)
* Bayesian model averaging for river flow prediction (2018)



### Rule-based classifiers:
* Multi-value Rule Sets for Interpretable Classification with Feature-Efficient Representations ***Advances in Neural Information Processing Systems. 2018***
	- [Paper Link](http://papers.nips.cc/paper/8281-multi-value-rule-sets-for-interpretable-classification-with-feature-efficient-representations.pdf)
### Attention mechanism: 	
* TabNet: Attentive Interpretable Tabular Learning 
	- [Paper Link](https://arxiv.org/pdf/1908.07442.pdf)
	- [Source code](https://github.com/google-research/google-research/tree/master/tabnet)
	- TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning
	- TabNet inputs raw tabular data without any feature preprocessing and is trained using gradient
descent-based optimization to learn flexible representations and enable flexible integration into end-to-end learning.
	- TabNet uses sequential attention to choose which features to reason from at each decision step



### Disentangled Representation Learning.
* Auto-encoding variational bayes (2013)
* Interpretable representation learning by information maximizing generative adversarial nets ***Advances in neural information processing systems. 2016.***
* betavae: Learning basic visual concepts with a constrained variational framework. ***Iclr 2.5 (2017): 6.***

### Others:
* Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations (2017)
* Learning Explainable Models Using Attribution Priors (Gabriel Erion, Joseph D. Janizek, Pascal Sturmfels, Scott Lundberg, Su-In Lee,2019)
	- [Paper Link](https://arxiv.org/pdf/1906.10670.pdf)
	- [Source code](https://github.com/suinleelab/attributionpriors)
	- Model priors transfer information from humans to a model by constraining the model’s parameters
	- Model attributions transfer information from amodel to humans by explaining the model’s behavior. 
* Hybrid Predictive Model: When an Interpretable Model Collaborates with a Black-box Model (2019)
* Interpretable Companions for Black-Box Models (2020):
	- The companion model is trained from data and the predictions of the black-box model, with the objective combining area under the transparency-accuracy curve and model complexity.


## Model-Specific Explanation Methods <a name="specify"></a>:
### Knowledge distillation:
* Distilling the knowledge in a neural network (2015)
* Distilling a neural network into a soft decision tree (2017)


## Model-Agnostic <a name="agnostic"></a> :
### Explanation by simplification:
* Interpretability via Model Extraction 
	- [Paper Link](https://arxiv.org/abs/1706.09773):
	- The authors formulate model simplification as a model extraction process by approximating a transparent model to the complex one.

* Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation ***Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society. 2018.***
	- [Paper Link](https://arxiv.org/abs/1710.06169)
* Interpretable & explorable approximations of black box models (2017)
* Local interpretable model-agnostic explanations for music content analysis
* Interpreting tree ensembles with intrees (2014)
* Making Tree Ensembles Interpretable: A Bayesian Model Selection Approach (2017)
	- [Paper link](http://proceedings.mlr.press/v84/hara18a.html)
	- Presents the usage of two models (simple and complex) being the former the one in charge of interpretation and the latter of prediction by means of Expectation-Maximization and Kullback-Leibler divergence.
	- Given a complex tree ensemble, the author try to obtain the simplest representation that is essentially equivalent to the original one.
	- Derive a Bayesian model selection algorithm that optimizes the simplified model while maintaining the prediction performance
		* Adopt the probabilistic model for representing ensemble trees
		* Bayesian model selection algorithm called factorized asymptotic Bayesian (FAB) inference for finding the parameters.
### Feature relevance explanation:
* Understanding Black-box Predictions via Influence Functions - ***Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017***
	- [Paper Link](https://arxiv.org/pdf/1703.04730.pdf)
	- influence function is a measure of how strongly the model parameters or predictions depend on a training instance. Instead of deleting the instance, the method upweights the instance in the loss by a very small step. This method involves approximating the loss around the current model parameters using the gradient and Hessian matrix.
* A unified approach to interpreting model predictions (2017) - ***Advances in neural information processing systems (pp. 4765-4774).**
	- [Paper Link](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
	- [Source code](https://github.com/slundberg/shap)
	- SHAP Tree explainer: which focuses on polynomial time fast SHAP value estimation specific for tree and ensemble tree
	- SHAP Deep Explainer: is the high speed approximation SHAP value for deep learning model
* An efficient explanation of individual classifications using game theory (2010)
	- [Paper Link](http://www.jmlr.org/papers/volume11/strumbelj10a/strumbelj10a.pdf)
* Explaining models by propagating shapley values of local components (2019) 
* Interpretation of nonlinear relationships between process variables by use of random forests

### Local Explanations:
* "Why Should I Trust You?": Explaining the Predictions of Any Classifier
	- [Paper Link](https://arxiv.org/pdf/1602.04938.pdf)
	- [Source code](https://github.com/marcotcr/lime)
* Anchors: High-Precision Model-Agnostic Explanations (2018) 
	- [Paper Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16982/15850)
* Explaining Predictions from Tree-based Boosting Ensembles (2019)
* Improving the Quality of Explanations with Local Embedding Perturbations

### Example-based explantions
* COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR 
	- [Paper Link](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
* Counterfactuals in explainable artificial intelligence (XAI): Evidence from human reasoning
* Examples are not Enough, Learn to Criticize!Criticism for Interpretability 
	- [Paper Link](https://papers.nips.cc/paper/6300-examples-are-not-enough-learn-to-criticize-criticism-for-interpretability.pdf)
* Learning functional causal models with generative neural networks, in: Explainable and Interpretable Models in Computer Vision and Machine Learning
* Discovering causal signals in images

### Others:
* Making Bayesian Predictive Models Interpretable: A Decision Theoretic Approach

## Causal Interpretability <a name="survey"></a>:
### Causal interpretable models:
* Structural Causal Models
* Causal Bayesian Network
* Average Causal Effect
### Model-based:
* Explaining deep learning models using causal inference. (2018):
	- [Paper Link](https://arxiv.org/pdf/1811.04376.pdf)
	- Consider the DNN as an structural causal model, apply a function on each filter of the model to obtain the targeted value such as variance or expected value of each filter and reason on the obtained SCM.

* Causal learning and explanation of deep neural networks via autoencoded activations. (2018):
	- [Paper Link](https://arxiv.org/pdf/1802.00541.pdf)
	- suggest that in order to have an effective interpretability, having a human-understandable causal model of DNN, which allows different kinds of causal interventions, is necessary. Based on this hypothesis, the authors propose an interpretability framework, which extracts humanunderstandable concepts such as eyes and ears of a cat from deep neural networks, learns the causal structure between the input, output and these concepts in an SCM and performs causal reasoning on it to gain more insights into the model.

* Neural network attributions: A causal perspective. (2019)
* Counterfactuals uncover the modular structure of deep generative models. (2018)
* Generating counterfactual and contrastive explanations using SHAP (2019):
	- [Paper Link](https://arxiv.org/pdf/1906.09293.pdf)
	- Generates counterfactual explanations using shapely additive explanations (SHAP).
### Example-based Interpretation: 
They are designed based on a new type of conditional probability <a href="https://www.codecogs.com/eqnedit.php?latex=P(y_{x}|x^{'},y^{'})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y_{x}|x^{'},y^{'})" title="P(y_{x}|x^{'},y^{'})" /></a>. This probability indicates how likely the outcome (label) of an observed instance, i.e., y′, would change to yx if x′ is set to x.

* Interpretable credit application predictions with counterfactual explanations:
	- Propose a method to generate counterfactual examples in a high dimensional setting.
	- For **credit application prediction** via off-the-shelf interchangeable black-box classifiers.
	- Propose to reweigh the distance between the features of an instance and its corresponding counterfactual with the inverse median absolute deviation
* Multimodal explanations by predicting counterfactuality in videos
* Generating counterfactual explanations with natural language
* Counterfactual explanations without opening the black box: Automated decisions and the GDPR.
* Explaining machine learning classifiers through diverse counterfactual explanations. (2019)
	- [Paper Link](https://arxiv.org/pdf/1905.07697.pdf)
	- Generating counterfactual examples which satisfy the following two criteria, (1) generated examples must be feasible given users conditions and context such as range for the features or features to be changed; (2) counterfactual examples generated for explanations should be as diverse as possible. 
	- Maximize the point-wise distance between examples in feature-space or leverage the concept from Determinantal point processes to select a subset of samples with the diversity constraint.
* Explaining deep learning models with constrained adversarial examples.
* Interpretable counterfactual explanations guided by prototypes.
* Generative counterfactual introspection for explainable deep learning.
* Explaining classifiers with causal concept effect (cace). (2019):
	- propose to explain classifiers’ decisions by measuring the Causal Concept Effect (CACE). 
	- CACE is defined as the causal effect of a concept (such as the brightness or an object in the image) on the prediction. In order to generate counterfactuals, authors leverage a VAE-based architecture. 
* Generative counterfactual introspection for explainable deep learning:
	- propose a generative model to generate counterfactual explanations for explaining a model’s decisions
### Fairness: 
* Counterfactual fairness - ***Advances in Neural Information Processing Systems. 2017.***
	- [Paper Link](https://papers.nips.cc/paper/6995-counterfactual-fairness.pdf)
	- the actual world 
	- a counterfactual world where the individual belonged to a different demographic group.
* Avoiding discrimination through causal reasoning ***Advances in Neural Information Processing Systems. 2017.***
* Fairness through causal awareness: Learning latent-variable models for biased data.
* Fairness in decision making – the causal explanation formula. ***Thirty-Second AAAI Conference on Artificial Intelligence. 2018.***


### Guarantee: 
* Learning interpretable models with causal guarantees (2019) - 
	- propose a framework to bridge the gap between causal and interpretable models by transforming any algorithm into an interpretable individual treatment effect estimation framework.



## Useful Links <a name="useful"></a>:
1. On Model Explainability [Link](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html#7_explainable_boosting_machine)


## Authors

**Dung Duong** - PhD Student at UTS 

