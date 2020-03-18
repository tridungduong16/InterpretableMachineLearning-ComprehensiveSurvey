# Interpretable Machine Learning:
# Table of Contents
1. [Intrinsic interpretable model](#intrinsic)
2. [Model-Specific Explanation Methods](#specify)
3. [Model-Agnostic](#agnostic)
4. [Causal interpretability](#survey)
5. [Useful Link](#useful)

## Intrinsic interpretable model <a name="intrinsic"></a>
1. Linear Regression: 

 	<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i&space;=&space;0}^{n}x_{i}\beta_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i&space;=&space;0}^{n}x_{i}\beta_{i}" title="\sum_{i = 0}^{n}x_{i}\beta_{i}" /></a>


2. Generalized linear regression:

<a href="https://www.codecogs.com/eqnedit.php?latex=g(E_Y(y|x))=\beta_0&plus;\beta_1{}x_{1}&plus;\ldots{}\beta_p{}x_{p}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(E_Y(y|x))=\beta_0&plus;\beta_1{}x_{1}&plus;\ldots{}\beta_p{}x_{p}" title="g(E_Y(y|x))=\beta_0+\beta_1{}x_{1}+\ldots{}\beta_p{}x_{p}" /></a>

	- Poisson regression
	- Negative binomial regression
  	- Beta regression
3. Generalized addictive model
	* Axiomatic Interpretability for Multiclass Additive Models (2019)
		- [Paper Link](https://arxiv.org/pdf/1810.09092.pdf)
		- [Source code](https://github.com/interpretml/interpret)
		- Generalize a state-of-the-art GAM learning algorithm based on boosted trees to the multiclass setting
		- Additive Post-Processing for Interpretability (API) that provably transforms a pretrained additive model to satisfy the interpretability axioms without sacrificing accuracy
		- Optimization procedure is cyclic gradient boosting
	* Intelligible Models for HealthCare: Predicting PneumoniaRisk and Hospital 30-day Readmission (2015) 
		- [Paper Link](http://people.dbmi.columbia.edu/noemie/papers/15kdd.pdf)
	* Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models (2019) 
		- [Paper Link](https://arxiv.org/pdf/1911.04974.pdf)
	* An interpretable probabilistic machine learning method for heterogeneous longitudinal studies (2019):
		- [Paper Link](https://arxiv.org/pdf/1912.03549.pdf)
		- Present a widely applicable and interpretable probabilistic machine learning method for nonparametric longitudinal data analysis using additive Gaussian process regression.
		
4. Decision sets 
5. Rule-based classifiers:
	* Multi-value Rule Sets for Interpretable Classification with Feature-Efficient Representations (NIPS 2018)
9. TabNet: Attentive Interpretable Tabular Learning 
	* [Paper Link](https://arxiv.org/pdf/1908.07442.pdf)
	* [Source code](https://github.com/google-research/google-research/tree/master/tabnet)
	* TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning
	* TabNet inputs raw tabular data without any feature preprocessing and is trained using gradient
descent-based optimization to learn flexible representations and enable flexible integration into end-to-end learning.
	* TabNet uses sequential aention to choose which features to reason from at each decision step
	
12. Bayesian model:
	* INTERPRETABLE CLASSIFIERS USING RULES AND BAYESIANANALYSIS: BUILDING A BETTER STROKE PREDICTIONMODEL - [Link](https://arxiv.org/pdf/1511.01644.pdf)
	*  The Bayesian Case Model: A Generative Approachfor Case-Based Reasoning and Prototype Classification - [Link](https://beenkim.github.io/papers/KimRudinShahNIPS2014.pdf)
	

12. Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations (2017)
13. Learning Explainable Models Using Attribution Priors (Gabriel Erion, Joseph D. Janizek, Pascal Sturmfels, Scott Lundberg, Su-In Lee,2019)
	* [Paper Link](https://arxiv.org/pdf/1906.10670.pdf)
	* [Source code](https://github.com/suinleelab/attributionpriors)
	* Model priors transfer information from humans to a model by constraining the model’s parameters
	* Model attributions transfer information from amodel to humans by explaining the model’s behavior. 
14. Hybrid Predictive Model: When an Interpretable Model Collaborates with a Black-box Model (2019)
15. Interpretable Companions for Black-Box Models (2020)
16. Concept based explanation
17. Attention mechanism
18. Disentangled Representation Learning.


## Model-Specific Explanation Methods <a name="specify"></a>:
1. Knowledge distillation
2. Ensembles and Multiple Classifier Systems
3. Support Vector Machines
4. Deep learning

## Model-Agnostic <a name="agnostic"></a> :
1. Explanation by simplification:
	* Interpretability via Model Extraction 
		- [Paper Link](https://arxiv.org/abs/1706.09773):
		- The authors formulate model simplification as a model extraction process by approximating a transparent model to the complex one.

	* Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation - [Link](https://arxiv.org/abs/1710.06169)
	* Rule-based learner:
		- Interpretable & explorable approximations of black box models (2017)
		- Local interpretable model-agnostic explanations for music content analysis

	* Decision Tree:
		- Interpreting tree ensembles with intrees (2014)
		- Making Tree Ensembles Interpretable: A Bayesian Model Selection Approach (2017)
			* [Paper link](http://proceedings.mlr.press/v84/hara18a.html)
			* Presents the usage of two models (simple and complex) being the former the one in charge of interpretation and the latter of prediction by means of Expectation-Maximization and Kullback-Leibler divergence.
			* Given a complex tree ensemble, the author try to obtain the simplest representation that is essentially equivalent to the original one.
			* Derive a Bayesian model selection algorithm that optimizes the simplified model while maintaining the prediction performance
			* Adopt the probabilistic model for representing ensemble trees
			* Bayesian model selection algorithm called factorized asymptotic Bayesian (FAB) inference for finding the parameters.
2. Feature relevance explanation:
	* Understanding Black-box Predictions via Influence Functions (2017)
		- [Paper Link](https://arxiv.org/pdf/1703.04730.pdf)
		- influence function is a measure of how strongly the model parameters or predictions depend on a training instance. Instead of deleting the instance, the method upweights the instance in the loss by a very small step. This method involves approximating the loss around the current model parameters using the gradient and Hessian matrix.
	* Game theory inspired: 
		- A unified approach to interpreting model predictions:
			+ [Paper Link](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
			+ [Source code](https://github.com/slundberg/shap)
			+ SHAP Tree explainer: which focuses on polynomial time fast SHAP value estimation specific for tree and ensemble tree
			+ SHAP Deep Explainer: is the high speed approximation SHAP value for deep learning model
		- An efficient explanation of individual classifications using game theory (2010)
			+ [Paper Link](http://www.jmlr.org/papers/volume11/strumbelj10a/strumbelj10a.pdf)
		- Explaining models by propagating shapley values of local components (2019) 
	* Interpretation of nonlinear relationships between process variables by use of random forests

3. Local Explanations:
	* "Why Should I Trust You?": Explaining the Predictions of Any Classifier
		- [Paper Link](https://arxiv.org/pdf/1602.04938.pdf)
		- [Source code](https://github.com/marcotcr/lime)
	* Anchors: High-Precision Model-Agnostic Explanations (2018) 
		- [Paper Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16982/15850)
	* Explaining Predictions from Tree-based Boosting Ensembles (2019)
	* Improving the Quality of Explanations with Local Embedding Perturbations

4. Example-based explantions
	* Counterfactual Explanations:
		- COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR - [Link](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
		- Counterfactuals in explainable artificial intelligence (XAI): Evidence from human reasoning
	* Prototypes and Criticisms MMD-Critic:
		- Examples are not Enough, Learn to Criticize!Criticism for Interpretability - [Link](https://papers.nips.cc/paper/6300-examples-are-not-enough-learn-to-criticize-criticism-for-interpretability.pdf)
	* Learning functional causal models with generative neural networks, in: Explainable and Interpretable Models in Computer Vision and Machine Learning
	* Discovering causal signals in images

5. Making Bayesian Predictive Models Interpretable: A Decision Theoretic Approach

## Causal Interpretability <a name="survey"></a>:
1. Causal interpretable models:
	* Structural Causal Models
	* Causal Bayesian Network
	* Average Causal Effect
2. Model-based:
	* Explaining deep learning models using causal inference. (2018)
	* Causal learning and explanation of deep neural networks via autoencoded activations. (2018)
	* Neural network attributions: A causal perspective. (2019)
	* Counterfactuals uncover the modular structure of deep generative models. (2018)
	* Generating counterfactual and contrastive explanations using SHAP (2019):
		- [Paper Link]
		- generates counterfactual explanations using shapely additive explanations (SHAP).
3. Example-based: 
	* Interpretable credit application predictions with counterfactual explanations:
		- Propose a method to generate counterfactual examples in a high dimensional setting.
		- For **credit application prediction** via off-the-shelf interchangeable black-box classifiers.
		- Propose to reweigh the distance between the features of an instance and its corresponding counterfactual with the inverse median absolute deviation

	* Multimodal explanations by predicting counterfactuality in videos
	* Generating counterfactual explanations with natural language
	* Counterfactual explanations without opening the black box: Automated decisions and the GDPR.
	* Explaining machine learning classifiers through diverse counterfactual explanations. (2019)
	* Explaining deep learning models with constrained adversarial examples.
	* Interpretable counterfactual explanations guided by prototypes.
	* Generative counterfactual introspection for explainable deep learning.
	* Explaining classifiers with causal concept effect (cace). (2019)
	* Generative counterfactual introspection for explainable deep learning:
		- propose a generative model to generate counterfactual explanations for explaining a model’s decisions
4. Fairness: 
	* Counterfactual fairness.
	* Avoiding discrimination through causal reasoning.
	* Fairness through causal awareness: Learning latent-variable models for biased data.
	* Fairness in decisionmaking – the causal explanation formula. (2018)
5. Guarantee: 
	* Learning interpretable models with causal guarantees (2019)



## Useful Links <a name="useful"></a>:
1. On Model Explainability [Link](https://everdark.github.io/k9/notebooks/ml/model_explain/model_explain.nb.html#7_explainable_boosting_machine)


## Authors

**Dung Duong** - PhD Student at UTS 

