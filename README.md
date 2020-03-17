# Interpretable Machine Learning:
## Intrinsic interpretable model
1. Linear Regression
2. Generalized linear regression
	- Poisson regression
	- Negative binomial regression
  	- Beta regression
3. Generalized addictive model
	* Axiomatic Interpretability for Multiclass Additive Models - [Link](https://arxiv.org/pdf/1810.09092.pdf)
	* Intelligible Models for HealthCare: Predicting PneumoniaRisk and Hospital 30-day Readmission - [Link](http://people.dbmi.columbia.edu/noemie/papers/15kdd.pdf)
4. Decision sets 
5. Rule-based classifiers 
6. Scorecards
7. Short rules 
8. Concept based explanation
10. Support vector machine
11. TabNet: Attentive Interpretable Tabular Learning [Link](https://arxiv.org/pdf/1908.07442.pdf)
12. Bayesian model:
	* INTERPRETABLE CLASSIFIERS USING RULES AND BAYESIANANALYSIS: BUILDING A BETTER STROKE PREDICTIONMODEL - [Link](https://arxiv.org/pdf/1511.01644.pdf)
	*  The Bayesian Case Model: A Generative Approachfor Case-Based Reasoning and Prototype Classification - [Link](https://beenkim.github.io/papers/KimRudinShahNIPS2014.pdf)

## Model-Specific Explanation Methods:
1. Knowledge distillation
2. Ensembles and Multiple Classifier Systems
3. Support Vector Machines
4. Deep learning

## Model-Agnostic:
1. Explanation by simplification:
	* Interpretability via Model Extraction - [Link](https://arxiv.org/abs/1706.09773):
		- The authors formulate model simplification as a model extraction process by approximating a transparent model to the complex one.

	* Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation - [Link](https://arxiv.org/abs/1710.06169)]
	* Rule-based learner:
		- Interpretable & explorable approximations of black box models (2017)
		- Local interpretable model-agnostic explanations for music content analysis

	* Decision Tree:
		- Interpreting tree ensembles with intrees (2014)
		- Making tree ensembles interpretable: presents the usage of two models (simple and complex) being the former the one in charge of interpretation and the latter of prediction by means of Expectation-Maximization and Kullback-Leibler divergence.
2. Feature relevance explanation:
	* Understanding Black-box Predictions via Influence Functions - [Link](https://arxiv.org/pdf/1703.04730.pdf)
	* Game theory inspired: 
		- A unified approach to interpreting model predictions:
			+ SHAP Tree explainer: which focuses on polynomial time fast SHAP value estimation specific for tree and ensemble tree
			+ SHAP Deep Explainer: is the high speed approximation SHAP value for deep learning model
			+ SHAP Kernel Explainer: 

		- An efficient explanation of individual classifications using game theory
		- Explaining models by propagating shapley values of local components (2019) 
	* Interpretation of nonlinear relationships between process variables by use of random forests

3. Local Explanations:
	* "Why Should I Trust You?": Explaining the Predictions of Any Classifier - [Link](https://arxiv.org/pdf/1602.04938.pdf)
	* Anchors: High-Precision Model-Agnostic Explanations - [Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16982/15850)
4. Example-based explantions
	* Counterfactual Explanations:
		- COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR - [Link](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
	* Prototypes and Criticisms MMD-Critic:
		- Examples are not Enough, Learn to Criticize!Criticism for Interpretability - [Link](https://papers.nips.cc/paper/6300-examples-are-not-enough-learn-to-criticize-criticism-for-interpretability.pdf)
	
## Application:
1. Recommendation system
2. Ecnometrics
3. Natural language processing
4. Computer vision
5. Tabular Data
