# Causal Inference:
# Table of Contents
1. [Definition](#definition)
	* [Potential outcomes](#potential)
2. [Useful Link](#link)

## Definition <a name="definition"></a>

Following the original paper of Rosenbaum & Rubin 2, in a randomized trial the treatment assignment Z and the (unobservable) potential outcomes <a href="https://www.codecogs.com/eqnedit.php?latex={Y_1,&space;Y_0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{Y_1,&space;Y_0}" title="{Y_1, Y_0}" /></a> are conditionally independent given the covariates X, i.e. <a href="https://www.codecogs.com/eqnedit.php?latex={Y_1,&space;Y_0}&space;\perp&space;Z&space;\mid&space;X." target="_blank"><img src="https://latex.codecogs.com/gif.latex?{Y_1,&space;Y_0}&space;\perp&space;Z&space;\mid&space;X." title="{Y_1, Y_0} \perp Z \mid X." /></a>


One of the main differences between (supervised) Machine Learning and Econometrics is that the former is mostly concerned with making accurate predictions, while the latter cares more about unbiased parameter estimation. In fact, in ML models, parameters are often intentionally biased if this improves prediction accuracy. But in certain cases, ML might be preferred to econometric methods - for example if the number of variables is large. The question is then how do we reap the benefits of ML algorithms if our goal is ***inference and not prediction***? Chernozhukov et al. address this issue with a method called Double/Debiased Machine Learning (DML). DML allows us to use a range of ML algorithms such as Random Forests, Gradient Boosting and Neural Networks to obtain unbiased parameter estimates.


### Randomized controlled trials (RCTs):
Randomized controlled trials (RCTs) are considered the gold standard approach for estimating the effects of treatments, interventions, and exposures (hereafter referred to as treatments) on outcomes.

### Potential outcome <a name="potential"></a>
For example, a person would have a particular income at age 40 if she had attended college, whereas she would have a different income at age 40 if she had not attended college. To measure the causal effect of going to college for this person, we need to compare the outcome for the same individual in both alternative futures. Since it is impossible to see both potential outcomes at once, one of the potential outcomes is always missing. This dilemma is the "fundamental problem of causal inference". 

### Average causal effect
An estimate of the ***Average Causal Effect*** (also referred to as the ***Average Treatment Effect***) can then be obtained by computing the difference in means between the treated (college-attending) and control (not-college-attending) samples. 

### Confounder:
A confounder is a variable that is associated or has a relationship with both the exposure and the outcome of interest
### Propensity score matching: 
By predicting Z based on X, we have estimated the propensity score, i.e. p(Z=1|x). This of course assumes that we have used a classification method that returns probabilities for the classes Z=1 and Z=0. Let e_i=p(Z=1|x_i) be the propensity score of the i-th observation, i.e. the propensity of the i-th participant getting the treatment (Z=1).

We can use the propensity score to define weights w_i to create a synthetic sample in which the distribution of measured baseline covariates is independent of treatment assignment5, i.e.
w_i=\frac{z_i}{e_i}+\frac{1-z_i}{1-e_i},

where z_i indicates if the i-th subject was treated.

The covariates from our data sample x_i are then weighted by w_i to eliminate the correlation between X and Z, which is a technique known as inverse probability of treatment weighting (IPTW). This allows us to estimate the causal effect via the following approach:

    Train a model with covariates X to predict Z,
    calculate the propensity scores e_i by applying the trained model to all x_i,
    train a second model with covariates X and Z as features and response Y as target by using w_i as sample weight for the i-th observation,
    use this model to predict the causal effect like in the randomized trial approach.

IPTW is based on a simple intuition. For a randomized trial with p(Z=1)=k the propensity score would be equal for all patients, i.e. e_i=\frac{1}{k} and thus w_i=k. In a nonrandomized trial, we would assign low weights to samples where the assignment of treatment matches our expectation and high weights otherwise. By doing so, we draw the attention of the machine learning algorithm to the observations where the effect of treatment is most prevalent, i.e. least confounded with the covariates.


### Instrumental Variables

### Average Treatment Effect 
(ATE= E(Y=1)-E(Y=0))

### Manhattan distance:
Manhattan distance is used to measure the distance between treated instance and 

### Inverse Probability of Treatment Weighting (IPTW)


### Double Machine Learning
Chernozhukov et al. address this question by proposing a method called Double Machine Learning (DML). This technique applies to the case in which we want to explain:
	- an outcome variable (e.g. educational attainment)
	- a number of control variables (e.g. age of the child, income of the parents etc.)
	- a treatment/policy variable (e.g. whether the family benefited from a certain policy), which itself also depends on the control variables. 

What we are really interested in is the effect of the policy on the outcome, and we wish to find an estimate of the effect that is as close to the truth (unbiased) as possible.

	- We fit a preliminary ML algorithm to predict the outcome variable using the control variables;	
	- We fit a second ML algorithm to partial out the effect of the control variables on the policy variable.



## Useful Link <a name="link"></a>
* https://en.wikipedia.org/wiki/Rubin_causal_model
* https://florianwilhelm.info/2017/04/causal_inference_propensity_score/
