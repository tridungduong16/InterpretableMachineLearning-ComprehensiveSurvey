<h1><img src="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/MSR-ALICE-HeaderGraphic-1920x720_1-800x550.jpg" width="130px" align="left" style="margin-right: 50px;"> <a href="https://github.com/Microsoft/EconML">EconML:</a> A Python Package for ML-Based Heterogeneous Treatment Effects Estimation</h1>

**EconML** is a Python package for estimating heterogeneous treatment effects from observational data via machine learning. This package was designed and built as part of the [ALICE project](https://www.microsoft.com/en-us/research/project/alice/) at Microsoft Research with the goal to combine state-of-the-art machine learning 
techniques with econometrics to bring automation to complex causal inference problems. The promise of EconML:

* Implement recent techniques in the literature at the intersection of econometrics and machine learning
* Maintain flexibility in modeling the effect heterogeneity (via techniques such as random forests, boosting, lasso and neural nets), while preserving the causal interpretation of the learned model and often offering valid confidence intervals
* Use a unified API
* Build on standard Python packages for Machine Learning and Data Analysis

In a nutshell, this
toolkit is designed to measure the causal effect of some treatment variable(s) `T` on an outcome 
variable `Y`, controlling for a set of features `X`. For detailed information about the package, 
consult the documentation at https://econml.azurewebsites.net/.

# Getting Started with EconML Sample Notebooks

Clone this project to your free Azure Notebooks account. To do so, you first have to sign in [here](https://notebooks.azure.com/). The sign in process will require you to have a Microsoft account. If you don't have one, click "Create one!" to make one. The process should only take a few minutes. 

To run a notebook, open it and run the cells like you normally would. If this is the first notebook you run, you might have to wait a few seconds for the kernel to install dependencies. It is not recommended that you run the notebook before the setup is finished as it may lead to undesired outcomes (such as not finding certain packages).

**Note:** You might experience decreased performance or long runtimes due to limitations in Azure Notebooks virtual machine size. To have a better experience, we recommend installing the latest `econml` release from [PyPI](https://pypi.org/project/econml/) on your own machine
```
pip install econml
```
and downloading the notebooks from [here](https://github.com/Microsoft/EconML/tree/master/notebooks).

# About Treatment Effect Estimation

One of the biggest promises of machine learning is to automate decision making in a multitude of domains. At the core of many data-driven personalized decision scenarios is the estimation of heterogeneous treatment effects: what is the causal effect of an intervention on an outcome of interest for a sample with a particular set of features? 

Such questions arise frequently in customer segmentation (what is the effect of placing a customer in a tier over another tier), dynamic pricing (what is the effect of a pricing policy on demand) and medical studies (what is the effect of a treatment on a patient). In many such settings we have an abundance of observational data, where the treatment was chosen via some unknown policy, but the ability to run control A/B tests is limited.

# Example Applications

<table style="width:80%">
  <tr align="left">
    <td width="20%"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Business_card_-_The_Noun_Project.svg/610px-Business_card_-_The_Noun_Project.svg.png" width="200px"/></td>
    <td width="80%">
        <h4>Customer Targeting</h4>
        <p> Businesses offer personalized incentives to customers to increase sales and level of engagement. Any such personalized intervention corresponds to a monetary investment and the main question that business analytics are called to answer is: what is the return on investment? Analyzing the ROI is inherently a treatment effect question: what was the effect of any investment on a customer's spend? Understanding how ROI varies across customers can enable more targeted investment policies and increased ROI via better targeting. 
        </p>
    </td>
  </tr>
  <tr align="left">
    <td width="20%"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c9/Online-shop_button.jpg" width="200px"/></td>
    <td width="80%">
        <h4>Personalized Pricing</h4>
        <p>Personalized discounts have are widespread in the digital economy. To set the optimal personalized discount policy a business needs to understand what is the effect of a drop in price on the demand of a customer for a product as a function of customer characteristics. The estimation of such personalized demand elasticities can also be phrased in the language of heterogeneous treatment effects, where the treatment is the price on the demand as a function of observable features of the customer. </p>
    </td>
  </tr>
  <tr align="left">
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/VariousPills.jpg/640px-VariousPills.jpg" width="200px"/></td>
    <td width="80%">
        <h4>Stratification in Clinical Trials</h4>
        <p>
        Which patients should be selected for a clinical trial? If we want to demonstrate that a clinical treatment has an effect on at least some subset of a population then fully randomized clinical trials are inappropriate as they will solely estimate average effects. Using heterogeneous treatment effect techniques, we can use observational data to come up with estimates of these effects and identify good candidate patients for a clinical trial that our model estimates have high treatment effects.
        </p>
    </td>
  </tr>
  <tr align="left">
    <td width="20%"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Mouse-cursor-hand-pointer.svg/1023px-Mouse-cursor-hand-pointer.svg.png" width="200" /></td>
    <td width="80%">
        <h4>Learning Click-Through-Rates</h4>
    <p>
        In the design of a page layout and ad placement, it is important to understand the click-through-rate of page components on different positions of a page. Modern approaches may be to run multiple A/B tests, but when such page component involve revenue considerations, then observational data can help guide correct A/B tests to run. Heterogeneous treatment effect estimation can provide estimates of the click-through-rate of page components from observational data. In this setting, the treatment is simply whether the component is placed on that page position and the response is whether the user clicked on it.
    </p>
    </td>
  </tr>
</table>


# Blogs and Publications

* May 2019: [Open Data Science Conference Workshop](https://staging5.odsc.com/training/portfolio/machine-learning-estimation-of-heterogeneous-treatment-effect-the-microsoft-econml-library) 

* 2018: [Orthogonal Random Forests paper](https://arxiv.org/abs/1806.03467)

* 2017: [DeepIV paper](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)

# References

M. Oprescu, V. Syrgkanis and Z. S. Wu.
**Orthogonal Random Forest for Causal Inference.**
[*ArXiv preprint arXiv:1806.03467*](http://arxiv.org/abs/1806.03467), 2018.

Jason Hartford, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. **Deep IV: A flexible approach for counterfactual prediction.** [*Proceedings of the 34th International Conference on Machine Learning*](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf), 2017.

V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, and a. W. Newey. **Double Machine Learning for Treatment and Causal Parameters.** [*ArXiv preprint arXiv:1608.00060*](https://arxiv.org/abs/1608.00060), 2016.
