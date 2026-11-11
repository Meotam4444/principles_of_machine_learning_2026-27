# Lecture 01 — A first glimpse at machine learning & the ML pipeline
*Prof. Christian Bauckhage, Principles of Machine Learning (69 slides)*

## Overview
1. Machine learning: what, why, and how
2. The ML pipeline: data pre-processing, model selection, model fitting, model evaluation
3. Summary

---

## 1. What is machine learning?
- **AI** builds *cognitive machines* (see, hear, read, write, plan, steer robots, …). **ML** is an important branch of AI.
- **Core definition (slide 6):**
  > Machine learning is the science of **fitting mathematical models to data**.
  > Equivalently: running algorithms on example data to **adjust the parameters of other algorithms (models)** so that those models can perform cognitive tasks.
- Two follow-up questions: *what are data?* and *what are models?*

### Data
- Philosophically: known facts that can be recorded, stored and retrieved, and that carry implicit "meaning".
- Technically: collections of (numerical) values from measurements, surveys, … that computers can process.
- **Structured data**: fits into spreadsheets or relational DBs; elementary data types; easy to search or hash.
- **Unstructured data**: text, speech, images, video; hard to analyze automatically.
- **Meta-data / annotations**: data about data (labels, tags, timestamps, locations, captions).
- The course focuses on **algorithms and their mathematical principles**, not ML systems engineering.

### Mathematical models
- Formal descriptions of aspects of the observable world. They characterize principles, states or dynamics, which lets us make predictions or decisions.
- Models are coarse abstractions: *"All models are wrong, but some are useful."* (George Box)
- Neural networks are "nothing but very general mathematical models".

### Three ML paradigms
| Paradigm | Typical tasks | In this course? |
|---|---|---|
| Supervised learning | classification, regression | yes |
| Unsupervised learning | clustering, dimensionality reduction (density estimation) | yes |
| Reinforcement learning | — | no (separate course) |

## 2. Why do we use ML?
Model fitting lets us make sense of data and **generalize from known observations to new ones**. Typical problems: regression, classification / pattern recognition, density estimation / clustering, simplification / surrogate modeling, (multi-modal) content generation.

### Didactic examples
- **Regression**: data $D=\{(x_j,y_j)\}_{j=1}^n$, $x_j,y_j\in\mathbb R$. We look for $f:\mathbb R\to\mathbb R$ with $y_j\approx f(x_j)$.
  - There are infinitely many possible $f$, so we must **hypothesize a model family**, e.g. $f(x\mid a,b)=x^a\sin(bx)$.
  - That hypothesis class is still infinite (every choice of $a,b$), so we need smart fitting algorithms (statistical optimization). In the example the fit gives $a=1.10,\ b=1.99$.
  - "We may regress anything onto everything", e.g. next-word prediction (an LLM is regression with a very sophisticated model).
- **Classification**: $x_j\in\mathbb R^2$ from classes $\Omega_1,\Omega_2$ with labels $y_j\in\{+1,-1\}$; train $f:\mathbb R^2\to\{-1,+1\}$.
- **Density estimation**: find a pdf $f$ that describes how the $x_j$ are distributed. Since about 2015 this has been **the most important ML task**: all modern generative AI is at heart a fitted density model.
- **Clustering**: find $f:\mathbb R^2\to\{1,\dots,k\}$ that assigns points to $k$ clusters.
- Common recipe: **(1) hypothesize a model, (2) use statistical optimization to fit its parameters to the data.**

### Supervised vs. unsupervised
- **Supervised** (regression, classification): data are pairs $(x_j,y_j)$, so the structure is *explicit*. The task is to learn to replicate and **generalize known structures**.
- **Unsupervised** (density estimation, clustering): only the $x_j$ are given, so the structure is *implicit at best*. The task is to **make implicit information explicit**.

### Why fit models at all?
Good models let us compress information, generalize, reason about cause and effect, predict, decide, plan and generate content. In short, they let us implement cognitive capabilities on computers.
- Compression example: many data points are replaced by a model with 2 parameters, $y=x^{1.1}\sin(1.99x)$.

## 3. How do we do ML?
The three fundamental questions of the whole course:
1. How do we **choose** models?
2. How do we **fit** them to data?
3. What must we **be aware of** while doing so?

Mathematical tools: geometry and linear algebra, probability and statistics, calculus and optimization. CS tools: data storage and processing systems, efficient and robust implementations of fitting algorithms.

---

## 4. The ML pipeline
- Version 1 (von Rueden et al. 2021): training data $(x_1,y_1),\dots,(x_n,y_n)$ and a **hypothesis set** for the problem $f:X\to Y$ go into a **learning algorithm**, which outputs the **final hypothesis** $y=f(x\mid\theta)$.
- Version 2: data collection → pre-processing → model selection → model fitting → model evaluation → model application.
- Terminology is still fuzzy (*ML is a young discipline*): "hypothesis set" = "model family", "final hypothesis" = "fitted model".

### Stages
1. **Problem specification / data collection.** Sometimes the order is reversed: we have data and use ML for *knowledge discovery*.
2. **Data pre-processing**
   - *cleansing*: remove outliers and incomplete or faulty points
   - *labeling*: (manual) annotation; annotated data is **incredibly valuable**
   - *feature extraction*: more abstract representations, either from background knowledge (height and weight → BMI) or learned automatically (deep nets)
   - *normalization*: bring values into manageable numerical ranges
3. **Model selection**: choose a model class, i.e. a family of functions.
4. **Model fitting / training**: run optimization to find the parameters.
5. **Model evaluation / testing**: does the model **generalize**?
6. **Model refinement**: get more or better data, or reconsider the model class, and iterate (for foundation models this is *fine-tuning*).
7. **Model application**: feed in new data and use the predictions.

### Key principles (exam-relevant!)
- **Training data must be representative** of everything that is to be learned (cautionary example: the 2015 Google Photos incident).
- **Test data must be independent of training data**: $D_{trn}\cap D_{tst}=\emptyset$.
- **Good performance on training data means nothing.** What counts is generalization to independent test data. Avoid the "beginner's mistake of improper model testing".
- Caveat: frontier models trained on trillions of data points worry less about over-fitting, but unless you are OpenAI or DeepMind, the principles above still apply.

## Summary
- ML = fitting models to data. Terminology is fuzzy. ML is a toolbox for data analysis and synthesis.
- Pipeline: pre-processing → model selection → fitting → evaluation.
- ML sits at the intersection of mathematics and computer science (together with pattern recognition, data mining and data science).
