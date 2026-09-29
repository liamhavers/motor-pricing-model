# Learning checkpoints

Questions to answer at the end of each stage. Answers go under each question.

## 1. Frequency model
- Why use a log(exposure) offset rather than dividing claims by exposure and modelling the rate?
- Why Poisson for counts?
- How did we handle exposure in the GBM, and why is that equivalent?

## 2. Severity model
- Why Gamma for severity?
- Why weight by claim count?
- Why cap large losses, and what happens to the capped amount in a real pricing process?

## 3. Pure premium
- What is a Tweedie distribution and why does it suit total claim cost?
- When would you prefer frequency x severity over a single Tweedie model?

## 4. Evaluation
- What does deviance measure, and why not RMSE?
- What does a Gini of 0.3 mean in practice?
- What does a double-lift chart show that a single metric doesn't?

## 5. Commercial analysis
- The GBM beat the GLM. Would you deploy it?
- What would an underwriter, a regulator and a pricing manager each want to know first?

## 6. README
- Explain the whole project in two minutes to a non-technical interviewer.
