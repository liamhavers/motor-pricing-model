# Learning checkpoints

Questions to answer at the end of each stage. Answers go under each question.

## 1. Frequency model
- Why use a log(exposure) offset rather than dividing claims by exposure and modelling the rate?

We use a log offset, as dividing claims by exposure would give the same weightings to say, a one month policy with 12 claims, as a year-long policy, even though a year-long policy would show you far less.

- Why Poisson for counts?

We use poisson for claims count as it directly describes how many times a rare event happens in a fixed window of time, which works best for claims as most policies have 0, few have 1, even fewer have 2 etc. It also only allows whole, non-negative counts, with a variance that grows with the mean. Linear regression can predict negative claims and assumes every policy is equally noisy.


- How did we handle exposure in the GBM, and why is that equivalent?

We handled the exposure in the GBM by being in the same form as the GLM, log(exposure) is added with a fixed coefficient of 1 and never learned, with the trees only learning the rate per year. It goes in through LightGBM's init_score, the starting value each prediction is built on. so it's the same as the GLM offset: exposure scales the expected claims, and the trees only ever learn the rate per year.

## 2. Severity model
- Why Gamma for severity?

Claim costs are positive and right-skewed, and their spread grows with their size, so we use a Gamma distribution, where the standard deviation is proportional to the mean. That means Gamma deviance judges errors in proportion: missing a €1,000 claim by €500 counts the same as missing a €10,000 claim by €5,000. RMSE squares the errors, so it would treat the second miss as 100 times worse and the model would chase the expensive claims.

- Why weight by claim count?

Each severity row is a policy's average cost per claim, and an average of four claims is much more reliable than a single claim. Weighting by claim count gives each claim one vote rather than each policy, so the model estimates the cost per claim. That's what we need, because it gets multiplied by the predicted number of claims.

- Why cap large losses, and what happens to the capped amount in a real pricing process?

A handful of very large claims are rare and essentially random. In our data, 0.3% of claims made up about a quarter of the cost. There's no pattern for a model to learn from them, and leaving them in makes the model unstable and pushes up prices for whoever happens to share a rating cell with them. So we cap each claim, at €50,000 here, and model the capped amounts. The cost above the cap still has to be paid, so it's spread back across all policies as a large-loss loading. In practice that's estimated from several years of large-loss history or a separate large-loss model, because one year is too volatile.

## 3. Pure premium
- What is a Tweedie distribution and why does it suit total claim cost?

Total claim cost per policy is an awkward target: 96% of policies cost nothing, and the rest are positive and heavily skewed. A Poisson model is for counts, and a Gamma model can't produce a zero. A Tweedie distribution with power between 1 and 2 fits exactly this shape, because it's a compound Poisson-Gamma: a Poisson number of claims, each with a Gamma-distributed cost, added up. The power sets how variance grows with the mean. I chose 1.8, which is what our claim-size distribution implies, and validation showed results were flat between 1.4 and 1.9.

- When would you prefer frequency x severity over a single Tweedie model?

I'd usually prefer frequency × severity because it shows you which part is driving the price. For example, young drivers claim much more often, but their claims don't cost much more. Each part can also have its own factors: our severity model only needed three, while frequency needed eight. It also lets you trend claims inflation separately from frequency. On our data it also predicted better: the frequency × severity GBM beat the Tweedie GBM, probably because severity has so little signal that keeping it separate stops cost noise drowning out the frequency signal. A single Tweedie model makes sense when you want one simpler model to build and maintain, or when you only have total cost and no reliable claim counts.

## 4. Evaluation
- What does deviance measure, and why not RMSE?
- What does a Gini of 0.3 mean in practice?
- What does a double-lift chart show that a single metric doesn't?

## 5. Commercial analysis
- The GBM beat the GLM. Would you deploy it?
- What would an underwriter, a regulator and a pricing manager each want to know first?

## 6. README
- Explain the whole project in two minutes to a non-technical interviewer.
