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

Deviance measures how far predictions are from a perfect fit, judged against the noise the distribution expects at each prediction. So a miss on a policy that's expected to be expensive counts for less, because a bigger miss was expected there. RMSE squares every error and treats them all equally, and with insurance data, mostly zeros and a few huge claims, a handful of policies decide the score. In my project, 55% of the squared error came from 10 policies out of 135,000, and RMSE rated the GLM, the GBM and charging everyone the average within 0.1% of each other. Deviance separated them clearly.
- What does a Gini of 0.3 mean in practice?

The Gini measures risk ranking: how well the model sorts policies from cheapest to most expensive, not whether the overall price level is right. It comes from the Lorenz curve: 0 is random ranking and 1 is perfect. In practice, with a Gini of about 0.3, the 10% of policies the model rates riskiest produce about 25% of the claims, against 10% if the ranking were random. That's typical for motor, because individual claims are largely chance. Better ranking matters because it lets you undercut competitors on good risks, while competitors who rank worse end up with the bad risks you've priced properly.


- What does a double-lift chart show that a single metric doesn't?

A single metric tells you which model is better on average, but not where two models disagree or which one is right there. Two models can have similar Ginis and still price very different customers very differently. A double lift sorts policies by the ratio of the two models' predictions, cuts them into bands, and plots actual results against both models in each band. Whichever line the actuals follow is the model that's right about those customers. In my project, where the GBM priced 1.7 times higher than the GLM, actual claims came in at 1.9 times average, so the GLM was underpricing those customers by about 80%.

## 5. Commercial analysis
- The GBM beat the GLM. Would you deploy it?

I wouldn't replace the GLM outright. I'd take it in two steps. First, I'd feed what the GBM found back into the GLM. A single flag for customers who've claimed their way off the no-claims discount path closed about half the gap, and it stays fully explainable. Second, I'd consider a hybrid: the GLM sets the base price, with a GBM adjustment capped at something like ±15%, so the part that's harder to explain can only move a price so far. That keeps the transparency, governance and fairness testing of a GLM while capturing more of the GBM's better risk ranking. On the test set, the GLM was undercharging one group of customers by about €1.3m. Before going live, I'd validate on another year of data, test the outcomes for fairness, and ideally build a demand model to see how customers react to the price changes.
- What would an underwriter, a regulator and a pricing manager each want to know first?

Each would start from a different angle. An underwriter manages risk, so they'd want to know which customers' prices change and whether it makes sense. For example: why does a BonusMalus of 52 cost more than 51? The answer is that 52 can only be reached after a claim. A regulator is focused on fair outcomes, so they'd ask whether we can explain an individual price, whether the model disadvantages groups with protected characteristics, including through proxies, and whether it complies with the FCA's pricing rules and Consumer Duty. A pricing manager cares about results and delivery, so they'd want the impact on profit and customer numbers, how many customers see large price changes and whether they'd need phasing in, and what it costs to implement and monitor."

## 6. README
- Explain the whole project in two minutes to a non-technical interviewer.
