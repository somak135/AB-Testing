# june-2020

## Confounder Logistic Model
`confounder.R` implements this. //needs to be updated yet

## Bayesian A/B Testing method
`1.py` implements the numerical procedure of finding required sample size.
  
`2.py` carries out simulation study to check the comparative performance of bayesian method and frequentist(aka classical) method.


### Classical Method

Suppose the loss function is denoted by $\mathscr{L}(\text{parameter, decision})$. Then, for classical method, the loss function is a binary one, and looks like this:
$$\mathscr{L}(p_c, p_t, \text{ Accept null}) = \begin{cases} 0 & \text{if $p_c=p_t$}\\ 1 & \text{if $p_c \ne p_t$} \end{cases}$$
$$\mathscr{L}(p_c, p_t, \text{ Reject null}) = \begin{cases} 1 & \text{if $p_c=p_t$}\\ 0 & \text{if $p_c \ne p_t$} \end{cases}$$

Let us denote $\theta = \{p_c, p_t\}$. The risk$(R(\theta))$, also known as the expected loss, follows from the above definition of loss function.
$$R(\theta) = \begin{cases} P_\theta(\text{accept null}) & \text{if } p_c \ne p_t \\[.5cm] P_\theta(\text{reject null}) & \text{if } p_c = p_t \end{cases}$$

Notice that, these are the type-I and type-II errors! Level of significance thus keeps a bound on the Risk when $p_c = p_t$. 
But, the main question is: Is it sensible to have same loss for two scenarios like $p_c = 0.2, p_t = 0.25$ and $p_c = 0.2, p_t = 0.205$ and in both case accepting the null?

Also, why is it even considered a loss when $p_c = p_t$ and we end up rejecting null. Either way, we leave with same conversion rate we were already having and incur no loss.

$$N = \left[ \sqrt{2 \bar{p}\bar{q}} z_{1-\alpha/2} + \sqrt{p q + p' q'} z_{\beta} \right]^2 \bigg/ \delta^2$$


### Bayesian Method

The trick is in taking a more sensible loss function. Let us quantify the loss exactly in the way they are supposed to be.

$$\mathscr{L}(p_c, p_t, \text{ Control winner}) = \begin{cases} p_t - p_c & \text{if $p_c<p_t$}\\ 0 & \text{if $p_c \ge p_t$} \end{cases}$$
$$\mathscr{L}(p_c, p_t, \text{ Treatment winner}) = \begin{cases} p_c - p_t & \text{if $p_t < p_c$}\\ 0 & \text{if $p_t \ge p_t$} \end{cases}$$

So, with same notations as above, the risk aka the expected loss is
$$R(\theta) = \begin{cases} E(\mathscr{L}(p_c, p_t, \text{ Control winner})) & \text{if $p_c<p_t$}\\[.5cm] E(\mathscr{L}(p_c, p_t, \text{Treatment winner})) & \text{if $p_c>p_t$} \end{cases}$$

Does the loss function make more sense now?
