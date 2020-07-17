# DIA_2nd_try

HANDIN: JUST GITHUB REPO & SLIDES OF PLOTS (plus some explanations)

PseudoCode:

Initialisation
 •	Define time horizon T\
 •	Initialize 4 websites with respective subcampaigns of (4+) different advertizers
 •	Define the 3 occuring user classes features (ie. Pop the 4th combination of the 2 binary features) per subcampaign (so all 4 classes can occur, and actually even change by the campaign (on diff. sites we could get different context data…) – but this does not apply when we just randomly sample only binary values anyway)
 •	For each subcampain & user class:
  o	define a stochastic measure of users (class type) per day
  o	The user class’ specific probability to click on an ad (from a certain subcampaign) i in slot j q_i,j (each website has 4+ slots, where the amount can differ between publishers)
  o	Set some stochastic advertiser’ bidding model
 •	Define budget for our advertiser (pt5)

Daily Routine:
 •	Combinatorial bandit algorithm estimates q_i,j for our advertisers campaigns: Problem: the other advertisers bids and qs are known: yield a fixed value (no need for a bandit) -> Still handle them as possible arms, while just their distribution has 0 variance, and the bandit would pull them as soon as they seem more promising than the other arms (including our advertiser’s, which is not known precisely)
 •	Advertisers give bids (per subcampaign): truthful bid/value of a publisher?? (caveat: it is not per slot!!) -> place a bid for unlimited (pt3,knapsack) and limited (pt5f, use knapsack for bids and vcg-auction for payments) budget, in the latter case (acc to pt6) define the daily bid&budget (for each subcampaign) by an optimizer
 •	Matching algorithm: maximizes each publisher’s reward theoretically (individually) for both cases/bids per advertiser -  just twice a bipartite matching with input nodes (ads), connected to output nodes (slots) by weight_i,j= q_i,j *b_i, – Q: is it okay to display the same ad several times?? Assumed yes, as a bid only is payed once as defined by the problem statement, while the restriction itself is not mentioned
  o	Save rewards of the bandits and add the one of the clairvoyant algo for plotting the cumulative regret
 •	For pt6: do the matching of the optimized bids with the clairvoyant matcher knowing all q_i,j values. Save the rewards for later plots
 •	Acc. to pt4
  o	collect context data (sort click probabilities per user class)
  o	After 7 days: generate the contexts for the next week (only for pt3-algo) (ie. Decide whether we want to split a feature (Can split several on 1 weekend??), and accordingly determine the bandits to use in which case (should we switch bandits entirely, like copy the aggregated model, and feed it henceforth only with the context data to specialize it/the two new bandits?) -> this approach should yield the best reachable reward; besides for pt3, the aggregated model still lives on)
  o	use all data for context in pt5,6,7 (so using 3 bandits on each campaign, as we know its is 3 user classes occurring) ->this will yield some less reward, as we directly take on the disaggregated model
  
Termination:
 •	Plot all the cumulative regrets:
  o	Learning Matcher with unlimited budget-bids (pt3)
  o	Context-regret (pt4)
  o	Learning advertiser (limited budget, clairvoyant matcher – pt6)
  o	Separately for matcher&advertiser (pt7)


Qs:
Pt 4: Bandit after splitting feature: use new one, train one beforehand, or derive/copy from the so far used bandit
