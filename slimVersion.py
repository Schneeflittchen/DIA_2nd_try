import numpy as np
from Hungarian_Matcher import *
from knapsack import *
#Time horizon in days

T=100

#number of websites
nr_websites=4

#Buddget of our advertizer
budget=1000
campaign_budget=budget/nr_websites
campaign_daily_budget=campaign_budget/T

daily_budget = budget / T

#number of slots (ads) per website
#nr_slots=[5 for i in range(nr_websites)] #different numberr of slots for each website
nr_slots=5 #using the same amount of slots for all websites

#number of advertizers per website
nr_ads=8 #using the same advertizers/number of ads on all websites - this number includes our advertiser

#Daily users visiting a website (same for every website, as different dataset sizes would bias the result and hinder an direct comparison)
daily_users_mean=808
daily_users_deviation=144


def get_advertiser_utility(bid, expected_clicks, average_bid,num_slot,num_advertiser):
    prob = (bid/average_bid) * (num_slot/num_advertiser) #probability of winning the auction
    utility = prob * expected_clicks  # utility is the reward

    return utility

def get_daily_users():
    return np.random.normal(daily_users_mean,daily_users_deviation)
    #return np.random.normal(daily_users_mean,daily_users_mean,nr_websites)

#User class probability, defined as twice a Binomial drawing of the binary features
    #we generate separate users for each website (so they have 1. different user classes and 2. different occurrence probabilities of user classes), to stay more realistic
feature1_probability=np.random.uniform(0.01,1,nr_websites) #using uniform to get some rather different and maybe extreme patterns
feature2_probability=np.random.uniform(0.01,1,nr_websites)
class_probabilities=[ [feature1_probability[i]*feature2_probability[i],
                    (1-feature1_probability[i])*feature2_probability[i],
                    feature1_probability[i]*(1-feature2_probability[i]) ] for i in range(nr_websites)] #because each website only receives 3 of the theoretically 4 user classes, we will neglect one class per website (always the same class, since the uniform distribution allows to do this without loss of generality)

def get_user_classes(nr_users):
     return [np.random.choices((0,1,2),class_probabilities[i,:],nr_users) for i in range(nr_websites)]

#click probabilities on the different websites per user class (amount=3), to click on ad i in slot j
    #click probabilites of the different ads in the different slots for each website vary, as every website can display them differently and attracts different users
Q=np.random.rand(nr_websites,3,nr_ads,nr_slots) #again uniform distribution, assume each entry where index nr_ads=0 belongs to our advertiser

#define stochastic advertisers
    #on average they should bid as much as our advertiser (to allow every advertiser to win some auctions)
    #each advertiser bids only once per day, a daily constant bid per subcampaign
def get_stochastic_bids(nr_auctions):
    return [[np.random.normal(campaign_daily_budget/nr_auctions, campaign_daily_budget/nr_auctions/5, nr_auctions) for i in range(nr_ads-1)] for k in range(nr_websites)]

#TS-Learner and Plot variables
beta_params=np.ones((nr_websites,4,2)) #each website has its own TS learner, which learns the 3 user class demand curves + 1 aggregated curve, with 2 parameters for each beta distribution
rewards=[[] for i in range(nr_websites)]
clairvoyant_rewards=[[] for i in range(nr_websites)]
budget_discretization_steps = 20
#Matching algorithm for publishers: Hungarian for matching each publisher's ads and slots - Alireza
def hungarian_matcher(weights):
    hr = Hungarian_Matcher(weights)
    idxs = hr.hungarian_algo()
    return  idxs[0]

    return idx #the index of the ad, in the order of the slots, e.g. in idx=[2,5,1,3,7], ad 2 is chosen for slot 1, ad 5 for slot 2 etc.

for day in range(T):
    if day%7 == 0: #at beginning and every 7 days the contexts are defined
        context=0

    nr_daily_users=get_daily_users()

    #sampling the user classes of each website's visitors
    user_classes = get_user_classes(nr_daily_users)




    # expected_clicks_per_subcampaign
    ## TODO will drawn from TS
    expected_clicks = []
    for subcampaign in range(nr_websites):
        print('TODO will drawn from TS')
        #expected_click[i] = TS.draw()


    step = daily_budget / (len(nr_websites) - 1)
    bids = [i * step for i in range(len(nr_websites))]
    estimated_utilities = []
    for subcampaign in range(nr_websites):
        estimated_utilities.append([get_advertiser_utility(bid,expected_clicks[subcampaign], campaign_daily_budget/daily_users_mean) for bid in bids])

    optimum_allocation = Knapsack(daily_budget, estimated_utilities).optimize()  # output is 2D array [[subcampaing bid] [subcampaing bid] [subcampaing bid] [subcampaing bid]]

    ###using knapsack to determine our advertisers bid - Talip
    bid0 = 42
    #receiving the stochastic advertiser's bids
    bids= [bid0].append(get_stochastic_bids(nr_daily_users))


    for auction in range(nr_daily_users): #evaluating each auction individually

        ### (pt6) using MULTI-knapsack to determine our advertisers bid AND BUDGET (Hungarian) - Volunteers

        ###Matching of ads and slots for each publisher
        for i in range(nr_websites):
            # using the TS-sample for our advertiseers click probability
            selected_ad = [hungarian_matcher(np.random.beta(beta_params[i, 0, 0], beta_params[i, 0, 1]) * bids[0],
                              Q[i, user_classes[i, auction], 1:, :] * bids[i])]
            rewards[i].append(0) #add at least a 0 reward
            for s in range(nr_slots):
                #getting a user input
                reward = np.random.binomial(1, Q[i,user_classes[i,auction],selected_ad[s],s])*bids[i]  # Bernoulli
                if reward > 0: #if clicked
                    rewards[i,-1] += reward
                    if selected_ad[s]==0: #if our advertisers ad was displayed in the slot: update beta-distribution (since clicked)
                        beta_params[i,0,0] += reward/bids[0]
                        beta_params[i,0,1] += (1-reward/bids[0])
                    else:
                        if any([selected_ad[k] for k in range(s+1)]==0):
                            beta_params[i, 0, 1] += 1 #in case our advertiser was shown before, but not clicked, update beta-distribution accordingly with a reward=0
                    break
                else:
                    if s == nr_slots-1: #if no ad was clicked at all
                        if any(selected_ad==0):
                            beta_params[i, 0, 1] += 1 #in case our advertiser was shown before, but not clicked, update beta-distribution accordingly with a reward=0

            # using the clairvoyant algorithm
            selected_ad_clairvoyant = [hungarian_matcher(Q[i, user_classes[i, auction], 0, :] * bids[0],
                                             Q[i, user_classes[i, auction], 1:, :] * bids[i])]
            clairvoyant_rewards[i].append(0)
            for s in range(nr_slots):
                reward = np.random.binomial(1, Q[i, user_classes[i, auction], selected_ad[s], s]) * bids[i]  # Bernoulli
                if reward>0:
                    clairvoyant_rewards[i,-1] += reward


        ###VCG-auctions for our advertisers payments - Anne

        ###using clairvoyant implementations, determine the cumulative regrets of each learner - Volunteers??



###PLOTS

#TODO: del or use the following code
"""
Probs this code is not being used

#maximum matching for ads to slots: weights=probability*bid
# click-probabilities of ad i on slot j is given in matrix q (row per ad, column per slot), bids for each ad are given in vector b (same order as q)
# it is a dense bipartite graph, and thus best solved with Hungarian algorithm using adjacency matrix
def maxMatch(q,b):
    weights=q.dot(b)
    max_ad=[]
    for slot in weights.T:
        max_ad.__add__(np.where(slot == np.amax(slot))[0]) #determining the displayed ad: the max value in the weights' column is the best choice of ad for the slot, which is the [0] parameter when transposed
"""
