import numpy as np

#Time horizon in days
T=100

#number of websites
nr_websites=4

#Buddget of our advertizer
budget=1000
campaign_budget=budget/nr_websites
campaign_daily_budget=website_budget/T

#number of slots (ads) per website
#nr_slots=[5 for i in range(nr_websites)] #different numberr of slots for each website
nr_slots=5 #using the same amount of slots for all websites

#number of advertizers per website
nr_ads=8 #using the same advertizers/number of ads on all websites

#Daily users visiting a website (same for every website, as different dataset sizes would bias the result and hinder an direct comparison)
daily_users_mean=808
daily_users_deviation=144

def get_daily_users():
    return np.random.normal(daily_users_mean,daily_users_deviation)
    #return np.random.normal(daily_users_mean,daily_users_mean,nr_websites)

#User class probability, defined as twice a Binomial drawing of the binary features
    #we generate separate users for each website (so they have 1. different user classes and 2. different occurrence probabilities of user classes), to stay more realistic
feature1_probability=np.random.uniform(0.01,1,nr_websites) #using uniform to get some rather different and maybe extreme patterns
feature2_probability=np.random.uniform(0.01,1,nr_websites)
pop_user=np.random.randint(0,4,nr_websites) #because each website only receives 3 of the 4 user classes, we will pop one class per website by re-drawing/neglecting if the unwanted class occurs

def get_user_classes(nr_users,feature1_probability,feature2_probability):
     return [ [np.random.binomial(1, feature1_probability, nr_users)],
              [np.random.binomial(1, feature2_probability, nr_users)] ] #binomial with n=1 consecutive trials equals the bernoulli distribution with 0 or 1 as outcome in each run/entry

#click probabilities on the different websites per user class (amount=3), to click on ad i in slot j
    #click probabilites of the different ads in the different slots for each website vary, as every website can display them differently and attracts different users
Q=np.random.rand(nr_websites,3,nr_ads,nr_slots) #again uniform distribution

#define stochastic advertisers
    #on average they should bid as much as our advertiser (to allow every advertiser to win some auctions)
    #each advertiser bids only once per day, a daily constant bid per subcampaign
def get_stochastic_bids(nr_auctions):
    return [np.random.normal(campaign_daily_budget/nr_auctions, campaign_daily_budget/nr_auctions/5, nr_websites) for i in range(nr_ads)]



for day in range(T):
    if day%7 == 0: #at beginning and every 7 days the contexts are defined
        context=0

    ###using a bandit (TS) to estimate the q_i,j of our advertisers subcampaigns on each website - Me

    ###using knapsack to determine our advertisers bid - Talip

    ### (pt6) using MULTI-knapsack to determine our advertisers bid AND BUDGET (Hungarian) - Volunteers

    ###Hungarian for Matching each publisher's ads and slots - Alireza

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