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
    return [np.random.normal(campaign_daily_budget/nr_auctions, campaign_daily_budget/nr_auctions/5, nr_ads) for k in range(nr_websites)]

#utilities for our advertiser's bid (task 6)
def get_advertiser_utility(bid, expected_clicks, average_bid,num_slot,num_advertiser):
    prob = (bid/average_bid) * (num_slot/num_advertiser) #probability of winning the auction
    utility = prob * expected_clicks  # utility is the reward
    return utility

budget_discretization_steps = 20

#TS-Learner (beta distribution), context generation and plot variables
    #each website has its own TS learner, which learns the 3 user class demand curves + 3 aggregated curves (only splitting one or no feature each),
    # with as many arms as slots and 2 parameters for each beta distribution
    #order: 0= 0 features split, 1=first user class, 2=2nd user class, 3=3rd user class, 4=first feature split=1st and 3rd user class, 5=2nd feature split= 1st and 2nd user class
beta_params=np.ones((nr_websites,6,nr_slots,2))
    #rewards-order acc to bandit-param order: 0=basic TS, 1=1.class bandit,2=2.class, 3=3.class, 4=1.&3.class, 5=1.&3. class,
    #additionally clairvoyant rewards: 6= for task 3, 7= for task 4, 8= for task 6, 9= for task 7
    #for context generation, use click probabilites as rewards
rewards=np.zeros((nr_websites,8))
cumulative_regret3=[[] for i in range(nr_websites)] #for task 3
cumulative_regret4=[[] for i in range(nr_websites)] #for task 4
cumulative_regret6=[[] for i in range(nr_websites)] #for task 6
cumulative_regret7=[[] for i in range(nr_websites)] #for task 7

def get_weights(website,bandit,user_class):
    our_adv_weights = np.multiply(np.random.beta(beta_params[website, bandit, :, 0], beta_params[website, bandit, :, 1]), bids[website][0])[newaxis].T
    other_adv_weights = np.multiply(Q[website, user_class, 1:, :].T, bids[website][1:])
    return np.hstack(our_adv_weights, other_adv_weights)

def sample(i, user_class, bandit): #i=nr of website, user_class=nr of user class acc. to Q definition {0,1,2}, bandit=nr of bandit acc to rewards order
    matcher = Hungarian_Matcher(get_weights(i, bandit, user_class))
    selected_ad = np.array(np.argwhere(matcher.hungarian_algo()[0] > 0)[:, 1])
    for s in range(nr_slots): # getting a user input
        reward = np.random.binomial(1, Q[i, user_class, selected_ad[s], s]) * bids[i][selected_ad[s]]  # Bernoulli
        if reward > 0:  # if clicked
            rewards[i, bandit] += reward
            if selected_ad[s] == 0:  # if our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                beta_params[i, bandit, 0] += 1
            else:
                if any([selected_ad[k] for k in range(s + 1)] == 0):
                    beta_params[i, bandit, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)
            break  # at max one ad clicked on a website per auction
        elif s == nr_slots - 1:  # if no ad was clicked at all
            if any(selected_ad == 0):
                beta_params[i, bandit, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)



nr_total_users=0
nr_total_users_per_class=np.zeros((nr_websites,3)) #3 classes for each website
def beta_expectation(alpha,beta):
    return np.divide(alpha,np.add(alpha,beta))

def lower_bound(mean, dataset, confidence=0.95):
    return np.subtract(mean,np.sqrt(np.divide(-np.log(confidence)/2,dataset)))

#VCG-Auction utilities:
# Generate prominence rate
def slots_click_probas(nr_ads, nr_slots):
    click_probas = np.zeros(nr_ads)
    click_probas[0] = random.randint(0, 100) / 100
    for i in range(1, nr_slots):
        click_probas[i] = random.randint(0, click_probas[i - 1]) / 100
    # probas in the other slots, decreasing while the slot index increases
    return click_probas


proba_slots = [slots_click_probas(nr_ads, nr_slots) for i in range(nr_websites)]


# lambda vector (prominence rates) in the slides


def slot_allocation(nr_ads, prominence, quality, bids, nr_slots):
    slot = [-1 for i in range(nr_ads)]
    quality_price = quality * np.transpose(bids)  # q_a * v_a
    for i in range(nr_slots):
        slot[i] = np.argmax(quality_price)
        quality_price[slot[i]] = 0
    return slot
    # slot[a] = value of the slot for an advertiser a (-1 if none)


def reverse_slot_allocation(nr_ads, prominence, quality, bids, nr_slots):
    slot_alloc = slot_allocation(nr_ads, proba_slots, quality, bids, nr_slots)
    # for each slot, corresponding ad
    slots = np.zeros(nr_slots)
    for i in range(nr_ads):
        if slot_alloc[i] != -1:
            slots[slot_alloc] = i
    return slots
    # slots[i]=a is the advertiser a corresponding to slot i


def optimal_SW_without_a(a, nr_ads, prominence, quality, bids, nr_slots):
    X = 0
    quality2 = quality
    bids2 = bids
    for i in range(a, nr_ads - 1):
        quality2[i] = quality2[i + 1]
        bids2[i] = bids2[i + 1]

    # allocation of slots without a
    slots = reverse_slot_allocation(nr_ads - 1, prominence, quality2, bids2, nr_slots)
    for i in range(nr_slots):
        a_ = slots[i]
        X += prominence[i] * quality2[a_] * bids2[a_]
    return X


for day in range(T):
    if day%7 == 0: #at beginning and every 7 days the contexts are defined
        for i in nr_websites:
            # expected mean (reward) of a campaign bandit by averaging the expectation values of each of its slot
            u_mean = np.average(beta_expectation(beta_params[i,:,:,0],beta_params[i,:,:,1]),1)
            u_lowerbound = lower_bound(u_mean,nr_total_users)
            p_mean_occurrence_class=np.divide(nr_total_users_per_class[i,:],nr_total_users)
            p_lowerbound_occurrence_class = lower_bound(p_mean_occurrence_class,nr_total_users_per_class[i,:])

            # theoretically, only one feature split at a time, but practically we seek the highest expected reward
            split_feature = argmax(u_lowerbound[0],
                                   (p_lowerbound_occurrence_class[0]+p_lowerbound_occurrence_class[2])*u_lowerbound[4]+p_lowerbound_occurrence_class[1]*u_lowerbound[2],
                                   (p_lowerbound_occurrence_class[0]+p_lowerbound_occurrence_class[1])*u_lowerbound[5]+p_lowerbound_occurrence_class[2]*u_lowerbound[3],
                                   p_lowerbound_occurrence_class[0]*u[1]+p_lowerbound_occurrence_class[1]*u_lowerbound[2]+p_lowerbound_occurrence_class[2]*u_lowerbound[3])

    nr_daily_users = get_daily_users()
    nr_total_users+=nr_daily_users

    #sampling the user classes of each website's visitors
    user_classes = get_user_classes(nr_daily_users)
    for i in nr_websites:
        _ , class_count =np.unique(user_classes[i],return_counts=True)
        np.add(nr_total_users_per_class[i,:],class_count)

    #receiving the stochastic advertiser's bids
    bids= get_stochastic_bids(nr_daily_users)

    ###using knapsack to determine our advertisers bid - task 6
    # expected_clicks_per_subcampaign
    ## TODO will drawn from TS
    expected_clicks = []
    for subcampaign in range(nr_websites):
        print('TODO will drawn from TS')
        # expected_click[i] = TS.draw()

    step = daily_budget / (len(nr_websites) - 1)
    bids = [i * step for i in range(len(nr_websites))]
    estimated_utilities = []
    for subcampaign in range(nr_websites):
        estimated_utilities.append(
            [get_advertiser_utility(bid, expected_clicks[subcampaign], campaign_daily_budget / daily_users_mean) for bid
             in bids])

    optimum_allocation = Knapsack(daily_budget,
                                  estimated_utilities).optimize()  # output is 2D array [[subcampaing bid] [subcampaing bid] [subcampaing bid] [subcampaing bid]]


    bid0 = np.random.normal(42,4,4)
    bids[:][0]=bid0

    for auction in range(nr_daily_users): #evaluating each auction individually

        ### (pt6) using MULTI-knapsack to determine our advertisers bid AND BUDGET (Hungarian) - Volunteers

        ###Matching of ads and slots for each publisher, using Thompson-Sampling-Bandits to find our advertisers click probabilities
        for i in range(nr_websites):
            # using the TS-sample for our advertiseers click probability
            sample(i,user_classes[i, auction],0)

            # using the clairvoyant algorithm for task 3
            cv3_matcher = Hungarian_Matcher(np.multiply(np.average(Q[i, :, :, :],0).T,bids[i])) #averaging class' click probabilities to get clairvoyant aggregated probability
            selected_ad_clairvoyant3 = np.array(np.argwhere(cv3_matcher.hungarian_algo()[0]>0)[:,1])
            for s in range(nr_slots):
                reward = np.random.binomial(1, Q[i, user_classes[i, auction], selected_ad_clairvoyant3[s], s]) * bids[i][selected_ad_clairvoyant3[s]]  # Bernoulli
                if reward>0:
                    rewards[i,6]+=reward

            #using different TS-Bandits to train for different contexts in task 4
            if user_classes[i][auction]==0:
                #context of split feature 1
                sample(i, 0, 4)
                #context of split feature 2
                sample(i, 0, 5)
                #context of split feature 1 & 2
                sample(i, 0, 1)
            elif user_classes[i][auction]==1:
                #context of split feature 1
                sample(i, 1, 2)
                #context of split feature 2
                sample(i, 1, 5)
                #context of split feature 1 & 2
                sample(i, 1, 2)
            elif user_classes[i][auction]==2:
                #context of split feature 1
                sample(i, 2, 4)
                #context of split feature 2
                sample(i, 2, 3)
                #context of split feature 1 & 2
                sample(i, 2, 3)

            # using the clairvoyant algorithm for task 4
            cv4_matcher = Hungarian_Matcher(np.multiply(Q[i, user_classes[i, auction], :, :].T, bids[i]))
            selected_ad_clairvoyant4 = np.array(np.argwhere(cv4_matcher.hungarian_algo()[0] > 0)[:, 1])
            for s in range(nr_slots):
                reward = np.random.binomial(1, Q[i, user_classes[i, auction], selected_ad_clairvoyant4[s], s])*bids[i][selected_ad_clairvoyant4[s]]  # Bernoulli
                if reward > 0:
                    rewards[i,7]+=reward


        ###VCG-auctions for our advertisers payments
        quality = [random.randint(0, 100) for i in range(nr_ads)]  # to change
        for user in range(nr_daily_users):
            # depends on slots_click_probas, quality of ads, bids, daily_budgets
            # for one website for now

            # check if our advertiser does not exceed his daily budget
            # (we assume that the others have an unlimited budget)
            # first order - slot allocation by q_a * bid
            quality_price = quality * np.transpose(bids)
            lambda_q = proba_slots * np.transpose(quality)
            # slot allocation by the publisher: slot index if allocated ad, -1 otherwise
            slot_alloc = slot_allocation(nr_ads, proba_slots, quality, bids, nr_slots)
            # for each slot i, corresponding ad
            slots = reverse_slot_allocation(nr_ads, proba_slots, quality, bids, nr_slots)
            # payments post-auction
            payments = np.zeros(nr_ads)
            for i in range(nr_slots):
                a = slots[i]  # index of the advertiser chosen for slot i
                X = optimal_SW_without_a(a, nr_ads, proba_slots, quality, bids, nr_slots)
                Y = [prominence[j] * quality[slots[j]] * bids[slots[j]] for j in range(nr_slots)]
                payments[a] = (X - Y) / (quality[a] * proba_slots(i))
            # update budget
            for i in range(nr_slots):
                if slot_alloc[i] == 0:
                    budget -= bids[0]  # let's assume that our advertiser's index is 0

        ###using clairvoyant implementations, determine the cumulative regrets of each learner - Volunteers??



###PLOTS
