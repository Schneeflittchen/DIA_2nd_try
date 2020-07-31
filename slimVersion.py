import numpy as np
from Hungarian_Matcher import *
from knapsack import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import cProfile

#GENERAL

    #Time horizon in days
T=100

    #number of websites
nr_websites=4

    #number of slots (ads) per website
nr_slots=4 #using the same amount of slots for all websites

    #number of advertizers per website
nr_ads=6 #using the same advertizers/number of ads on all websites - this number includes our advertiser

DoPrint = False #prints a weekly update to the console

#USER DISTRIBUTIONS
    #Daily users visiting a website (same for every website, as different dataset sizes would bias the result and hinder an direct comparison)
daily_users_mean=100
daily_users_deviation=30

nr_total_users=np.array([0])
nr_total_users_per_class=np.zeros((nr_websites,3)) #3 classes for each website

def get_daily_users():
    return max(int(np.random.normal(daily_users_mean,daily_users_deviation)+0.5),1)

    #User class probability, defined as twice a Binomial drawing of the binary features
    #we generate separate users for each website (so they have 1. different user classes and 2. different occurrence probabilities of user classes), to stay more realistic
feature1_probability=np.random.uniform(0.01,1,nr_websites) #using uniform to get some rather different and maybe extreme patterns
feature2_probability=np.random.uniform(0.01,1,nr_websites)
class_probabilities=[ np.array([feature1_probability[i]*feature2_probability[i],
                    (1-feature1_probability[i])*feature2_probability[i],
                    feature1_probability[i]*(1-feature2_probability[i]) ]) for i in range(nr_websites)] #because each website only receives 3 of the theoretically 4 user classes, we will neglect one class per website (always the same class, since the uniform distribution allows to do this without loss of generality)
    #normalising
for probs in class_probabilities:
    #probs=np.divide(probs,probs.sum()) - does not work with np.random.choice, for some reason the precision is too low :/
    probs[2]=1-probs[0]-probs[1] #ugly and untrue with regards to class_probabilities[2] probability definition, but works


def get_user_classes(nr_users):
    return [np.random.choice(3,nr_users,p=class_probabilities[i]) for i in range(nr_websites)]

#CLICKING PROBABILITIES
    #click probabilities=ad_quality_i on the different websites per user class (amount=3)
Q=np.random.rand(nr_websites,3,nr_ads) #again uniform distribution, assume each entry where index nr_ads=0 belongs to our advertiser

#truncate so we wont have too low values of Q and some ads (esp. from our advertiser) would not show at all
Q[Q<0.2]=0.2
Q[Q>0.9]=0.9
Q[Q[:,:,0]<0.5]=0.5 #our advertiser gets a good chance to be clicked, as the product is extremely cool (and this prevents excessivey long runs of the algorithm)

    #lambda vector (prominence rates) - known beforehand for each advertiser (realistic, as they have experience from the past of their website's ads)
prominence_rates=np.random.rand(nr_websites,nr_slots)

def user_is_looking_at(i,s): # check if user looks at ad in slot s at  all with given prominence rates
    if s == 0:
        if np.random.binomial(1, prominence_rates[i, 0]) == 0:  # Bernoulli - user lost interest
            return False
    elif np.random.binomial(1,min(1,prominence_rates[i, s] / prominence_rates[i, s - 1])) == 0:  # Bernoulli - user lost interest
        return False
    return True

    #get Qij: holding q_i,j variables as described in the problem statement, for each website and user class
def get_cv_CTRs(website,user_class):
    ctr = np.zeros([nr_slots, nr_ads])
    ctr[0, :] = np.multiply(Q[website,user_class, :], prominence_rates[website, 0])
    for i in range(nr_slots - 1):
        ctr[i + 1, :] = np.multiply(ctr[i,:], min(1,prominence_rates[website, i + 1] / prominence_rates[website, i])) #maxing function to ceil the the prominence_rates: a user can not more than definitely observe a certain ad; it does not imply directly a click
    return ctr.T #returns the true click through rates by multiplying the ad qualities with the prominence rates

Qij= np.zeros([nr_websites,3, nr_ads, nr_slots]) #q_i,j variables as described in the problem statement, for each website and user class
for i in range(nr_websites):
    for k in range(3):
        Qij[i,k,:,:]=get_cv_CTRs(i,k)

    #get Qij if user class is neglected:
def get_aggregated_cv_CTRs(website):
    ctr = np.zeros([nr_slots, nr_ads])
    ctr[0, :] = np.multiply(np.average(np.multiply(Q[website, :, :],class_probabilities[website][:,None]),0), prominence_rates[website, 0])  # averaging user classes to get aggregated demands
    for i in range(nr_slots - 1):
        ctr[i + 1, :] = np.multiply(ctr[i,:], min(1,prominence_rates[website, i + 1] / prominence_rates[website, i])) #maxing function to ceil the the prominence_rates: a user can not more than definitely observe a certain ad; it does not imply directly a click
    return ctr.T #CTRs, aggregated over all user classes, applied per website

Qij_aggregated= np.zeros([nr_websites, nr_ads, nr_slots]) #similar to Qij, but with aggregated/averaged user class demands
for i in range(nr_websites):
        Qij_aggregated[i,:,:]=get_aggregated_cv_CTRs(i)


#ADVERTISERS
    #define stochastic advertisers
    #on average they should bid as much as our advertiser (to allow every advertiser to win some auctions)
    #each advertiser bids only once per day, a daily constant bid per subcampaign
def get_stochastic_bids(nr_auctions):
    return [np.random.normal(campaign_daily_budget/nr_auctions, campaign_daily_budget/nr_auctions/5, nr_ads) for k in range(nr_websites)]

    #Buddget of our advertizer
budget=np.array([1000] * 3)
campaign_budget=budget[0]/nr_websites
campaign_daily_budget=campaign_budget/T
daily_budget = budget[0] / T

    #utilities for our advertiser's bid (task 6)
def get_advertiser_utility(bid, expected_clicks, average_bid):
    prob = (bid/average_bid) * (nr_slots/nr_ads) #some measure to estimate the advertisers ad_value
    return prob * expected_clicks  # utility is the reward

budget_discretization_steps = 20
step = daily_budget / budget_discretization_steps
stepped_bids = [i * step for i in range(budget_discretization_steps)]

def get_bids(bandit=0,clairvoyant=False):
    # expected_clicks_per_subcampaign: TS to learn average Click thorugh rate per subcampaign -task 6&7
    if not clairvoyant:
        expected_clicks = []
        for subcampaign in range(nr_websites):
            expected_clicks.append(np.random.beta(beta_params_advertiser[subcampaign,bandit, 0], beta_params_advertiser[subcampaign,bandit, 1]))
    else:
        expected_clicks =np.average(Q[:, :, 0], 1)

    estimated_utilities = []
    for subcampaign in range(nr_websites):
        estimated_utilities.append(
            [get_advertiser_utility(bid, expected_clicks[subcampaign], campaign_daily_budget / daily_users_mean) for bid
             in stepped_bids])

    bids = Knapsack(daily_budget, estimated_utilities).optimize()  # optimum_allocation: output is 2D array [[subcampaing bid] [subcampaing bid] [subcampaing bid] [subcampaing bid]]
    bids.sort(key=lambda entry:entry[0]) #sort so we get the conventional order by website/subcampaign
    q=np.zeros(len(bids))
    for i in range(len(bids)):
        q[i]=bids[i][1]
    return q #holds only the bids, sorted by subcampaign


#TASK-SPECIFICS
    #TS-Learner (beta distribution) for task 3,4,6,7. Additional utilities for context generation and plotting
    #each website has its own TS learners,
    #which learn the 3 user class demand curves + 3 aggregated curves (only splitting one or no feature each)
        #in context generation, all bandits can learn with every incoming user
    #dimension -2: as each bandit learns actually the CTRs, it has one arm per slot
    #with each TS-learner having 2 parameters for each beta distribution
    #order: 0= 0 features split (task 3), 1=first user class, 2=2nd user class, 3=3rd user class,
    # 4=first feature split=1st and 3rd user class, 5=2nd feature split= 1st and 2nd user class
    #6= 0 features split (task 4), 7 =aggregated for task 7
beta_params=np.ones((nr_websites,8,nr_slots,2))
beta_params_advertiser=np.ones((nr_websites,2,2)) #only one TS bandit per website, learning the average CTR; (params[i,1,:]) for task 7
split_feature=np.zeros((nr_websites)) #split_feature[i]==0: no feature split, ..==1: first feature split, ..==2: 2nd feature split, ..==3: both features split

def update_bandits(i,bandit,greek,slot,user_class=None): #i and bandit as in sample(), greek {0,1} determines whether alpha (0) or beta (1) of is updated, if context is used then the user class must be defined
    if bandit==0 or bandit==7:
        beta_params[i, bandit, slot, greek] += 1
    else: #for task4 with context: one user input is used to learn as much as possible:
        beta_params[i, 6, greek] += 1  # the bandit of the aggregated curve always learns - only difference to bandit of task3 is the ad-slot matching, according to split_feature
                                        # (and this only changes when the context-learners are trained enough to generate more profits and the aggregated one is unlikely to be used again)
        if user_class == 0:
            beta_params[i, 1, greek] += 1
            beta_params[i, 4, greek] += 1
            beta_params[i, 5, greek] += 1
        elif user_class == 1:
            beta_params[i, 2, greek] += 1
            beta_params[i, 5, greek] += 1
        elif user_class == 2:
            beta_params[i, 3, greek] += 1
            beta_params[i, 4, greek] += 1

    #rewards-order acc. to bandit-param order: 0=basic TS (task3),
    # for task 4: 1=1.class bandit,2=2.class, 3=3.class, 4=1.&3.class, 5=1.&3. class, 6=aggregated classes
        #for context generation, use click probabilites as rewards
    # for task 7: 7=aggregated classes, 8=from our advertisers learners (order changed bc it must comply with the respective bandit's order)
    # for task 6: 9=from our advertisers learners
    #additionally clairvoyant rewards: 10= for task 3, 11= for task 4, 12= for task 6 (for task 7 reuse rewards 10 &12)
#rewards=np.zeros((nr_websites,13)) - do not use like global: python scopes dont work but suck!

daily_regret3=[] #for task 3
daily_regret4=[] #for task 4
daily_regret6=[] #for task 6
daily_regret70=[] #for task 7, index 0 for our advertiser, index 1 for the publishers
daily_regret71=[] #for task 7, index 0 for our advertiser, index 1 for the publishers

cumulative_regret3=[] #for task 3
cumulative_regret4=[] #for task 4
cumulative_regret6=[] #for task 6
cumulative_regret70=[] #for task 7, index 0 for our advertiser, index 1 for the publishers
cumulative_regret71=[] #for task 7, index 0 for our advertiser, index 1 for the publishers

Do_average_regrets = False
    # Determine, whether the regrets of the individual sites should be plotted or not,
    # this would show if learning rates differ (possible if some click probability is extremely low for example)
Do_cumulate_regrets = True #for debugging, plot daily regrets instead of cumulative regrets


def get_estimated_CTRs(website,bandit, user_class=None): #returns the TS-estimated click-through rates
    if user_class==None:
        ctr = Qij_aggregated[website, :, :]
    else:
        ctr = Qij[website,user_class,:,:]

    ctr[0, :] = np.random.beta(beta_params[website, bandit,:, 0], beta_params[website, bandit,:, 1])

    return ctr

def hungarian_matcher(weights):
    """minimising_weights=np.divide(1,np.where(weights>0,weights,1e-30)) #seems like a not ideal matcher here...
    rowind,colind = linear_sum_assignment(minimising_weights) #the index of the ad, in the order of the slots, e.g. in idx=[2,5,1,3,7], ad 2 is chosen for slot 1, ad 5 for slot 2 etc.
    lin_sum_indices=rowind[colind.sort()][0]"""
    max_indices=np.argsort(weights[:,0])[-nr_slots:][::-1]
    """if not all(max_indices==lin_sum_indices): #used to compare different matchers
        print("My ind != ind")
        sum=[0,0]
        for n in range(nr_slots):
            sum[0]+=weights[lin_sum_indices[n],n]
            sum[1]+=weights[max_indices[n],n]
        print("my ind is higher by:" +str(sum[1]-sum[0]))"""
    return max_indices


"""def sample(i, user_class, bandit, bids, no_context=False): #i=nr of website, user_class=nr of user class acc. to Q definition {0,1,2}, bandit=nr of bandit acc to rewards order
    #matcher = Hungarian_Matcher(get_estimated_CTRs(i, bandit, user_class))
    #selected_ad = np.array(np.argwhere(matcher.hungarian_algo()[0] > 0)[:, 1])
    if no_context:
        weights=np.multiply(get_estimated_CTRs(i, bandit), bids[i][None].T)
    else:
        weights=np.multiply(get_estimated_CTRs(i, bandit, user_class), bids[i][None].T)
    _, selected_ad = hungarian_matcher(weights) #passing the allocated values for each ad in each slot to the matcher
    if len(np.nonzero(selected_ad==0)): #speed up the algorithm (if ad 0 not displayed, no learning)
        for s in range(nr_slots): # getting a user input
            if not user_is_looking_at(i,s):
                return 0
            if np.random.binomial(1, Q[i, user_class, selected_ad[s]]) > 0:  # if clicked - according to Qij probabilites (incorporated via the loop and the user_is looking fct)
                if selected_ad[s] == 0:  # if our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                    update_bandits(i,bandit,0, user_class)
                    return 1 #reward=1, as the TS example reward was one as well
                else:
                    return 0  # at max one ad clicked on a website per auction
            elif selected_ad[s] == 0:  # our advertiser was shown, but not clicked: update beta-distribution (beta+=1)
                update_bandits(i,bandit,1, user_class)
                return 0 #no more learning in this auction
    return 0 #no ad was clicked at all"""

    #embedding the updating of TS learners in the right context:
def sample(i, user_class, bandit, bids, no_context=False): #i=nr of website, user_class=nr of user class acc. to Q definition {0,1,2}, bandit=nr of bandit acc to rewards order
    #matcher = Hungarian_Matcher(get_estimated_CTRs(i, bandit, user_class))
    #selected_ad = np.array(np.argwhere(matcher.hungarian_algo()[0] > 0)[:, 1])
    if no_context:
        weights=np.multiply(get_estimated_CTRs(i, bandit), bids[i][None].T)
    else:
        weights=np.multiply(get_estimated_CTRs(i, bandit, user_class), bids[i][None].T)
    selected_ad = hungarian_matcher(weights) #passing the allocated values for each ad in each slot to the matcher
    expected_reward=0
    for k in range(nr_slots):
        if no_context:
            """#ideal=weights_t3_first[selected_ad[k],k] #for some reason, all these ideal and real values match, but in the end the expected reward of the TS learner is always bigger than the "ideal"...
            #ideal=bids_task3[i][selected_ad[k]]
            ideal=Qij_aggregated[i,selected_ad[k],k]
            #real=np.multiply(Qij_aggregated[i,selected_ad[k],k], bids[i][selected_ad[k]])
            #real=bids[i][selected_ad[k]]
            real=Qij_aggregated[i,selected_ad[k],k]
            print("mmmmmmmmmmmmmmm             " +str(ideal-real))"""
            expected_reward+=np.multiply(Qij_aggregated[i,selected_ad[k],k], bids[i][selected_ad[k]]) #finding the expected reward of this allocation
            expected_reward-=np.multiply(Qij_aggregated[i,selected_ad_clairvoyant3[i][k],k], bids[i][selected_ad_clairvoyant3[i][k]]) #subtracting expected reward of ideal allocation
        else:
            expected_reward+=np.multiply(Qij[i,user_class,selected_ad[k],k], bids[i][selected_ad[k]]) #finding the expected reward of this allocation
    if len(np.nonzero(selected_ad==0)[0])>0: #speed up the algorithm (if ad 0 not displayed, no learning)
        s = np.where(selected_ad==0)[0] # getting a user input - directly looking at our ad: samping in for-loop more realistic, but less efficient and irrelevant, as other samples arent kept anyway (plus, like so the user distribution equals exactly our sample size/increased sample size compared to for-loop, as only the sampels are considered which make it to our ad)
        if not user_is_looking_at(i,s):
            # our advertiser was shown, but not clicked: update beta-distribution (beta+=1)
            update_bandits(i, bandit, 1,s, user_class) #updating here to incorporate prominence rate and find the CTR for the bandit
            return 0, expected_reward
        if np.random.binomial(1, Q[i, user_class, selected_ad[s]]) > 0:  # if clicked - according to Q probabilites
                # our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                update_bandits(i,bandit,0,s, user_class)
                return 1,expected_reward #advertiser_reward+=1, as the TS example reward was one as well
        else:  # our advertiser was shown, but not clicked: update beta-distribution (beta+=1)
            update_bandits(i,bandit,1,s, user_class)
            return 0,expected_reward #no more learning in this auction
    return 0,expected_reward #our ad was not shown

def beta_expectation(alpha,beta):
    return np.divide(alpha,np.add(alpha,beta))

def lower_bound(mean, dataset, confidence=0.95):
    datasetnp=np.array(dataset)
    datasetnp[datasetnp==0]=1
    return np.subtract(mean,np.sqrt(np.divide(-np.log(confidence)/2,datasetnp)))

bids_task3 = get_stochastic_bids(daily_users_mean)
# using the clairvoyant algorithm for task 3
expected_reward_clairvoyant3=np.zeros((nr_websites))
# cv3_matcher = Hungarian_Matcher(np.multiply(np.average(Q[i, :, :],0),bids[i])) #averaging class' click probabilities to get clairvoyant aggregated probability
# selected_ad_clairvoyant3 = np.array(np.argwhere(cv3_matcher.hungarian_algo()[0]>0)[:,1])
selected_ad_clairvoyant3=[]
for i in range(nr_websites):
    weights_task3 = np.multiply(Qij_aggregated[i, :, :], bids_task3[i][None].T)
    selected_ad_clairvoyant3.append(hungarian_matcher(weights_task3))  # passing clairvoyant allocated values for aggregated demands
    if i==0:
        weights_t3_first=weights_task3
    for k in range(nr_slots):
        expected_reward_clairvoyant3[i] += weights_task3[selected_ad_clairvoyant3[i][k],k]  # finding the expected reward of ideal allocation

    #VCG-Auction utilities:
def optimal_SW_without_our_ad(a, prominence, quality, bids):
    X = 0
    quality2 = quality
    bids2 = bids
    for i in range(a, nr_ads - 1):
        quality2[i] = quality2[i + 1]
        bids2[i] = bids2[i + 1]

    # allocation of slots without a
    #matcher = Hungarian_Matcher(np.multiply(quality[1:], nu_bids[1:]))  # ad_quality*bid to maximize expected value
    #selected_ads = np.array(np.argwhere(matcher.hungarian_algo()[0] > 0)[:, 1])
    selected_ads = hungarian_matcher(np.multiply(quality[1:], nu_bids[1:]))
    for i in range(nr_slots):
        a_ = np.where(selected_ads == 0)[0]
        X += prominence[i] * quality2[a_] * bids2[a_]
    return X

def dailyloop(day):
    rewards = np.zeros((nr_websites, 13)) # reset rewards each round - for the order see the definition in initialisation part
    if day%7 == 0 and day!=0: #every 7 days the contexts are defined (not at day=0)
        if DoPrint:
            string="Week %d passed" % (day/7)
            #str.append(", total of %d users" % (nr_total_users[0]))
            #string+=(", budget of %d remains" % (budget[0]))
            #string+=(", budget remaining" + str(budget))

            ##Publishers TS-learners
            a=beta_expectation(beta_params[:, 0, 0], beta_params[:, 0, 1])
            b=np.average(np.multiply(Q[:,:,0],class_probabilities),axis=1)
            string+=("\nexpected mean: " + str(a) + "\nreal mean    : " +str(b) +"\nDifference   : "+str(abs(b-a))+"\n")
            #string+=(", samples per bandit(pubs), pos: " + str(beta_params[:, 0, 0]-1)+ "neg:" + str(beta_params[:,0,1]-1))
            string+=(", samples per bandit(pubs): " + str(beta_params[:, 0, 0]+beta_params[:,0,1]-2))
                # bandits receive 0.1-6% samples from users (almost 2 magnitudes!! - the seldomly used bandits will over time be less and less frequent, so just consider the higher performing bandits)
            #string+=(", positive/negative sample-ratio per bandit (pub): " + str((beta_params[:, 0, 0]-1)/(beta_params[:, 0, 1]-1)))
                # around 1/5 up to 4/1 (reciprocal!!!) - again shows how the advertisers Bandit samples only the rewarding arms (chooses rather the website with higher CTR)

            ##Advertiser TS-learners
            #string+=(", samples per bandit (adv): " + str(beta_params_advertiser[:, :, 0]+beta_params_advertiser[:,:,1]-2))
                #looks correct for beta_params_advertiser: task 6 receives 3-25% samples per user, task 7 receives 1.5-10% (so the natural range is ~8 or one magnitude)
            #string+=(", positive/negative sample-ratio per bandit (adv): " + str((beta_params_advertiser[:, :, 0]-1)/(beta_params_advertiser[:, :, 1]-1)))
                #around 1/5 up to 4/1 (reciprocal!!!) - again shows how the advertisers Bandit samples only the rewarding arms (chooses rather the website with higher CTR)
            print(string)
        for i in range(nr_websites):
            # expected mean (reward) of a campaign bandit by averaging the expectation values of each of its slot
            u_mean = beta_expectation(beta_params[i,:6,0],beta_params[i,:6,1]) #bandit indices stop at 6: dont go further than task 4
            u_lowerbound = lower_bound(u_mean,nr_total_users[0])
            p_mean_occurrence_class=np.divide(nr_total_users_per_class[i,:],nr_total_users[0])
            p_lowerbound_occurrence_class = lower_bound(p_mean_occurrence_class,nr_total_users_per_class[i,:])

            # theoretically, only one feature split at a time, but practically we seek the highest expected reward
            split_feature[i] = np.argmax([u_lowerbound[0], #no feature split = base case
                                   (p_lowerbound_occurrence_class[0]+p_lowerbound_occurrence_class[2])*u_lowerbound[4]+p_lowerbound_occurrence_class[1]*u_lowerbound[2], #first feature split
                                   (p_lowerbound_occurrence_class[0]+p_lowerbound_occurrence_class[1])*u_lowerbound[5]+p_lowerbound_occurrence_class[2]*u_lowerbound[3], #second feature split
                                   p_lowerbound_occurrence_class[0]*u_lowerbound[1]+p_lowerbound_occurrence_class[1]*u_lowerbound[2]+p_lowerbound_occurrence_class[2]*u_lowerbound[3]]) #both features split

    nr_daily_users = get_daily_users()
    nr_total_users[0]+=nr_daily_users

    #sampling the user classes of each website's visitors
    user_classes = get_user_classes(nr_daily_users)
    for i in range(nr_websites):
        found_class , class_count =np.unique(user_classes[i],return_counts=True)
        for c in range(len(class_count)):
            nr_total_users_per_class[i,found_class]+=class_count[c]

    #receiving the stochastic advertiser's bids
    bids= get_stochastic_bids(nr_daily_users)

    ###using knapsack to determine our advertisers bid - task 6
    our_bids=get_bids()
    our_bids_task7=get_bids(bandit=1) #using our second bandit to train alongside the publisher's bandits
    our_clairvoyant_bids=get_bids(clairvoyant=True)


    for auction in range(nr_daily_users): #evaluating each auction individually

        ###Matching of ads and slots for each publisher, using Thompson-Sampling-Bandits to find our advertisers click probabilities
        for i in range(nr_websites):
            # using the TS-sample for our advertiseers click probability in task3
            rewards[i, 0]+=sample(i,user_classes[i][auction],0,bids_task3,True)[1]

        """
            #using different TS-Bandits to train for different contexts in task 4:
            if split_feature[i] == 0:
                # context of unsplitted features - sampling parameters independent of user_class at hand
                rewards[i, 6]+=sample(i, user_classes[i][auction], 6, bids)
            elif user_classes[i][auction]==0:
                if split_feature[i] == 1:
                    #context of split feature 1
                    rewards[i, 4]+=sample(i, 0, 4, bids)
                elif split_feature[i] == 2:
                    #context of split feature 2
                    rewards[i, 5]+=sample(i, 0, 5, bids)
                elif split_feature[i] == 3:
                    #context of split feature 1 & 2
                    rewards[i, 1]+=sample(i, 0, 1, bids)
            elif user_classes[i][auction]==1:
                if split_feature[i] == 1:
                    #context of split feature 1
                    rewards[i, 2]+=sample(i, 1, 2, bids)
                elif split_feature[i] == 2:
                    #context of split feature 2
                    rewards[i, 5]+=sample(i, 1, 5, bids)
                elif split_feature[i] == 3:
                    #context of split feature 1 & 2
                    rewards[i, 2]+=sample(i, 1, 2, bids)
            elif user_classes[i][auction]==2:
                if split_feature[i] == 1:
                    #context of split feature 1
                    rewards[i, 4]+=sample(i, 2, 4, bids)
                elif split_feature[i] == 2:
                    #context of split feature 2
                    rewards[i, 3]+=sample(i, 2, 3, bids)
                elif split_feature[i] == 3:
                    #context of split feature 1 & 2
                    rewards[i, 3]+=sample(i, 2, 3, bids)

            # using the clairvoyant algorithm for task 4
            #cv4_matcher = Hungarian_Matcher(np.multiply(Q[i, user_classes[i][auction], :], bids[i]))
            #selected_ad_clairvoyant4 = np.array(np.argwhere(cv4_matcher.hungarian_algo()[0] > 0)[:, 1])
            _, selected_ad_clairvoyant4 = hungarian_matcher(np.multiply(Qij[i,user_classes[i][auction],:,:],bids[i][None].T)) #passing clairvoyant allocated values for disaggregated demands
            if len(np.nonzero(selected_ad_clairvoyant3 == 0)):  # speed up the algorithm (if ad 0 not displayed, no learning)
                for s in range(nr_slots):
                    if not user_is_looking_at(i,s):
                        break
                    if np.random.binomial(1, Q[i, user_classes[i][auction], selected_ad_clairvoyant4[s]]) > 0:
                        rewards[i,11]+=1
                        #rewards[i,11]+=Qij[i, user_classes[i][auction], selected_ad_clairvoyant4[s],s]*bids[i][selected_ad_clairvoyant4[s]]


            ###VCG-auctions for our advertisers payments

            #determining the expected payment X-Y:
                #the publishers choose their ideal allocations
                # (acc. to lecture "pay per click": just sort ad_value*ad_quality by highest first - but we choose hungarian over this greedy approach, to find an optimum without allocating the same ad several times) - task 6&7
            nu_bids=np.append(our_bids[i],bids[i][1:]) #keep stochastic bids, ad our_bid at index 0
                # using the clairvoyant matcher for task 6 (aggregated since using "all data together")
                #best slot allocation with our ad:
            weights_with_our_ad = np.multiply(Qij_aggregated[i, :, :], nu_bids[None].T)
            _, ad_allocation_with_our_ad = hungarian_matcher(weights_with_our_ad)
            #cv6_matcher = Hungarian_Matcher(np.multiply(Q[i,user_classes[i][auction],:],nu_bids)) #ad_quality*bid to maximize expected value
            #selected_ad_clairvoyant6 = np.array(np.argwhere(cv6_matcher.hungarian_algo()[0] > 0)[:, 1])
            #sampling (the users get to click) - task 6&7
            if len(np.nonzero(ad_allocation_with_our_ad==0)): #speed up the algorithm (if ad 0 not displayed, no learning, no payment)
                for s in range(nr_slots):
                    if not user_is_looking_at(i,s):
                        break
                    if np.random.binomial(1, Q[i, user_classes[i][auction], ad_allocation_with_our_ad[s]]) > 0: # Bernoulli - clicked
                        if ad_allocation_with_our_ad[s]==0:  # if our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                            rewards[i,9]+=1
                            #rewards[i,9]+=Qij[i,user_classes[i][auction],0,s]*nu_bids[0]
                            beta_params_advertiser[i,0, 0] += 1
                        elif any([ad_allocation_with_our_ad[k] == 0 for k in range(s + 1)]):
                            beta_params_advertiser[i,0, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)
                        break  # at max one ad clicked on a website per auction
                    elif s == nr_slots - 1:  # if no ad was clicked at all
                        beta_params_advertiser[i,0, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)

                # Y-variable (accumulated welfare without our advertiser's included):
                Y = 0
                for s in range(nr_slots):
                    Y += weights_with_our_ad[s, ad_allocation_with_our_ad[s]]

                    # X-Variable:
                X = 0
                weights_without_our_ad = np.multiply(Qij[i, user_classes[i][auction], 1:, :],nu_bids[1:][None].T)
                _, ad_allocation_without_our_ad = hungarian_matcher(weights_without_our_ad)
                for s in range(nr_slots - 1):
                    X += weights_without_our_ad[s, ad_allocation_without_our_ad[s]]

                    # price for our advertiser:
                budget[0] -= (X - Y) / Qij_aggregated[i, 0, np.where(ad_allocation_with_our_ad == 0)[0]]

            #For the clairvoyant advertiser:
            nu_ideal_bids=np.append(our_clairvoyant_bids[i],bids[i][1:]) #keep stochastic bids, ad our_bid at index 0
            #cv6_ideal_bid_matcher = Hungarian_Matcher(np.multiply(Q[i, user_classes[i][auction], :], nu_ideal_bids))  # ad_quality*bid to maximize expected value
            #selected_ad_cv6_ideal_bid = np.array(np.argwhere(cv6_ideal_bid_matcher.hungarian_algo()[0] > 0)[:, 1])
            weights_with_our_ad_ideal_bid = np.multiply(Qij_aggregated[i, :, :], nu_ideal_bids[None].T)
            _, selected_ad_cv6_ideal_bid = hungarian_matcher(weights_with_our_ad_ideal_bid)
            if len(np.nonzero(selected_ad_cv6_ideal_bid == 0)):  # speed up the algorithm (if ad 0 not displayed, no learning,no payments)
                # sampling (the users get to click) - task 6&7
                for s in range(nr_slots):
                    if not user_is_looking_at(i,s):
                        break
                    if np.random.binomial(1, Q[i, user_classes[i][auction], selected_ad_cv6_ideal_bid[s]])  > 0:  # Bernoulli - clicked
                        if selected_ad_cv6_ideal_bid[s] == 0:  # if our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                            rewards[i,12]+=1
                            #rewards[i,12]+=Qij[i,user_classes[i][auction],0,s]*nu_ideal_bids[0]
                            beta_params_advertiser[i,0, 0] += 1
                        elif any([selected_ad_cv6_ideal_bid[k] == 0 for k in range(s + 1)]):
                            beta_params_advertiser[i,0, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)
                        break  # at max one ad clicked on a website per auction
                    elif s == nr_slots - 1:  # if no ad was clicked at all
                        beta_params_advertiser[i,0, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)

                    # Y-variable (accumulated welfare without our advertiser's included):
                    Y = 0
                    for s in range(nr_slots):
                        Y += weights_with_our_ad_ideal_bid[s, selected_ad_cv6_ideal_bid[s]]

                        # X-Variable:
                    X = 0
                    weights_without_our_ad_ideal_bid = np.multiply(Qij[i, user_classes[i][auction], 1:, :],nu_bids[1:][None].T)
                    _, ad_allocation_without_our_ad_ideal_bid = hungarian_matcher(weights_without_our_ad_ideal_bid)
                    for s in range(nr_slots - 1):
                        X += weights_without_our_ad_ideal_bid[s, ad_allocation_without_our_ad_ideal_bid[s]]

                        # price for our advertiser:
                    budget[1] -= (X - Y) / Qij_aggregated[i, 0, np.where(selected_ad_cv6_ideal_bid == 0)[0]]

            #Task7: simultaneous learning:
            nu_bids_simultaneous = np.append(our_bids_task7[i], bids[i][1:])
            weights_with_our_ad_simultaneous = np.multiply(get_estimated_CTRs(i,7), nu_bids_simultaneous[None].T)
            _, selected_ad_simultaneous = hungarian_matcher(weights_with_our_ad_simultaneous)
            if len(np.nonzero(selected_ad_simultaneous == 0)):  # speed up the algorithm (if ad 0 not displayed, no learning,no payments)
                for s in range(nr_slots):
                    if not user_is_looking_at(i,s):
                        break
                    if np.random.binomial(1, Q[i, user_classes[i][auction], selected_ad_simultaneous[s]])  > 0:  # Bernoulli - clicked
                        if selected_ad_simultaneous[s] == 0:  # if our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                            rewards[i,7]+=1
                            #rewards[i,7]+=Qij[i,user_classes[i][auction],0,s]*nu_bids_simultaneous[0]
                            beta_params_advertiser[i,1, 0] += 1
                        elif any([selected_ad_simultaneous[k] == 0 for k in range(s + 1)]):
                            beta_params_advertiser[i,1, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)
                        break  # at max one ad clicked on a website per auction
                    elif s == nr_slots - 1:  # if no ad was clicked at all
                        beta_params_advertiser[i,1, 1] += 1  # in case our advertiser was shown, but not clicked, update beta-distribution (beta+=1)

                    # Y-variable (accumulated welfare without our advertiser's included):
                    Y = 0
                    for s in range(nr_slots):
                        Y += weights_with_our_ad_simultaneous[s, selected_ad_simultaneous[s]]

                        # X-Variable:
                    X = 0
                    weights_without_our_ad_simultaneous = np.multiply(Qij[i, user_classes[i][auction], 1:, :],nu_bids[1:][None].T)
                    _, ad_allocation_without_our_ad_simultaneous = hungarian_matcher(weights_without_our_ad_simultaneous)
                    for s in range(nr_slots - 1):
                        X += weights_without_our_ad_simultaneous[s, ad_allocation_without_our_ad_simultaneous[s]]

                        # price for our advertiser:
                    budget[2] -= (X - Y) / Qij_aggregated[i, 0, np.where(selected_ad_simultaneous == 0)[0]]
    """
    #saving the cumulative regrets - theoretically, in one or another auction the clairvoyant reward is not the highest, as it is a stochastic value.
    # But on average (over the day for example) the clairvoyant choice gets definitely the highest reward possible, so the cumulative regret never falls <0.
    # This is true because sample() in each round is called with the same user class, only varying the matching of ads to slots (thus increasing the click probabilities in clairvoyant case)
    """print(expected_reward_clairvoyant3*nr_daily_users)
    print(rewards[:,0])
    print(expected_reward_clairvoyant3*nr_daily_users-rewards[:,0])"""
    daily_regret3.append(rewards[:,0]) #for task 3, the reward variable=the regret
    cr4=np.zeros((4))
    for i in range(nr_websites):
        if split_feature[i] == 0:
            # context of unsplitted features
            cr4[i] += rewards[i,11]-rewards[i,6]
        elif split_feature[i] == 1:
            # context of split feature 1
            cr4[i] += rewards[i,11]-rewards[i,2]-rewards[i,4]
        elif split_feature[i] == 2:
            # context of split feature 2
            cr4[i] += rewards[i,11]-rewards[i,3]-rewards[i,5]
        elif split_feature[i] == 3:
            # context of split feature 1 & 2
            cr4[i] += rewards[i,11]-rewards[i,1]-rewards[i,2]-rewards[i,3]
    daily_regret4.append(cr4)
    daily_regret6.append(rewards[:,12]-rewards[:,9])
    daily_regret70.append(rewards[:,10]-rewards[:,7])
    daily_regret71.append(rewards[:,12]-rewards[:,8])

    #assert all(daily_regret3[-1]>=0)
    #print("our ads rewards: " +str(rewards[:,0]))
    #print("our ads regrets: " +str(daily_regret3[-1]))

def normalize_regrets():
    for i in range(nr_websites):
        mx_val3=max(daily_regret3, key=lambda element:element[i])[i]
        mx_val4=max(daily_regret4, key=lambda element:element[i])[i]
        mx_val6=max(daily_regret6, key=lambda element:element[i])[i]
        mx_val70=max(daily_regret70, key=lambda element:element[i])[i]
        mx_val71=max(daily_regret71, key=lambda element:element[i])[i]
        for day in range(T):
            daily_regret3[day][i] /= mx_val3
            """daily_regret4[day][i] /= mx_val4
            daily_regret6[day][i] /= mx_val6
            daily_regret70[day][i] /= mx_val70
            daily_regret71[day][i] /= mx_val71"""

def average_regrets():
    #summarizing/averaging over all websites the bandit learning, which now are comparable as percentages
    cumulative_regret3.extend(np.average(daily_regret3,-1))
    cumulative_regret4.extend(np.average(daily_regret4,-1))
    cumulative_regret6.extend(np.average(daily_regret6,-1))
    cumulative_regret70.extend(np.average(daily_regret70,-1))
    cumulative_regret71.extend(np.average(daily_regret71,-1))
    """DONE BY np.average inBUILT already:
    for day in range(T):
        cumulative_regret3.append(np.average(daily_regret3[day]))
        cumulative_regret4.append(np.average(daily_regret4[day]))
        cumulative_regret6.append(np.average(daily_regret6[day]))
        cumulative_regret70.append(np.average(daily_regret70[day]))
        cumulative_regret71.append(np.average(daily_regret71[day]))"""


###PLOTS
def plot_average(): #displaying the averaged regrets
    normalize_regrets()
    x = np.arange(1,T+1)
    average_regrets()

    if Do_cumulate_regrets:
        for i in range(1,T): #changing from daily to cumulative regrets
            cumulative_regret3[i]+=cumulative_regret3[i-1]
            cumulative_regret4[i]+=cumulative_regret4[i-1]
            cumulative_regret6[i]+=cumulative_regret6[i-1]
            cumulative_regret70[i]+=cumulative_regret70[i-1]
            cumulative_regret71[i]+=cumulative_regret71[i-1]

    plt.figure()
    a1= plt.subplot(511)
    a1.plot(x, cumulative_regret3)
    a1.set_title("cumulative regret task 3")
    a2=plt.subplot(512)
    a2.plot(x, cumulative_regret4)
    a2.set_title("cumulative regret task 4")
    a3=plt.subplot(513)
    a3.plot(x, cumulative_regret6)
    a3.set_title("cumulative regret task 6")
    a4=plt.subplot(514)
    a4.plot(x, cumulative_regret70)
    a4.set_title("cumulative regret of publishers task 7")
    a5=plt.subplot(515)
    a5.plot(x, cumulative_regret71)
    a5.set_title("cumulative regret of our advertiser task 7")
    plt.show()

def plot_individual(): #showing each websites regret individually
    normalize_regrets()
    x = np.arange(1,T+1)
    #cumulative_regret3=np.concatenate(np.atleast_3d(daily_regret3),axis=1) #alternative
    #cumulative_regret3=np.array(daily_regret3,ndmin=2) #alternative - must switch in plot fct the order of the axis! (commented ou
    cumulative_regret3=np.vstack(daily_regret3)
    if Do_cumulate_regrets:
        for i in range(1,T): #changing from daily to cumulative regrets
            cumulative_regret3[i,:]+=cumulative_regret3[i-1,:]
            """cumulative_regret4[i]+=cumulative_regret4[i-1]
            cumulative_regret6[i]+=cumulative_regret6[i-1]
            cumulative_regret70[i]+=cumulative_regret70[i-1]
            cumulative_regret71[i]+=cumulative_regret71[i-1]"""
        """
    #cumulative_regret3.extend([np.array(daily_regret3[d]) for d in range(T)])
    cumulative_regret4.extend([np.array([daily_regret4[d]]) for d in range(T)])
    cumulative_regret6.extend([np.array([daily_regret6[d]]) for d in range(T)])
    cumulative_regret70.extend([np.array([daily_regret70[d]]) for d in range(T)])
    cumulative_regret71.extend([np.array([daily_regret71[d]]) for d in range(T)])"""
    plt.figure(0)
    plt.suptitle("Task 3")
    """plt.plot(x, cumulative_regret3[0][:],
             x, cumulative_regret3[1][:],
             x, cumulative_regret3[2][:],
             x, cumulative_regret3[3][:])"""
    plt.plot(x, cumulative_regret3[:,0],label="website0")
    plt.plot(x, cumulative_regret3[:,1],label="website1")
    plt.plot(x, cumulative_regret3[:,2],label="website2")
    plt.plot(x, cumulative_regret3[:,3],label="website3")
    plt.legend()
    plt.xlabel("Time horizon (days)")
    if Do_cumulate_regrets:
        plt.ylabel("Cumulative normalized regret per website")
    else:
        plt.ylabel("Normalized regret per website")
    plt.matshow(np.average(np.multiply(Q,np.atleast_3d(class_probabilities)),axis=1))
    plt.show()


def RUN():
    for day in range(T):
        dailyloop(day)
    if Do_average_regrets:
        plot_average()
    else:
        plot_individual()

#cProfile.run(RUN())
RUN()