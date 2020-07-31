import numpy as np
from Hungarian_Matcher import *
from knapsack import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import cProfile

#GENERAL

    #Time horizon in days
T=120

    #number of websites
nr_websites=4

    #number of slots (ads) per website
nr_slots=4 #using the same amount of slots for all websites

    #number of advertizers per website
nr_ads=6 #using the same advertizers/number of ads on all websites - this number includes our advertiser

DoPrint = True #prints a weekly update to the console

#USER DISTRIBUTIONS
    #Daily users visiting a website (same for every website, as different dataset sizes would bias the result and hinder an direct comparison)
daily_users_mean=800
daily_users_deviation=100

nr_total_users=np.array([0])
nr_total_users_per_class=np.zeros((nr_websites,3)) #3 classes for each website

def get_daily_users():
    return max(int(np.random.normal(daily_users_mean,daily_users_deviation)+0.5),1)


#CLICKING PROBABILITIES
    #click probabilities=ad_quality_i on the different websites per user class (amount=3)
Q=np.random.normal(0.6,0.1,(nr_websites,nr_ads)) #again uniform distribution, assume each entry where index nr_ads=0 belongs to our advertiser

#truncate so we wont have too low values of Q and some ads (esp. from our advertiser) would not show at all
Q[Q<0.2]=0.2
Q[Q>0.9]=0.9
for o in range(nr_websites):
    if Q[o,0]<0.5:
        Q[o,0]=0.5
    print("Q's 2norm of displayed ads on website "+ str(o)+":"+str(np.linalg.norm(Q[o,np.argsort(Q[o,:])[-nr_slots:]],2)))
    print("Q's average of displayed ads on website "+ str(o)+":                          "+str(np.average(Q[o,np.argsort(Q[o,:])[-nr_slots:]])))
print("Q itself:\n"+str(Q))
    #lambda vector (prominence rates) - known beforehand for each advertiser (realistic, as they have experience from the past of their website's ads)
prominence_rates=np.random.normal(0.7,0.1,(nr_websites,nr_slots))

prom_matrix=[[prominence_rates[z,0]] for z in range(nr_websites)]
print("Prominance factors:")
for i in range(nr_websites):
    for s in range(1,nr_slots):
        prom_matrix[i].append(min(1,prominence_rates[i, s] / prominence_rates[i, s - 1]))
    print(prom_matrix[i])


def user_is_looking_at(i,s): # check if user looks at ad in slot s at  all with given prominence rates
    if s == 0:
        if np.random.binomial(1, prominence_rates[i, 0]) == 0:  # Bernoulli - user lost interest
            return False
    elif np.random.binomial(1,min(1,prominence_rates[i, s] / prominence_rates[i, s - 1])) == 0:  # Bernoulli - user lost interest
        return False
    return True

    #get Qij: holding q_i,j variables as described in the problem statement, for each website and user class
def get_cv_CTRs(website):
    ctr = np.zeros([nr_slots, nr_ads])
    ctr[0, :] = np.multiply(Q[website, :], prominence_rates[website, 0])
    for i in range(nr_slots - 1):
        ctr[i + 1, :] = np.multiply(ctr[i,:], min(1,prominence_rates[website, i + 1] / prominence_rates[website, i])) #maxing function to ceil the the prominence_rates: a user can not more than definitely observe a certain ad; it does not imply directly a click
    return ctr.T #returns the true click through rates by multiplying the ad qualities with the prominence rates

Qij= np.zeros([nr_websites, nr_ads, nr_slots]) #q_i,j variables as described in the problem statement, for each website and user class
for i in range(nr_websites):
    Qij[i,:,:]=get_cv_CTRs(i)

for i in range(nr_websites):
    print("Qij's 2norm of displayed ads on website "+ str(i)+":"+str(np.linalg.norm(Qij[i,np.argsort(Qij[i,:,0])[-nr_slots:],0],2)))
    print("Qij's average of displayed ads on website "+ str(i)+":                       "+str(np.average(Qij[i,np.argsort(Qij[i,:,0])[-nr_slots:],0])))

#ADVERTISERS
    #define stochastic advertisers
    #on average they should bid as much as our advertiser (to allow every advertiser to win some auctions)
    #each advertiser bids only once per day, a daily constant bid per subcampaign
def get_stochastic_bids(nr_auctions):
    return [np.random.normal(0.6, 0.1, nr_ads) for k in range(nr_websites)]

    #Buddget of our advertizer
budget=np.array([1000])
campaign_budget=budget/nr_websites
campaign_daily_budget=campaign_budget/T
daily_budget = budget / T


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

cumulative_regret3=[] #for task 3

plot_average_regrets = True
plot_individual_regrets=True
    # Determine, whether the regrets of the individual sites should be plotted or not,
    # this would show if learning rates differ (possible if some click probability is extremely low for example)
Do_cumulate_regrets = False #for debugging, plot daily regrets instead of cumulative regrets


def get_estimated_CTRs(website,bandit, user_class=None): #returns the TS-estimated click-through rates
    ctr = Qij[website,:,:]

    ctr[0, :] = np.random.beta(beta_params[website, bandit,:, 0], beta_params[website, bandit,:, 1])

    return ctr

def hungarian_matcher(weights):
    """minimising_weights=np.divide(1,np.where(weights>0,weights,1e-30)) #seems like a not ideal matcher here...
    rowind,colind = linear_sum_assignment(minimising_weights) #the index of the ad, in the order of the slots, e.g. in idx=[2,5,1,3,7], ad 2 is chosen for slot 1, ad 5 for slot 2 etc.
    lin_sum_indices=rowind[colind.sort()][0]"""
    max_indices=np.argsort(weights[:,0])[-nr_slots:][::-1]
    return max_indices

    #embedding the updating of TS learners in the right context:
def sample(i, bandit, bids, no_context=False): #i=nr of website, user_class=nr of user class acc. to Q definition {0,1,2}, bandit=nr of bandit acc to rewards order
    weights=np.multiply(get_estimated_CTRs(i, bandit), bids[i][None].T)
    selected_ad = hungarian_matcher(weights) #passing the allocated values for each ad in each slot to the matcher
    expected_reward=0
    for k in range(nr_slots):
        #expected_reward+=weights_task3[i][selected_ad[k],k]
        #expected_reward-=weights_task3[i][selected_ad_clairvoyant3[i][k],k]
        expected_reward+=np.multiply(Qij[i,selected_ad[k],k], bids[i][selected_ad[k]]) #finding the expected reward of this allocation
        expected_reward-=np.multiply(Qij[i,selected_ad_clairvoyant3[i][k],k], bids[i][selected_ad_clairvoyant3[i][k]]) #subtracting expected reward of ideal allocation
    if len(np.nonzero(selected_ad==0)[0])>0: #speed up the algorithm (if ad 0 not displayed, no learning)
        s = np.where(selected_ad==0)[0] # getting a user input - directly looking at our ad: samping in for-loop more realistic, but less efficient and irrelevant, as other samples arent kept anyway (plus, like so the user distribution equals exactly our sample size/increased sample size compared to for-loop, as only the sampels are considered which make it to our ad)
        if not user_is_looking_at(i,s):
            # our advertiser was shown, but not clicked: update beta-distribution (beta+=1)
            update_bandits(i, bandit, 1,s) #updating here to incorporate prominence rate and find the CTR for the bandit
            return 0, expected_reward
        if np.random.binomial(1, Q[i, selected_ad[s]]) > 0:  # if clicked - according to Q probabilites
                # our advertisers ad was displayed in the slot: update beta-distribution (since clicked: alpha+=1)
                update_bandits(i,bandit,0,s)
                return 1,expected_reward #advertiser_reward+=1, as the TS example reward was one as well
        else:  # our advertiser was shown, but not clicked: update beta-distribution (beta+=1)
            update_bandits(i,bandit,1,s)
            return 0,expected_reward #no more learning in this auction
    return 0,expected_reward #our ad was not shown

bids_task3 = get_stochastic_bids(daily_users_mean)
# using the clairvoyant algorithm for task 3
expected_reward_clairvoyant3=np.zeros((nr_websites))
weights_task3=[]
# cv3_matcher = Hungarian_Matcher(np.multiply(np.average(Q[i, :, :],0),bids[i])) #averaging class' click probabilities to get clairvoyant aggregated probability
# selected_ad_clairvoyant3 = np.array(np.argwhere(cv3_matcher.hungarian_algo()[0]>0)[:,1])
selected_ad_clairvoyant3=[]
for i in range(nr_websites):
    weights_task3.append(np.multiply(Qij[i, :, :], bids_task3[i][None].T))
    selected_ad_clairvoyant3.append(hungarian_matcher(weights_task3[i]))  # passing clairvoyant allocated values for aggregated demands
    for k in range(nr_slots):
        expected_reward_clairvoyant3[i] += weights_task3[i][selected_ad_clairvoyant3[i][k],k]  # finding the expected reward of ideal allocation
    print("Weights' 2norm of displayed ads on website "+ str(i)+":"+str(np.linalg.norm(weights_task3[i][np.argsort(weights_task3[i][:,0])[-nr_slots:],0],2)))
    print("Weights' average of displayed ads on website "+ str(i)+":                       "+str(np.average(weights_task3[i][np.argsort(weights_task3[i][:,0])[-nr_slots:],0])))

def dailyloop(day):
    rewards = np.zeros((nr_websites, 13)) # reset rewards each round - for the order see the definition in initialisation part
    if day%7 == 0 and day!=0: #every 7 days the contexts are defined (not at day=0)
        if DoPrint:
            string="Week %d passed" % (day/7)
            #str.append(", total of %d users" % (nr_total_users[0]))
            #string+=(", budget of %d remains" % (budget[0]))
            #string+=(", budget remaining" + str(budget))

            ##Publishers TS-learners
            string+=(", samples per bandit:\n" + str(beta_params[:, 0, 0]+beta_params[:,0,1]-2))
                # bandits receive 0.1-6% samples from users (almost 2 magnitudes!! - the seldomly used bandits will over time be less and less frequent, so just consider the higher performing bandits)
            #string+=(", positive/negative sample-ratio per bandit (pub): " + str((beta_params[:, 0, 0]-1)/(beta_params[:, 0, 1]-1)))
                # around 1/5 up to 4/1 (reciprocal!!!) - again shows how the advertisers Bandit samples only the rewarding arms (chooses rather the website with higher CTR)

            ##Advertiser TS-learners
            #string+=(", samples per bandit (adv): " + str(beta_params_advertiser[:, :, 0]+beta_params_advertiser[:,:,1]-2))
                #looks correct for beta_params_advertiser: task 6 receives 3-25% samples per user, task 7 receives 1.5-10% (so the natural range is ~8 or one magnitude)
            #string+=(", positive/negative sample-ratio per bandit (adv): " + str((beta_params_advertiser[:, :, 0]-1)/(beta_params_advertiser[:, :, 1]-1)))
                #around 1/5 up to 4/1 (reciprocal!!!) - again shows how the advertisers Bandit samples only the rewarding arms (chooses rather the website with higher CTR)
            print(string)

    nr_daily_users = get_daily_users()
    nr_total_users[0]+=nr_daily_users

    for auction in range(nr_daily_users): #evaluating each auction individually

        ###Matching of ads and slots for each publisher, using Thompson-Sampling-Bandits to find our advertisers click probabilities
        for i in range(nr_websites):
            # using the TS-sample for our advertiseers click probability in task3
            rewards[i, 0]+=sample(i,0,bids_task3,True)[1]

    daily_regret3.append(rewards[:,0]) #for task 3, the reward variable=the regret
    #assert all(daily_regret3[-1]>=0)

def normalize_regrets():
    for i in range(nr_websites):
        mx_val3=max(daily_regret3, key=lambda element:element[i])[i]
        for day in range(T):
            daily_regret3[day][i] /= mx_val3

def average_regrets():
    #summarizing/averaging over all websites the bandit learning, which now are comparable as percentages
    cumulative_regret3.extend(np.average(daily_regret3,-1))


###PLOTS
def plot_average(): #displaying the averaged regrets
    x = np.arange(1,T+1)
    average_regrets()

    if Do_cumulate_regrets:
        for i in range(1,T): #changing from daily to cumulative regrets
            cumulative_regret3[i]+=cumulative_regret3[i-1]

    plt.figure(41)
    plt.plot(x, cumulative_regret3,label="averaged cumulative regret task 3")
    plt.legend()
    if not plot_individual_regrets:
        plt.show()

def plot_individual(): #showing each websites regret individually
    x = np.arange(1,T+1)
    cumulative_regret3=np.vstack(daily_regret3)
    if Do_cumulate_regrets:
        for i in range(1,T): #changing from daily to cumulative regrets
            cumulative_regret3[i,:]+=cumulative_regret3[i-1,:]
    plt.figure(0)
    plt.suptitle("Task 3")
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
    #plt.matshow(Q)
    #plt.colorbar()
    plt.show()


def RUN():
    for day in range(T):
        dailyloop(day)
    if any([plot_average_regrets,plot_individual_regrets]):
        normalize_regrets()
    if plot_average_regrets:
        plot_average()
    if plot_individual_regrets:
        plot_individual()

#cProfile.run(RUN())
RUN()