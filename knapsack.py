import numpy as np

''' Usage:

Input: 
daily budeget: integer
values: Estimated values // a 2D array containing estimated bids 
budget = 100
budget_discretization_density = 20
budget_discretization_steps = [i * daily_budget / budget_discretization_density for i in range(budget_discretization_density + 1)]
estimated_rewards = []
for subcampaign in subcampaigns:
    estimated_rewards.append([subcampaign.get_estimated_reward(bid) for bid in budget_discretization_steps])
optimal_super_arm = Knapsack(daily_budget, estimated_rewards).optimize()


'''
class Knapsack:
    def __init__(self, budget, values, arms = None):
        self.subcampaigns_number = len(values) + 1
        self.subcampaigns_list = list(range(len(values)))
        step = budget / (len(values[0])-1)

        self.budgets = [ i*step for i in range(len(values[0])) ]
        if arms != None:
            self.budgets = arms

        self.budget_value = budget
        self.combinations = []

        # It is a matrix: values[subcamp_id][budget_id] = value
        self.values = values

    def optimize(self):
        all_zero = self.all_zero_result()
        if all_zero[0]:
            return all_zero[1]

        res, self.combinations = self.knapsack_optimization()
        # Compute the assignment from the knapsack optimization results
        return self.compute_assignment(self.combinations[-1][-1], self.combinations.copy())

    def knapsack_optimization(self):
        numerical_results = [[0] * len(self.budgets)]
        indices = []

        for current_row in self.values:
            results_row = []
            indices_row = []
            for i in range(len(current_row)):
                best_value = 0
                best_indices = (i, 0)
                for old_index in range(0, i+1):
                    index = i - old_index
                    if(current_row[index] + numerical_results[-1][old_index] > best_value):
                        best_value = current_row[index] + numerical_results[-1][old_index]
                        best_indices = (index, old_index)
                results_row.append(best_value)
                indices_row.append(best_indices)
            numerical_results.append(results_row)
            indices.append(indices_row)

        return (numerical_results, indices)

    '''
        Returns a list of tuple of the following kind: (index of the sub-campaing, budget to assign)
    '''
    def compute_assignment(self, last_sub, combinations, assignment=None):

        if assignment is None:
            assignment = []

        assignment.append((len(combinations) - 1, self.budgets[last_sub[0]]))
        combinations.pop()

        if len(combinations) == 0:
            return assignment

        last_sub = combinations[-1][last_sub[1]]

        return self.compute_assignment(last_sub, combinations, assignment)

    def all_zero_result(self):
        final_sum = 0
        for subcamp in self.values:
            final_sum += sum(x > 0 for x in subcamp)

        if final_sum > 0:
            return (False, self.values)

        else:
            budget = self.budget_value
            res = []
            num = self.subcampaigns_number - 2
            while budget > 0 and num >= 0:
                step = budget / (len(self.values[0]) - 1)
                budgets = [i * step for i in range(len(self.values[0]))]
                budget_ass = np.random.choice(budgets, replace=True)
                res.append((num, budget_ass))
                num -= 1
                budget -= budget_ass

            if len(res) <= self.subcampaigns_number - 2:
                while num >= 0:
                    res.append((num, 0))
                    num -= 1

            # I assign to the last campaign all the budget remaining for consistency reasons
            elif budget != 0:
                res[-1] = (res[-1][0], res[-1][1] + budget)

            return (True, res)