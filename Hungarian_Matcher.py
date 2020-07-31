import numpy as np


class Hungarian_Matcher:
    def __init__(self, weights):
        self.weights = weights
        self.matrix = weights
        self.assignments = np.zeros(self.weights.shape, dtype=int)
        self.dimx = self.weights.shape[0]
        self.dimy = self.weights.shape[1]


    def normalize(self):
        if self.dimx > self.dimy:
            self.weights = np.append(self.weights, np.zeros([self.dimx, 1]), 1)

        if self.dimy > self.dimx:
            self.weights = np.append(self.weights, np.zeros([1, self.dimy]), 0)

        for i in range(self.dimx):
            self.weights[i,:] = self.weights[i,:] - np.min(self.weights[i,:])

        for j in range (self.dimy):
            self.weights[:,j] = self.weights[:,j] - np.min(self.weights[:,j])


    def assign(self):
        assigned = np.array([])
        for i in range(0,self.dimx):
            for j in range(0,self.dimy):
                if (self.weights[i,j]==0 and np.sum(self.assignments[i,:])==0 and np.sum(self.assignments[:,j])==0):
                    self.assignments[i,j] = 1
                    assigned = np.append(assigned,i)

        rows = np.linspace(0, self.dimx-1, self.dimx).astype(int)
        marked_rows = np.setdiff1d(rows, assigned)
        new_marked_rows = marked_rows.copy()
        marked_columns = np.array([])


        while(len(new_marked_rows)>0):
            new_marked_columns = np.array([], dtype=int)
            for r in new_marked_rows:
                zeros_columns = np.argwhere(self.weights[r,:]==0).reshape(-1)
                new_marked_columns = np.append(new_marked_columns, np.setdiff1d(zeros_columns, marked_columns))

            marked_columns = np.append(marked_columns,new_marked_columns)
            new_marked_rows = np.array([], dtype=int)


            for c in new_marked_columns:
                new_marked_rows = np.append(new_marked_rows, np.argwhere(self.assignments[:,c]==1).reshape(-1))

            marked_rows = np.unique(np.append(marked_rows,new_marked_rows))

        return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_columns)




    def normalize_uncovered(self, covered_rows, covered_columns):
        uncovered_rows = np.setdiff1d(np.linspace(0, self.dimx-1, self.dimx), covered_rows).astype(int)
        uncovered_columns = np.setdiff1d(np.linspace(0, self.dimy-1, self.dimy), covered_columns).astype(int)
        min_val = np.max(self.weights)

        for i in uncovered_rows.astype(int):
            for j in uncovered_columns.astype(int):
                if (self.weights[i,j] < min_val):
                    min_val = self.weights[i,j]

        for i in uncovered_rows.astype(int):
            self.weights[i,:] -= min_val

        for j in covered_columns.astype(int):
            self.weights[:,j] += min_val



    def rows_single_zero(self,m):
        for i in range(0, m.shape[0]):
            if (np.sum(m[i,:]== 0)== 1):
                j = np.argwhere(m[i,:] == 0).reshape(-1)[0]
                return i,j
        return False

    def columns_single_zero(self, m):
        for i in range(0, m.shape[1]):
            if (np.sum(m[:,i] == 0) == 1):
                j = np.argwhere(m[:,i] == 0).reshape(-1)[0]
                return i, j
        return False




    def assign_single_zero_lines(self,assignment):
        val = self.rows_single_zero(self.weights)
        while(val):
            i,j = val[0], val[1]
            self.weights[i,j] +=1
            self.weights[:,j] -=1
            assignment[i,j] = 1
            val = self.rows_single_zero(self.weights)

        val = self.columns_single_zero(self.weights)
        while(val):
            i,j = val[0], val[1]
            self.weights[i,:] +=1
            self.weights[i,j] -=1
            assignment[i,j] = 1
            val = self.columns_single_zero(self.weights)

        return assignment


    def final_assignment(self,initial_matrix):
        assignment = np.zeros(self.weights.shape, dtype=int)
        assignment = self.assign_single_zero_lines(assignment)

        while(np.sum(self.weights==0) >0):
            i,j = np.argwhere(self.weights==0)[0][0], np.argwhere(self.weights==0)[0][1]
            assignment[i,j] = 1
            self.weights[i,:] +=1
            self.weights[:,j] +=1

        assignment = self.assign_single_zero_lines(assignment)

        return  assignment* initial_matrix, assignment

    def hungarian_algo(self):
        self.normalize()
        n_lines = 0
        max_length = np.maximum(self.dimx,self.dimy)
        while (n_lines != max_length):
            lines = self.assign()
            n_lines = len(lines[0]) + len(lines[1])
            if n_lines != max_length:
                self.normalize_uncovered(lines[0], lines[1])
        return self.final_assignment(self.matrix)









