#################################################################
    def DiscriminativeDisaggregation(self,x,B,A):
        '''
        Taking the parameters as x_train_use and discriminate over the
        entire region
        '''
        A_star = np.vstack(A)
        B_cat = np.hstack(B)
        change = 1
        t = 0
        x_train_sum = self.train_set.values()
        while t <= self.steps and self.epsilon <= change:
            B_cat_p = B_cat
            # 4a
            acts = self.F(x,B_cat,A=A_star)
            # 4b
            B_cat = (B_cat-self.alpha*((x-B_cat.dot(acts))
                     .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))
            # 4c
            # scale columns s.t. b_i^(j) = 1
            B_cat = self._pos_constraint(B_cat)
            B_cat /= sum(B_cat)

            change = np.linalg.norm(B_cat - B_cat_p)
            t += 1
            print "DD change is %f and step is %d" %(change,t)

        return B_cat
#################################################################
