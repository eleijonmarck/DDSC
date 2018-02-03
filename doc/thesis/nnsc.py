################################################################
    def NonNegativeSparseCoding(self,appliances):
        '''
        Method as in NNSC from nonnegative sparse coding finland.
        from P.Hoyer
        TODO : (ericle@kth.se)
        '''
        epsilon = 0.01
        A_list = []
        B_list = []
        for x in appliances:
            A,B = self._initialization()
            Ap = A
            Bp = B
            Ap1 = Ap
            Bp1 = Bp
            t = 0
            change = 1
            while t <= self.steps and self.epsilon <= change:
                # 2a
                Bp = Bp - self.alpha*np.dot((np.dot(Bp,Ap) - x),Ap.T)
                # 2b
                Bp = self._pos_constraint(Bp)
                # 2c
                Bp /= sum(Bp)
                # element wise division
                dot2 = np.divide(np.dot(Bp.T,x),(np.dot(np.dot(Bp.T,Bp),Ap) + self.rp))
                # 2d
                Ap = np.multiply(Ap,dot2)

                change = np.linalg.norm(Ap - Ap1)
                Ap1 = Ap
                Bp1 = Bp
                t += 1

            print "Gone through one appliance"
            A_list.append(Ap)
            B_list.append(Bp)

        return A_list,B_list
################################################################
