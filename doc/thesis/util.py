################################################################
    def _initialization(self):
        a = np.random.random((self.n,self.m))
        b = np.random.random((self.T,self.n))
        b /= sum(b)
        return a,b

#################################################################
    def pre_training(self,x):
      A_list,B_list = self.nnsc(x)
      return A_list,B_list

################################################################
    @staticmethod
    def _pos_constraint(a):
        indices = np.where(a < 0.0)
        a[indices] = 0.0
        return a
##################################################################
    def predict(self,A,B):
        x = map(lambda x,y: x.dot(y),B,A)
        return x
##################################################################
    def F(self,x,B,x_train=None,A=None,rp_tep=False,rp_gl=False):
        '''
        input is lists of the elements
        output list of elements
        '''
        # 4b
        B = np.asarray(B)
        A = np.asarray(A)
        coder = SparseCoder(dictionary=B.T,
                            transform_alpha=self.rp, transform_algorithm='lasso_cd')
        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = self._pos_constraint(acts)


        return acts
