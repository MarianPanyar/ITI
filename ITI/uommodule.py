import numpy as np

class UOM(object):
    #A SOM without 2D structure
    #much faster
    #nnodes: number of neurons
    #alpha: renew rates
    def __init__(self,nnodes,alpha):
        self.nnodes = nnodes
        self.alpha = alpha
        
    def mxinit(self,indata,init='sample'):
        #sample: choose random sample from indata
        #tabula_rasa: random value    
        self.indata = indata
        if init == 'sample':
            index = np.random.randint(len(self.indata),size=self.nnodes)
            self.matrix = self.indata[index]
        elif init == 'tabula_rasa':
            self.matrix = np.random.rand(self.nnodes,np.shape(self.indata)[1])
        self.matrix = self.normalize(self.matrix)
        return self.matrix
    
    def find_bmu(self,data):
        f_rate = np.dot(data,self.matrix.T)
        bmu = np.argmax(f_rate,axis=1)
        return bmu,f_rate
    
    def train(self,data,r_mode,treshold):
        #update bmu in matrix with according sample
        data = self.normalize(data)
        bmu = self.find_bmu(data)
        self.renew(bmu,data,r_mode,treshold)
        #np.dot(data.T,np.max(bmu[1],axis=1)).T
        self.matrix = self.normalize(self.matrix)
        return self.matrix
    
    def train_we(self,data,we_tr,r_mode,treshold):
        #replace inactive elements with new ones
        #we_tr: repalcement threshold for activation 
        data = self.normalize(data)
        bmu = self.find_bmu(data)
        self.renew(bmu,data,r_mode,treshold)
        hitmap = np.bincount(bmu[0])
        remain_list = np.argwhere(hitmap>we_tr)
        add_size = self.nnodes-len(remain_list)
        index = np.random.randint(len(self.indata),size=add_size)
        add_matrix = self.indata[index]
        self.matrix = self.matrix[remain_list][:,0,:]
        self.matrix = np.concatenate((self.matrix,add_matrix),axis=0)
        self.matrix = self.normalize(self.matrix)
        return self.matrix
    
    def renew(self,bmu,data,r_mode='none',treshold=0):
        #rate mode: learning rate of a unit depends on active rate
        #treshold mode: those weak activities will not be learned
        if r_mode=='none':
            self.matrix[bmu[0]] = self.matrix[bmu[0]]+self.alpha*data
        elif r_mode=='rate':
            a_rate = np.max(self.normalize(bmu[1]),axis=1)
            self.matrix[bmu[0]] = self.matrix[bmu[0]]+self.alpha*(data.T*a_rate).T
        elif r_mode=='treshold':
            a_rate = np.max(self.normalize(bmu[1]),axis=1)
            a_rate[(a_rate<treshold)]=0
            a_rate[(a_rate>=treshold)]=1
            self.matrix[bmu[0]] = self.matrix[bmu[0]]+self.alpha*(data.T*a_rate).T
        return self.matrix     
    
    def loop_train(self,train_data,loops,re_elect=False,we_tr=0.1,r_mode='none',treshold=0):
        #renew in batches
        #re_elect: to replace inactive nodes
        stl = len(train_data)//loops
        if re_elect==True:
            for i in range(loops):
                self.train_we(train_data[i*stl:(i+1)*stl],we_tr,r_mode,treshold)
        elif re_elect==False:
            for i in range(loops):
                self.train(train_data[i*stl:(i+1)*stl],r_mode,treshold)
        return self.matrix 
        
    def normalize(self,weights,mean=1,bias=0):
        newweights = (weights.T/np.sum(weights,axis=1)).T *(np.shape(weights)[1])*mean -bias
        return newweights

    