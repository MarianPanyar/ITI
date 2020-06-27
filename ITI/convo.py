import numpy as np

class convolution(object):
    #cut vision into squares
    def __init__(self,csize,radius,stride=1,padding=True,channel=1):
        #csize: original size [x,y]
        #radius: radius of kernel
        self.csize=csize
        self.radius=radius
        self.diameter=radius*2+1
        self.stride=stride
        self.padding=padding
        self.channel=channel
        self.cv_index()
    

    def pd(self,data,margin=-1,pdvalue=0):
        #1D to 2D, padding, 2D to 1D
        #margin: padding margin, default=radius
        if margin==-1:
            margin=self.radius
        #reshape data to csize square
        sqdata=np.reshape(data,(len(data),self.csize[0],self.csize[1]))
        pdsqdata=np.pad(sqdata,margin,'constant',constant_values=pdvalue)[margin:-margin]
        pddata=np.reshape(pdsqdata,(len(data),(self.csize[0]+2*margin)*(self.csize[1]+2*margin)))
        return pddata
        
    def cv_index(self):
        #return index of tiles in 1D
        if self.padding!=True:
            self.csize[0]=self.csize[0]-2*self.radius
            self.csize[1]=self.csize[1]-2*self.radius
            #if no padding, size should -2*radius
        self.index=[]
        for i in range(self.csize[0]):
            for j in range(self.csize[1]):
                localindex = []
                for m in range(self.diameter):
                    for n in range(self.diameter):
                        d_index = [(i+m)*(self.csize[1]+self.radius*2)+j+n]
                        localindex.append(d_index)
                if i%self.stride==0 and j%self.stride==0:
                    self.index.append(localindex)
        return self.index
    
    def to_tiles(self,data):
        #output all tiles as 1D array, for training or forwarding
        if self.padding==True:
            pddata=self.pd(data)
        else:
            pddata=data
        self.tiles = pddata[:,self.index][:,:,:,0]
        if self.channel>1:
            self.tiles=np.reshape(self.tiles,(np.shape(self.tiles)[0],np.shape(self.tiles)[1],np.shape(self.tiles)[2]*np.shape(self.tiles)[3]))
            #if multichannel, output combine channels
        return self.tiles
    
    def norm(self,data):
        #normalization for batch of 1D tiles
        mean = np.mean(data,axis=1)
        if self.channel==1:
            nordata = data/(np.tile(mean,(self.diameter**2,1)).T)
        elif self.channel>1:
            nordata = data/(np.tile(mean,(self.channel*self.diameter**2,1)).T)
        return nordata
    
    def wt_tiles(self,data,tres=0.1,normalize=True):
        #generate 'well-temperated' tiles, a.k.a. beyond treshold, and normalized tiles, for training
        self.to_tiles(data)
        all_tiles = np.reshape(self.tiles,(np.shape(self.tiles)[0]*np.shape(self.tiles)[1],np.shape(self.tiles)[2]))
        av_index = np.argwhere(np.sum(all_tiles,axis=1)>tres)
        nice_tiles = all_tiles[av_index][:,0,:]
        if normalize==True:
            nice_tiles=self.norm(nice_tiles)
        return nice_tiles
    
    def to_forward(self,data,UOM):
        #forward for multi-channel matrix
        forward=[]
        t_data = self.to_tiles(data)
        for i in t_data:
            cl_forward = UOM.find_bmu(i)[1]
            forward.append(cl_forward)
        forward=np.asarray(forward)
        return forward

class pooling(object):
    def __init__(self,csize,diameter,stride=1,mode='max'):
        self.csize=csize
        self.diameter=diameter
        self.stride=stride
        self.mode=mode
        self.pl_index()
        
    def pl_index(self):
        self.index=[]
        for i in range(self.csize[0]-self.diameter+1):
            for j in range(self.csize[1]-self.diameter+1):
                localindex = []
                for m in range(self.diameter):
                    for n in range(self.diameter):
                        d_index = [(i+m)*(self.csize[1])+j+n]
                        localindex.append(d_index)
                if i%self.stride==0 and j%self.stride==0:
                    self.index.append(localindex)
        return self.index
    
    def to_pooling(self,data):
        tiles = data[:,self.index]
        tiles = tiles[:,:,:,0]
        if self.mode=='max':
            pooled = np.max(tiles,axis=2)
        elif self.mode=='average':
            pooled = np.mean(tiles,axis=2)
        elif self.mode=='hnh':
            pooled = (np.max(tiles,axis=2)+np.mean(tiles,axis=2))/2
        elif self.mode=='free':
            pooled = self.free_pooling(data)
        else:
            raise ValueError(self.mode,'mode is not ready yet.')
        return pooled
    
#    def free_pooling(self,data):

class LGN(object):
    #some basics,like on/off-center, colors2grey, etc...
    def __init__(self,data,csize,channel='mono'):
        self.csize=csize
        if channel=='mono':
            self.sqdata=np.reshape(data,(len(data),csize,csize))
        elif channel=='RGB':
            grey_data=np.mean(data,axis=2)
            self.sqdata=np.reshape(grey_data,(len(data),csize,csize))
            self.c_sqdata=np.reshape(data,(len(data),csize,csize,3))
        else:
            raise ValueError(channel,'is not ready yet.')
    
    def oocenter(self,tres=0):
        dfdata = 4*self.sqdata[:,1:-1,1:-1]-self.sqdata[:,:-2,:-2]-self.sqdata[:,2:,2:]-self.sqdata[:,:-2,2:]-self.sqdata[:,2:,:-2]
        on_center=dfdata+0
        off_center=dfdata+0
        on_center[on_center<tres]=0
        off_center[off_center>(-tres)]=0
        plt.matshow(-on_center[3])
        return on_center,-off_center
