#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import random
import itertools
#import matpy as mp
import matplotlib.pyplot as plt


# In[2]:


def get_jump_points_bin(delta_bin, npoints = 1,maxtries=5e05,maxjumps=4,imgsz=120,seed=None):
    

    np.random.seed(seed)
    i = 0

    pointlist = []    

    
    while i<maxtries*npoints and len(pointlist)<npoints:
    
        M = np.random.randint(2,maxjumps+1)

        # select random points. note: equal points will be rejected below
        points_x = np.sort(np.random.randint(0,imgsz,size=(M)))
    
        difs = np.concatenate( [points_x[1:] - points_x[0:-1], np.array([(imgsz - points_x[-1]) + points_x[0] ])],axis=0)
        

        if delta_bin[0] <= difs.min()/float(imgsz) < delta_bin[1]:
        
            N = np.random.randint(2,maxjumps+1)
            points_y = np.sort(np.random.randint(0,imgsz,size=(N)))
        
            difs = np.concatenate( [points_y[1:] - points_y[0:-1], np.array([imgsz - points_y[-1] + points_y[0] ])],axis=0)
            
            if delta_bin[0] <= difs.min()/float(imgsz) < delta_bin[1]:
                
                pointlist.append([points_x.tolist(),points_y.tolist()])
        
        i += 1
        
    if len(pointlist)<npoints:
        raise Warning("Found only " + str(len(pointlist)) + " of " + str(npoints) +" points for delta-bin: " + str(delta_bin))
    

    return pointlist


# In[3]:


print(get_jump_points_bin([0.02-0.005,0.02+0.005],npoints = 1,maxtries=5e05,maxjumps=15,imgsz=120,seed=None))


# In[4]:


def get_jump_points(deltas,npoints =1,maxtries=5e05,maxjumps = 4,imgsz=120,seed=None):


    w = (deltas[1] - deltas[0])/2
    
    
    delta_bins = [ [delta-w,delta + w] for delta in deltas]
    
    
    if  delta_bins[0][1] < 1.0/float(imgsz):
        print('Delta bin: ' + str(delta_bins[0]))
        raise Warning("Delta-bin impossible to match: Pixel distance < " + str(imgsz*delta_bins[0][0]))
    if  delta_bins[-1][0] >= 0.33:
        print('Delta bin: ' + str(delta_bins[-1]))
        raise Warning("Delta-bin impossible to match: Pixel distance > " + str(delta_bins[-1][-1]*imgsz))
    for delta_bin in delta_bins:
        if np.ceil(delta_bin[0]*imgsz) == np.ceil(delta_bin[1]*imgsz):
            print('Delta bin: ' + str(delta_bin))
            raise Warning("Delta-bin impossible to match: Interval: " + str(delta_bin[0]*imgsz) + ' / ' +  str(delta_bin[1]*imgsz))
    
    data = {}
    
    # Get lists of jump points
    for i in range(len(delta_bins)):
        
        data[deltas[i]] = get_jump_points_bin(delta_bins[i],npoints = npoints,maxtries=maxtries,maxjumps=maxjumps,imgsz=imgsz,seed=seed)
        print('Finished bin: '  + str(delta_bins[i]))
    
    return data

deltas = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

print(get_jump_points(deltas,npoints =3,maxtries=5e05,maxjumps = 4,imgsz=120,seed=None))


# In[5]:


def grad_per(img):

    grad = np.zeros(img.shape + (2,))
    # Dx
    grad[:-1,:,0] = img[1:,:] - img[:-1,:]
    grad[-1,:,0] = img[0,:] - img[-1,:]
    
    # Dy
    grad[:,:-1,1] = img[:,1:] - img[:,:-1]
    grad[:,-1,1] = img[:,0] - img[:,-1]
    
    return grad



def test_grad(vals,show=False,verbose=False):

    grad = grad_per(vals)
    gradx = grad[...,0]
    grady = grad[...,1]

    gradx = np.sign(gradx)
    grady = np.sign(grady)

    
    passed = True

    #Test gradient y
    gxpos = np.sign(np.maximum(gradx,0).sum(axis=1))
    gxneg = np.sign(np.minimum(gradx,0).sum(axis=1))
    
    gx_test = -5*np.ones(gxpos.shape) #
    
    # non-admissible values
    gx_test[gxpos == 1] = -1
    gx_test[gxneg == -1] = 1

    gx_test = gx_test[:,np.newaxis]

    n_gx_invalid = np.count_nonzero(gradx == gx_test)
    if n_gx_invalid:
        if verbose:
            print('Problem with gx')
            res = mp.output({})
            res.vals = vals
            res.gradx = gradx
            res.gx_test = gx_test
            mp.psave('wrong_grad',res)
        
        passed = False

    #Test gradient y
    gypos = np.sign(np.maximum(grady,0).sum(axis=0))
    gyneg = np.sign(np.minimum(grady,0).sum(axis=0))
    
    gy_test = -5*np.ones(gypos.shape) #
    
    # non-admissible values
    gy_test[gypos == 1] = -1
    gy_test[gyneg == -1] = 1

    gy_test = gy_test[np.newaxis,:]
    
    n_gy_invalid = np.count_nonzero(grady == gy_test)
    if np.any(grady == gy_test):
        if verbose:
            print('Problem with gy')
            if passed:
                res = mp.output({})
                res.vals = vals
                res.grady = grady
                res.gy_test = gy_test
                mp.psave('wrong_grad',res)

        passed = False

    if show:
        mp.imshow(gradx)
        mp.imshow(grady)
        

    if passed and verbose:
        print('Gradient valid.')
    
    n_g_invalid = n_gx_invalid + n_gx_invalid
    return passed,n_g_invalid


def get_valid_values(M,N,seed=None,img=False):

    np.random.seed(seed)
    
    grad_passed = False
    counter = 0
    
    eps = 1e-08
    
    while not grad_passed and counter < 10:
        
        if counter>0:
            print('Try. nr: ' + str(counter))
        
        if np.any(img):
            (M,N) = img.shape

        points = list(itertools.product(range(M), range(N)))
        random.shuffle(points)
        
        eps = 1e-09 # small tolerance to avoid exact equality
        
        dims = (M,N) # point dimensions
        
        vals = -np.ones(dims) # container for values, -1 means unset
        grad = [np.zeros(M),np.zeros(N)] # container for gradients
        
        
        
        # Stancil for value comparison (to be shuffled)
        stencil = [[1,0],[-1,0],[0,1],[0,-1]]

        for point in points:
        

            # Define possible value range
            mx = 1.0
            mn = 0.0
            
            random.shuffle(stencil) # randomly select order of stencil points
            for dx in stencil: #loop over stencil points

                idx = ((point[0]+dx[0])%M,(point[1]+dx[1])%N) #index of neighboring pixel

                if vals[idx] != -1: #if value is already set
                    
                    ax = 0 if dx[0] !=0 else 1 #set axis of stencil
                    # compare along axis ax
                    if grad[ax][ (point[ax]+min(dx[ax],0))%dims[ax] ] == dx[ax]: # neighboring pixel must be larger
                        mx_tmp = mx
                        mx = min(mx,vals[idx])
                        # correct if upper bound is too small
                        if mn>mx:
                            i = 0
                            while (grad[ax][ (point[ax]+min(dx[ax],0)+i*dx[ax])%dims[ax] ] == dx[ax]) & (i<dims[ax]):
                                pos = (point[ax]+(i+1)*dx[ax])%dims[ax] #current position
                                if ax==0: #first axis
                                    setvals = vals[pos,:] != -1
                                    vals[pos,setvals] += (mn-mx+eps)
                                    vals[pos,setvals] = np.clip(vals[pos,setvals],0.0,1.0)
                                else: #second axis
                                    setvals = vals[:,pos] != -1
                                    vals[setvals,pos] += (mn-mx+eps)
                                    vals[setvals,pos] = np.clip(vals[setvals,pos],0.0,1.0)
                                i += 1  
                            mx = min(mx_tmp,vals[idx]) #new maximum
                            
                    if grad[ax][ (point[ax]+min(dx[ax],0))%dims[ax] ] == - dx[ax]: # neighboring pixel must be smaller
                        mn_tmp = mn
                        mn = max(mn,vals[idx])
                        #correct if lower bound is too high
                        if mn>mx:
                            i = 0
                            while (grad[ax][ (point[ax]+min(dx[ax],0)+i*dx[ax])%dims[ax] ] == -dx[ax]) & (i<dims[ax]):
                                pos = (point[ax]+(i+1)*dx[ax])%dims[ax] #current position
                                if ax==0: #first axis
                                    setvals = vals[pos,:] != -1
                                    vals[pos,setvals] -= (mn-mx+eps)
                                    vals[pos,setvals] = np.clip(vals[pos,setvals],0.0,1.0)
                                else: #second axis
                                    setvals = vals[:,pos] != -1
                                    vals[setvals,pos] -= (mn-mx+eps)
                                    vals[setvals,pos] = np.clip(vals[setvals,pos],0.0,1.0)
                                i += 1
                            mn = max(mn_tmp,vals[idx]) #new minimum
       
            #Set value
            if not np.any(img):
                vals[point] = np.random.uniform(mn,mx)
                
            else:
                vals[point] = np.clip(img[point],mn,mx)
            
            if mn>mx:
                print('problem with mx/mn: ' + str(mx) + ' / ' + str(mn))
                print('value: ' + str(vals[point]))
            
            
            # Define resulting gradients
            for dx in stencil: #loop over neighboring pixels
                idx = ((point[0]+dx[0])%M,(point[1]+dx[1])%N) #index of neighboring pixel
                
                if vals[idx] != -1: #if value is already set
                    
                    ax = 0 if dx[0] !=0 else 1 #set axis of stencil
                    
                    if not grad[ax][ (point[ax]+min(dx[ax],0))%dims[ax] ]: # if gradient is not yet set
                        grad[ax][(point[ax]+min(dx[ax],0))%dims[ax]] = np.sign(vals[idx] - vals[point])*dx[ax]
    
        grad_passed = test_grad(vals)[0]
        
        grad_mag = np.abs(grad_per(vals)).sum()/(N*M)
        if grad_mag < eps:
            grad_passed = False
            print('Dedected zero gradient - retrying')
        
        
        counter +=1


    if not grad_passed:
            
        grad_passed = test_grad(vals,verbose=True)[0]
        raise Warning("Gradients not valid")
        
    return vals

print(get_valid_values(4,3,seed=None,img=False))


# In[6]:


def color_image(data, points, imgsz=120):

    lx = len(points[0])
    ly = len(points[1])

    u = np.zeros((imgsz,imgsz))

    for idx in range(lx-1) :
        for idy in range(ly-1):
        
            u[points[0][idx]:points[0][idx+1],points[1][idy]:points[1][idy+1]] = data[idx,idy]
        
        u[points[0][idx]:points[0][idx+1],:points[1][0]] = data[idx,-1]
        u[points[0][idx]:points[0][idx+1],points[1][-1]:] = data[idx,-1]

    for idy in range(ly-1):
    
        u[:points[0][0],points[1][idy]:points[1][idy+1]] = data[-1,idy]
        u[points[0][-1]:,points[1][idy]:points[1][idy+1]] = data[-1,idy]
    
    u[:points[0][0],:points[1][0]] = data[-1,-1]
    u[:points[0][0],points[1][-1]:] = data[-1,-1]
    u[points[0][-1]:,:points[1][0]] = data[-1,-1]
    u[points[0][-1]:,points[1][-1]:] = data[-1,-1]

    return u
print(color_image(get_valid_values(4,3,seed=None,img=False),[[7, 34, 36], [24, 38, 40, 57]],imgsz=120))


# In[7]:


def get_image(points,imgsz=120,valid=True,seed=None):

    
    lx = len(points[0])
    ly = len(points[1])


    if valid:
        data = get_valid_values(lx,ly,seed=seed)
    else:
        np.random.seed(seed)
        data = np.random.uniform(0,1,size=(lx,ly))

    return color_image(data,points,imgsz=imgsz) , data 




# In[8]:


# zusätzlich 0-Integral meines Groundtruth, da ich mich in meiner Anwendung auf solche Funktionen beschränke

def get_image_vanishing_integral(points, imgsz=120, valid=True, seed=None):
    img, data = get_image(points,imgsz=120,valid=True,seed=seed)
    pixel_num= img.size
    pixel_sum = np.sum(img)
    
    minuend = (1/pixel_num) * pixel_sum
    
    img_van = img - minuend
    data_van = data - minuend
    
    return img_van, data_van


# In[9]:


points= get_jump_points_bin([0.06-0.005,0.06+0.005],npoints = 1,maxtries=5e05,maxjumps=15,imgsz=120,seed=None)[0]

u, values =  get_image_vanishing_integral(points,imgsz=120,valid=True,seed=None)

print(points)
print(values)
plt.imshow(u, cmap='binary')
plt.colorbar()  # Optional: Adds a color scale to the side
plt.title("Visualized Image of u")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

print("X-Punkte:", points[0])
print("Y-Punkte:", points[1])

plt.scatter([0]*len(points[0]), points[0] , color='red', label='Sprungstellen X')
plt.scatter( points[1], [0]*len(points[1]), color='blue', label='Sprungstellen Y')
plt.legend()
plt.title("Sprungstellen Visualisierung")
plt.show()


# In[10]:


def tft_img(img, cut_f):
    
    
    d0 = np.fft.fft2(img)/np.sqrt(img.shape[0]*img.shape[1]) #We use the orthogonal dft
    
    # Store noise-free data (will be weighted by mask below)
    
    
    #Set variables
    N,M = img.shape
    
    mask = np.zeros(d0.shape)
    
    for i in range(-cut_f + 1,cut_f):
        for j in range(-cut_f + 1,cut_f):
        
            mask[i,j] = 1.0
            
    d0 *= mask
    
    
    tmp = np.zeros(d0.shape,dtype=complex)
    #Symmetrize data
    for i in range(0,N):
        for j in range(0,M):
            tmp[i,j] = 0.5*(d0[i,j] + d0[-i,-j].conj()) # variance

    d0 = tmp
    
    # Zero-fill recon
    rec0 = (np.fft.ifft2(d0)*np.sqrt(N*M)).real #We use the orthogonal dft
    
    return d0,  rec0



# In[11]:


d0, rec0 =  tft_img(u, 10)

plt.imshow(u, cmap='gray')
plt.title("Rekonstruiertes Bild")
plt.colorbar()
plt.show()

plt.imshow(rec0, cmap='gray')
plt.title("Rekonstruiertes Bild")
plt.colorbar()
plt.show()

plt.imshow(np.angle(d0), cmap='hsv')
plt.title("Rekonstruiertes Bild")
plt.colorbar()
plt.show()


# In[12]:


get_ipython().system('jupyter nbconvert --to script GroundTruth_erstellen_test_aus_Wirth.ipynb')


# In[ ]:




