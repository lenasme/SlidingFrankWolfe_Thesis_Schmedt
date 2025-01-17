import numpy as np

import random
import itertools
from  Cheeger.rectangular_set import RectangularSet 
#import matpy as mp
#import matplotlib.pyplot as plt


class GroundTruth:
    def __init__(self, imgsz=120, max_jumps=4, seed=None):
        self.imgsz = imgsz
        self.max_jumps = max_jumps
        self.seed = seed
        np.random.seed(seed)


#get_jump_points_bin: Construct a list of jump points with minimal distance in delta_bin 


    def get_jump_points_bin(self, delta_bin, npoints = 1,maxtries=5e05):
        i = 0
        pointlist = []    
        while i < maxtries * npoints and len(pointlist) < npoints:
            M = np.random.randint(2 , self.max_jumps + 1)
            # select random points. note: equal points will be rejected below
            points_x = np.sort(np.random.randint(0 , self.imgsz,size=(M)))
            difs = np.concatenate( [points_x[1:] - points_x[0:-1], np.array([(self.imgsz - points_x[-1]) + points_x[0] ])],axis=0)

            if delta_bin[0] <= difs.min()/float(self.imgsz) < delta_bin[1]:
                N = np.random.randint(2 , self.max_jumps + 1)
                points_y = np.sort(np.random.randint(0 , self.imgsz, size=(N)))
                difs = np.concatenate( [points_y[1:] - points_y[0:-1], np.array([self.imgsz - points_y[-1] + points_y[0] ])],axis=0)

                if delta_bin[0] <= difs.min()/float(self.imgsz) < delta_bin[1]:
                    pointlist.append([points_x.tolist(),points_y.tolist()])
        
            i += 1
        
        if len(pointlist) < npoints:
            raise Warning("Found only " + str(len(pointlist)) + " of " + str(npoints) +" points for delta-bin: " + str(delta_bin))
    
        return pointlist

#get_jump_points: Construct jump points for several delta_bins

    def get_jump_points(self, deltas,npoints =1,maxtries=5e05):
        w = (deltas[1] - deltas[0])/2
        delta_bins = [ [delta - w, delta + w] for delta in deltas]
    
    
    #if  delta_bins[0][1] < 1.0/float(imgsz):
     #   print('Delta bin: ' + str(delta_bins[0]))
      #  raise Warning("Delta-bin impossible to match: Pixel distance < " + str(imgsz*delta_bins[0][0]))
    #if  delta_bins[-1][0] >= 0.33:
     #   print('Delta bin: ' + str(delta_bins[-1]))
      #  raise Warning("Delta-bin impossible to match: Pixel distance > " + str(delta_bins[-1][-1]*imgsz))
    #for delta_bin in delta_bins:
     #   if np.ceil(delta_bin[0]*imgsz) == np.ceil(delta_bin[1]*imgsz):
      #      print('Delta bin: ' + str(delta_bin))
       #     raise Warning("Delta-bin impossible to match: Interval: " + str(delta_bin[0]*imgsz) + ' / ' +  str(delta_bin[1]*imgsz))
    
        data = {}
        for i, delta_bin in enumerate(delta_bins):
            data[deltas[i]] = self.get_jump_points_bin(delta_bin, npoints=npoints, maxtries=maxtries)
            print(f"Finished bin: {delta_bin}")
        
        return data

#grad_per: Compute gradient -> for consistent gradient direction

    def grad_per(self, img):

        grad = np.zeros(img.shape + (2,))
        # Dx
        grad[:-1,:,0] = img[1:,:] - img[:-1,:]
        grad[-1,:,0] = img[0,:] - img[-1,:]
    
        # Dy
        grad[:,:-1,1] = img[:,1:] - img[:,:-1]
        grad[:,-1,1] = img[:,0] - img[:,-1]
    
        return grad

#test_grad: test wheteher the gradient is valid
    def test_grad(self, vals, show=False, verbose=False):

        grad = self.grad_per(vals)
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


#Ensure valid values in the constant pieces of the image
    def get_valid_values(self, M, N, img=False):
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
    
            grad_passed = self.test_grad(vals)[0]
        
            grad_mag = np.abs(self.grad_per(vals)).sum()/(N*M)
            if grad_mag < eps:
                grad_passed = False
                print('Dedected zero gradient - retrying')
        
        
            counter +=1


        if not grad_passed:
            
            grad_passed = self.test_grad(vals,verbose=True)[0]
            raise Warning("Gradients not valid")
        
        return vals


    def color_image(self, data, points):

        lx = len(points[0])
        ly = len(points[1])

        u = np.zeros((self.imgsz, self.imgsz))

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


    def get_image(self, points, valid=True):
        lx = len(points[0])
        ly = len(points[1])

        if valid:
            data = self.get_valid_values(lx,ly)
        else:
            #np.random.seed(seed)
            data = np.random.uniform(0,1,size=(lx,ly))

        return self.color_image(data, points) , data 


#Additional constraint that the integral of the ground truth vanishes (required in my application)

    def get_image_vanishing_integral(self, points, valid=True):
        img, data = self.get_image(points, valid=True)
        pixel_num= img.size
        pixel_sum = np.sum(img)
    
        minuend = (1/pixel_num) * pixel_sum
    
        img_van = img - minuend
        data_van = data - minuend
    
        return img_van, data_van


    #def convert_into_simple_function(self):

# um den groundtruth als SimpleSet zu framen
    def create_rectangular_sets(self, jump_points):
        """
        Erstellt RectangularSet-Objekte aus den Jump Points.
        :param jump_points: Liste von Sprungpunkten [[horizontal], [vertikal]].
        :return: Liste von RectangularSet-Objekten.
        """
        # Extrahiere horizontale und vertikale Sprungpunkte
        horizontal_points = [0] + sorted(jump_points[0]) + [self.imgsz]
        vertical_points = [0] + sorted(jump_points[1]) + [self.imgsz]

        # Erstelle die Rechtecke
        rectangular_sets = []
        for i in range(len(horizontal_points) - 1):
            for j in range(len(vertical_points) - 1):
                xmin = horizontal_points[i]
                xmax = horizontal_points[i + 1]
                ymin = vertical_points[j]
                ymax = vertical_points[j + 1]
                boundaries = np.array([ [xmin, y_min] ,[ xmax, ymin] , [xmax, ymax], [xmin,ymax] ])
                # Erstelle ein RectangularSet
                rectangular_set = RectangularSet( boundaries )
                rectangular_sets.append(rectangular_set)

        return rectangular_sets



class EtaObservation:
    def __init__(self, cut_f, reg_par= 2 , variance= 0.1):
        self.cut_f = cut_f
        self.variance = variance
        self.reg_par = reg_par

    # image: u aus der jeweiligen iteration
    def trunc_fourier(self, image):
        N,M = image.shape
        fourier_transform = np.fft.fft2(image)/(np.sqrt(N * M)) # orthogonal
        # Truncated:
        mask = np.zeros(fourier_transform.shape)
        for i in range(-self.cut_f + 1,self.cut_f):
            for j in range(-self.cut_f + 1, self.cut_f):
                mask[i,j] = 1.0
            
        truncated_transform = fourier_transform * mask
    
        # Symmetrization
        tmp = np.zeros(truncated_transform.shape,dtype=complex)
   
        for i in range(0,N):
            for j in range(0,M):
                tmp[i,j] = 0.5*(truncated_transform[i,j] + truncated_transform[-i,-j].conj()) # variance

        return tmp


    #image: ground_truth zur berechnung von f_noisy
    def trunc_fourier_noise (self, image):
        N,M = image.shape
        fourier_transform = np.fft.fft2(image)/(np.sqrt(N * M)) # orthogonal
    
    
        ncoeff = float((2*(self.cut_f-1) + 1)*(2*(self.cut_f-1) + 1))
         
        fourier_transform.real += np.random.normal(loc=0.0, scale=np.sqrt(2*self.variance/ncoeff), size= fourier_transform.shape)
        fourier_transform.imag += np.random.normal(loc=0.0, scale=np.sqrt(2*self.variance/ncoeff), size= fourier_transform.shape)

        # Truncated:
        mask = np.zeros(fourier_transform.shape)
    
        for i in range(-self.cut_f + 1, self.cut_f):
            for j in range(-self.cut_f + 1, self.cut_f):
        
                mask[i,j] = 1.0
            
        truncated_transform = fourier_transform * mask
    
        # Symmetrization
        tmp = np.zeros(truncated_transform.shape, dtype=complex)
   
        for i in range(0,N):
            for j in range(0,M):
                tmp[i,j] = 0.5*(truncated_transform[i,j] + truncated_transform[-i,-j].conj()) # variance

        return tmp

    @staticmethod
    def preadjoint_trunc_fourier ( freq ):
        N,M = freq.shape
        back_transf = np.fft.ifft2(freq)*np.sqrt(N*M)
        return back_transf
    
    @staticmethod
    def preadjoint_trunc_fourier_vanish_int ( freq ):
        N,M = freq.shape
        back_transf = np.fft.ifft2(freq)*np.sqrt(N*M)
        back_transf = back_transf - 1/(N*M) * np.sum(back_transf)
        return back_transf
    

    def f_noisy (self, ground_truth):
        return self.trunc_fourier_noise (ground_truth)


# ingredients of eta = -1/alpha (K^#(Ku-f))

    def eta (self, image, ground_truth):
        # image: aktuelles rekonstruiertes Bild einer Iteration
        # f_delta 
        return -1 / self.reg_par * ( self.preadjoint_trunc_fourier(  self.trunc_fourier(image) - self.f_noisy(ground_truth)) )


    def eta_vanish_int (self, image, ground_truth):
    # image: aktuelles rekonstruiertes Bild einer Iteration
    # f_delta 
        return -1 / self.reg_par * ( self.preadjoint_trunc_fourier_vanish_int( self.trunc_fourier(image) - self.f_noisy(ground_truth)) )






