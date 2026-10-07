"""Explicit Bernoulli likelihoods, centered least squares and bounded Lloyd clustering."""
import numpy as np

def finite_matrix(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():raise ValueError('Expected finite nonempty feature matrix')
    return x

class BernoulliClassifier:
    def __init__(self,alpha=1):
        if not np.isfinite(alpha) or alpha<=0:raise ValueError('Positive finite smoothing required')
        self.alpha=alpha;self.prob=None
    def fit(self,x,y):
        x=finite_matrix(x);y=np.asarray(y)
        if not np.isin(x,[0,1]).all() or y.ndim!=1 or len(y)!=len(x):raise ValueError('Binary features and matching labels required')
        if y.dtype.kind=='O' and all(isinstance(v,str) for v in y):y=y.astype(str)
        if y.dtype.kind not in 'iuUS':raise ValueError('Integer or string labels required')
        self.classes_,inverse=np.unique(y,return_inverse=True)
        if len(self.classes_)<2:raise ValueError('At least two classes required')
        self.class_counts=np.bincount(inverse).astype(float);p=[]
        for i,count in enumerate(self.class_counts):p.append((x[inverse==i].sum(axis=0)+self.alpha)/(count+2*self.alpha))
        p=np.stack(p);self.prob=np.stack([1-p,p],axis=-1);self.prior=self.class_counts/self.class_counts.sum()
        return self
    def joint_log_likelihood(self,x):
        if self.prob is None:raise ValueError('Fit before predicting')
        x=finite_matrix(x)
        if x.shape[1]!=self.prob.shape[1] or not np.isin(x,[0,1]).all():raise ValueError('Binary features of fitted width required')
        return x@np.log(self.prob[:,:,1]).T+(1-x)@np.log(self.prob[:,:,0]).T+np.log(self.prior)
    def predict_batch(self,x):
        scores=self.joint_log_likelihood(x);return self.classes_[scores.argmax(axis=1)]
    def predict(self,img):return self.predict_batch(np.asarray(img)[None,:])[0]

def ordinary_least_squares(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or not np.isfinite(np.r_[x,y]).all():raise ValueError('Matching finite vectors with at least two observations required')
    dx=x-x.mean();dy=y-y.mean();denominator=dx@dx
    if denominator==0:raise ValueError('Constant explanatory feature has no unique slope')
    slope=float(dx@dy/denominator);intercept=float(y.mean()-slope*x.mean())
    if not np.isfinite([slope,intercept]).all():raise ValueError('Fit overflow')
    return slope,intercept

class LloydKMeans:
    def __init__(self,k=3,max_iter=100,tolerance=1e-6,seed=42):
        if type(k) is not int or k<1 or type(max_iter) is not int or max_iter<1 or not np.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid clustering settings')
        self.k,self.max_iter,self.tolerance,self.seed=k,max_iter,tolerance,seed;self.centroids=None
    def predict(self,x):
        if self.centroids is None:raise ValueError('Fit before predicting')
        x=finite_matrix(x)
        if x.shape[1]!=self.centroids.shape[1]:raise ValueError('Feature count mismatch')
        # Bound the working distance matrix for pixel-scale inputs.
        return np.concatenate([((chunk[:,None,:]-self.centroids[None,:,:])**2).sum(axis=2).argmin(axis=1) for chunk in np.array_split(x,max(1,int(np.ceil(len(x)/4096))))])
    def fit(self,x):
        x=finite_matrix(x);unique=np.unique(x,axis=0)
        if len(unique)<self.k:raise ValueError('Need at least k distinct observations')
        rng=np.random.default_rng(self.seed);chosen=[int(rng.integers(len(unique)))]
        for _ in range(1,self.k):
            distance=((unique[:,None,:]-unique[np.asarray(chosen)][None,:,:])**2).sum(axis=2).min(axis=1)
            chosen.append(int(rng.choice(len(unique),p=distance/distance.sum())))
        self.centroids=unique[chosen].copy();self.inertia_history=[]
        for _ in range(self.max_iter):
            labels=self.predict(x);updated=self.centroids.copy()
            for i in range(self.k):
                if (labels==i).any():updated[i]=x[labels==i].mean(axis=0)
                # Retain an empty centroid: this avoids NaNs and cannot increase the objective.
            movement=float(np.max(np.linalg.norm(updated-self.centroids,axis=1)))
            self.centroids=updated;labels=self.predict(x);inertia=float(((x-self.centroids[labels])**2).sum())
            if not np.isfinite(inertia):raise ValueError('Clustering overflow')
            self.inertia_history.append(inertia)
            if movement<=self.tolerance:break
        self.labels_=labels;self.inertia_=inertia;return self

def quantize_pixels(rgb,k=8):
    array=np.asarray(rgb)
    if array.ndim!=3 or array.shape[2]!=3 or not array.shape[0] or not array.shape[1] or array.size>12_000_000 or not np.isfinite(array).all() or (array<0).any() or (array>255).any():raise ValueError('Expected finite RGB values in [0,255], under four million pixels')
    pixels=array.reshape(-1,3).astype(float)
    if len(pixels)>20000:
        subset=np.random.default_rng(42).choice(len(pixels),20000,replace=False);training=pixels[subset]
    else:training=pixels
    model=LloydKMeans(k=k).fit(training);prediction=model.predict(pixels)
    return np.clip(np.rint(model.centroids[prediction]),0,255).astype(np.uint8).reshape(array.shape),model
