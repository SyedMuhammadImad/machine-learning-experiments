import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from numerical import *

def test_bernoulli_counts_and_unseen_binary_likelihood_match_library():
    x=np.array([[0,1,0],[0,1,1],[1,0,1],[1,0,0],[1,1,1]])
    y=np.array([-3,-3,50,50,50]);probe=np.array([[0,0,0],[1,1,1],[1,0,0]])
    manual=BernoulliClassifier(alpha=.5).fit(x,y);reference=BernoulliNB(alpha=.5,binarize=None).fit(x,y)
    np.testing.assert_allclose(manual.joint_log_likelihood(probe),reference._joint_log_likelihood(probe),atol=1e-12)
    np.testing.assert_array_equal(manual.predict_batch(probe),reference.predict(probe))
    assert manual.predict(probe[0])==reference.predict(probe)[0]

def test_bernoulli_long_feature_vector_stays_finite_and_labels_decode():
    x=np.vstack([np.zeros(10000),np.ones(10000)]);y=['cold','warm']
    model=BernoulliClassifier().fit(x,y)
    assert np.isfinite(model.joint_log_likelihood(x)).all();np.testing.assert_array_equal(model.predict_batch(x),y)
    with pytest.raises(ValueError):model.predict_batch(np.full((2,10000),2))
    with pytest.raises(ValueError):BernoulliClassifier(alpha=0)
    with pytest.raises(ValueError):BernoulliClassifier().predict(np.ones(2))
    with pytest.raises(ValueError):BernoulliClassifier().fit(x,[0,0])

def test_centered_regression_recovers_line_and_matches_independent_lstsq():
    rng=np.random.default_rng(7);x=rng.normal(size=100);y=3*x-2+rng.normal(0,.01,100)
    slope,intercept=ordinary_least_squares(x,y);expected=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]
    np.testing.assert_allclose([slope,intercept],expected,atol=1e-12)
    slope,intercept=ordinary_least_squares(np.arange(10.)+1e9,2*(np.arange(10.)+1e9)+5)
    assert slope==2 and intercept==5
    with pytest.raises(ValueError):ordinary_least_squares([1,1],[2,3])
    with pytest.raises(ValueError):ordinary_least_squares([1,2],[2,np.inf])

def test_lloyd_objective_decreases_centroids_match_partition_and_library():
    rng=np.random.default_rng(9);x=np.r_[rng.normal((-4,-4),.1,(50,2)),rng.normal((4,4),.1,(50,2)),rng.normal((-4,4),.1,(50,2))]
    model=LloydKMeans(k=3).fit(x)
    assert np.all(np.diff(model.inertia_history)<=1e-10)
    for i in range(3):np.testing.assert_allclose(model.centroids[i],x[model.labels_==i].mean(axis=0))
    reference=KMeans(3,random_state=42,n_init=10).fit(x)
    assert abs(model.inertia_-reference.inertia_)<1e-9
    np.testing.assert_array_equal(model.predict(x),LloydKMeans(k=3).fit(x).predict(x))
    with pytest.raises(ValueError):LloydKMeans(k=4).fit(np.ones((5,2)))

def test_pixel_quantization_preserves_shape_bounds_and_palette_count():
    # Synthetic colors are an implementation fixture, not an image-quality measurement.
    pixels=np.array([[[255,0,0],[254,1,0]],[[0,255,0],[0,254,1]]],dtype=np.uint8)
    output,model=quantize_pixels(pixels,2)
    assert output.shape==pixels.shape and output.dtype==np.uint8 and len(np.unique(output.reshape(-1,3),axis=0))==2
    assert np.mean((pixels.astype(float)-output.astype(float))**2)<2
    with pytest.raises(ValueError):quantize_pixels(np.zeros((2,2,4)),2)
