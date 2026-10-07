import itertools
import numpy as np
import pandas as pd
import pytest
from python_basics import *
from experiment import tabular

def test_small_python_functions_cover_truth_tables_and_list_boundaries():
    for ketchup,mustard,onion in itertools.product([False,True],repeat=3):
        expected=(not onion,ketchup and mustard and onion,not any([ketchup,mustard,onion]),sum([ketchup,mustard])==1 and not onion)
        assert hotdog_toppings(ketchup,mustard,onion)==expected
    assert second_player([]) is None and second_player([['only']]) is None
    assert second_player([['first','second']])=='second'
    assert second_element_of_list([]) is None and second_element_of_list([1,2])==2
    assert not meal_served_two_days_in_a_row(['rice','bread','rice'])
    assert meal_served_two_days_in_a_row(['rice','rice'])
    assert not meal_served_two_days_in_a_row([])
    assert fashionably_late(['a','b','c','d'],'c') and not fashionably_late(['a','b','c','d'],'d')
    assert [sign(n) for n in [-1,0,1]]==[-1,0,1]
    assert total_number_of_candies(91,12,2)==0 and round_decimal(1.234)==1.23

def test_tabular_models_fit_only_training_rows_and_exclude_post_contact_values(monkeypatch):
    from sklearn.pipeline import Pipeline
    original=Pipeline.fit;seen=[]
    def recording_fit(self,x,y=None,**kwargs):
        if 'model' in self.named_steps:
            assert 'duration' not in x and 'ID' not in x
            seen.append((set(x.index),x['value'].mean()))
        return original(self,x,y,**kwargs)
    monkeypatch.setattr(Pipeline,'fit',recording_fit)
    frame=pd.DataFrame({'value':np.arange(120.),'category':np.where(np.arange(120)%2,'left','right'),
        'duration':np.arange(120)*100,'ID':np.arange(120),'label':np.where(np.arange(120)%2,'yes','no')})
    result=tabular(frame,'label',['duration','ID'])
    assert [result[k] for k in ['train','validation','test']]==[72,24,24]
    assert len(seen)==8 and len({frozenset(v[0]) for v in seen})==1 and len(seen[0][0])==72
    assert set(result['models'])=={'decision_tree','random_forest','knn'}
    assert all(0<=v['test']['macro_f1']<=1 for v in result['models'].values())

def test_conflicting_identical_predictors_never_reach_training():
    frame=pd.DataFrame({'value':np.arange(100.),'label':['a']*50+['b']*50})
    frame=pd.concat([frame,pd.DataFrame({'value':[0.,1000.,1000.],'label':['a','a','b']})],ignore_index=True)
    result=tabular(frame,'label')
    assert result['conflicting_rows_removed']==2 and result['duplicate_rows_removed']==1 and result['retained_rows']==100
    frame.loc[0,'label']=None
    with pytest.raises(ValueError):tabular(frame,'label')
