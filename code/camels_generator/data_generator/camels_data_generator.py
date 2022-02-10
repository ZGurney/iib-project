import abc

import numpy as np
import pandas as pd
import torch
import os
import pdb
import random
import time
import datetime
import matplotlib.pyplot as plt

from .task_preprocessing import *
from .utils import device


__all__ = ['HydroGenerator']

def _rand(val_range, *shape):
    lower, upper = val_range
    return random.sample(range(int(lower),int(upper)),*shape)
    #return lower + np.random.rand(*shape) * (upper - lower)

def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')

"""def date_to_int(row):
    return int(time.mktime(datetime.datetime(year=int(row['YR']), month=int(row['MNTH']), day=int(row['DY'])).timetuple())/86400)"""


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self

class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch.
            Defaults to 256.
        x_range (tuple[float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Must be at
            least 3. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Must be at
            least 3. Defaults to 50.
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 min_train_points = 10,
                 min_test_points = 10,
                 max_train_points=30,
                 max_test_points=30):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_batches = num_tasks // batch_size

        if self.num_batches * batch_size != num_tasks:
            raise ValueError(
                f"Number of tasks {num_tasks} must be a multiple of "
                f"the batch size {batch_size}."
            )

        self.x_range = x_range
        self.min_train_points = min_train_points
        self.min_test_points = min_test_points
        self.max_train_points = max_train_points
        self.max_test_points = max_test_points

    @abc.abstractmethod
    def sample(self,x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """
    
    # @abc.abstractmethod
    # def generate_task(self):
    #     """Generate a task.
    #     Returns:
    #         dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
    #             `y_context`, `x_target`, and `y_target; and other optional keys 'm', 'f'
    #     """

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)

class HydroGenerator(DataGenerator):
    """ Generate samples from hydrological data"""
    
    def __init__(self,
                channels_c = ['Tmax(C)', 'Tmin(C)'],
                channels_t = ['Tmax(C)', 'Tmin(C)'],
                channels_att = ['gauge_id'],
                #channels_t_val = ['OBS_RUN_log_n_mean'],
                context_mask = [1,1],
                target_mask = [1,1],
                extrapolate = False,
                timeslice = 61,
                dropout_rate = 0,
                concat_static_features = False,
                observe_at_target = False,
                device = device,
                dict_df = None,
                dict_att = None,
                att_noise = 0,
                **kw_args):     

        self.channels_c = channels_c
        self.channels_t = channels_t
        self.channels_att = channels_att
        #self.channels_t_val = channels_t_val
        self.context_mask = context_mask
        self.target_mask = target_mask
        self.extrapolate = extrapolate
        self.timeslice = timeslice
        self.dropout_rate = dropout_rate
        self.concat_static_features = concat_static_features
        self.observe_at_target = observe_at_target
        self.device = device
        self.dict_df = dict_df
        self.dict_att = dict_att
        self.att_noise = att_noise
        DataGenerator.__init__(self,**kw_args)
    
    def sample(self,x,df):
        return np.vstack(tuple(df[key][x] for key in self.channels_c)), np.vstack(tuple(df[key][x] for key in self.channels_t))#, np.vstack(tuple(df[key][x] for key in self.channels_t_val)) 
    
    def sample_att(self,basin_key):
        return np.vstack(tuple(self.dict_att[basin_key][k].values + np.random.normal(0,self.att_noise) for k in self.channels_att))

    def sample_date(self,x,df):
        return np.vstack(tuple(df[key][x] for key in ['Year','DOY']))

    def generate_task(self,index=0):
        task = {'x' : [],
                'y' : [],
                'x_context' : [],
                'y_context' : [],
                'x_target' : [],
                'y_target' : [],
                #'y_target_val' : [],
                #'y_att' : [],
                }
        
        # Determine number of test and train points.
        num_train_points = np.random.randint(self.min_train_points, self.max_train_points + 1)
        num_test_points = np.random.randint(self.min_test_points, self.max_test_points + 1)
        num_points = num_train_points + num_test_points
        
        # Generate a random integer for each element in the bacth 
        #randoms = np.random.randint(0,len(self.dataframe),self.batch_size)

        for i in range(self.batch_size):
        # Sample inputs and outputs.
            b = np.random.choice(list(self.dict_df.keys())) # sample basin from dictionary 
            df = self.dict_df[b] # select dataframe from dictionary 
            idx = np.random.randint(len(df)-self.timeslice) # random index 
            s_ind = df.index[idx] # choose random start date 
            e_ind = s_ind + datetime.timedelta(days=self.timeslice) # define end date based on timeslice
            df_clip = df[(df.index > s_ind) * (df.index < e_ind)] # clip dataframe based on start-end dates
            x_ind = (df_clip.sample(num_points).index - df_clip.index[0]).days.values # indices to be selected
            
            if self.extrapolate:
                x_ind = sorted(x_ind)

            #y, y_t, y_t_val = self.sample(x_ind,df_clip)
            y, y_t = self.sample(x_ind,df_clip)

            #y_att = self.sample_att(basin_key=b)

            x = np.divide(np.array(x_ind), self.timeslice)

            # Extrapolate
            if self.extrapolate:
                inds = np.arange(len(x))
            else:
                inds = np.random.permutation(len(x))                
            
            # Indices for train and test set 
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['x_context'].append(x[inds_train])
            task['x_target'].append(x[inds_test])

            #task['y_att'].append(y_att)

            y_aux, y_context_aux, y_target_aux = [], [], []
            
            for i in range(len(y)):
                y_aux.append(y[i][np.argsort(x)])
                y_context_aux.append(y[i][inds_train])
            
            for i in range(len(y_t)):
                y_target_aux.append(y_t[i][inds_test])
            
            task['y'].append(np.stack(y_aux,axis=1).tolist())
            task['y_context'].append(np.stack(y_context_aux,axis=1).tolist())
            task['y_target'].append(np.stack(y_target_aux,axis=1).tolist())

            #task['y'].append(y[0][np.argsort(x)])
            #task['y_context'].append(y[0][inds_train])
            #task['y_target'].append(y[0][inds_test])
            #task['y_target_val'].append(y_t_val[0][inds_test])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(self.device)
                for k, v in task.items()}

        #task['y_att'] = task['y_att'].permute([0,2,1])
        #task['y_att_context'] = task['y_att'] * torch.ones(task['x_context'].shape).to(self.device)
        #task['y_att_target'] = task['y_att'] * torch.ones(task['x_target'].shape).to(self.device)
        
        if self.concat_static_features:
            task['y_context'] = torch.cat([task['y_context'],task['y_att_context']],dim=2)
            task['y_target'] = torch.cat([task['y_target'],task['y_att_target']],dim=2)

        task = prep_task(task,
                            context_mask=self.context_mask,
                            target_mask=self.target_mask,
                            dropout_rate=self.dropout_rate,
                            embedding=True,
                            concat_static_features=self.concat_static_features,
                            observe_at_target=self.observe_at_target,
                            device=self.device)

        return task

    def epoch(self):
        """Construct a generator for an epoch.
        Returns:
            generator: Generator for an epoch.
        """

        def lazy_gen_batch():
            return self.generate_task()

        return (lazy_gen_batch() for _ in range(self.num_batches))

    def generate_test_task(self,year,basin):
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': [],
                #'y_target_val': [],
                #'y_att': [],
                #'yr_context': [],
                #'yr_target': [],
                #'doy_context': [],
                #'doy_target': [],
                }
        
        # Determine number of test and train points.
        num_train_points = np.random.randint(self.min_train_points, self.max_train_points + 1)
        num_test_points = np.random.randint(self.min_test_points, self.max_test_points + 1)
        num_points = num_train_points + num_test_points
        
        df = self.dict_df[basin] # select dataframe from dictionary based on basin id

        s_ind, e_ind = df.index[df['Year']==year][[0,-1]] # start and end indices for basin and year
        s_ind_b, e_ind_b = df.index[[0, -1]] # start and end indices for basin 
        
        if (s_ind - s_ind_b).days > self.timeslice: # check if start index can be moved back by timeslice
            s_ind = s_ind - pd.Timedelta(days=self.timeslice) # if so, move start index 
        elif (s_ind - s_ind_b).days < self.timeslice: 
            s_ind = s_ind_b # otherwise, set to basin's start index 
            
        df_clip = df[s_ind:e_ind] # clip dataframe

        #ids = self.dataframe['id'][(self.dataframe['hru08']==basin)&(self.dataframe['YR']==year)].unique()[0]
        ##df = self.dataframe[(self.dataframe['hru08']==basin)&(self.dataframe['YR']==year)]
        #df = self.dataframe[(self.dataframe['id']==ids) | (self.dataframe['id_lag']==ids)]
        
        self.batch_size = len(df_clip) - self.timeslice # re-define batch size to be the entire clipped dataframe minus the timescale 
        #hru08 = df['hru08'].unique()[0]

        for i in range(self.batch_size):
        # Sample inputs and outputs.

            #df_s = df.copy()
            #df_s.drop_duplicates(inplace=True)
            #df_s = df_s.reset_index(drop=True)

            s_ind = i
            e_ind = self.timeslice + i

            x_ind = np.arange(s_ind, e_ind)

            #y, y_t, y_t_val = self.sample(x_ind,df_clip)
            y, y_t = self.sample(x_ind,df_clip)

            #y_att = self.sample_att(basin_key=basin)

            #print("y_t_val : " , y_t_val)
            
            #x_date = self.sample_date(x_ind,df_clip)

            x = np.divide(np.array(x_ind) - s_ind, e_ind - s_ind)

            # Determine indices for train and test set.
            if self.extrapolate == False:
                inds = np.random.permutation(len(x))
            elif self.extrapolate == True:
                inds = np.arange(len(x))
            
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task['x'].append(sorted(x))
            task['x_context'].append(x[inds_train])
            task['x_target'].append(x[inds_test])
            
            #task['doy_context'].append(x_date[1][inds_train])
            #task['yr_context'].append(x_date[0][inds_train])
            #task['doy_target'].append(x_date[1][inds_test])
            #task['yr_target'].append(x_date[0][inds_test])

            #task['y_att'].append(y_att)

            y_aux, y_context_aux, y_target_aux = [], [], []
            
            for i in range(len(y)):
                y_aux.append(y[i][np.argsort(x)])
                y_context_aux.append(y[i][inds_train])
            
            for i in range(len(y_t)):
                y_target_aux.append(y_t[i][inds_test])
                #y_target_val_aux.append(y_t_val[i][inds_test]) ###
            
            task['y'].append(np.stack(y_aux,axis=1).tolist())
            task['y_context'].append(np.stack(y_context_aux,axis=1).tolist())
            task['y_target'].append(np.stack(y_target_aux,axis=1).tolist())

            #task['y'].append(y[0][np.argsort(x)])
            #task['y_context'].append(y[0][inds_train])
            #task['y_target'].append(y[0][inds_test])
            #task['y_target_val'].append(y_t_val[0][inds_test])
        

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(self.device)
                for k, v in task.items()}

        #task['y_att'] = task['y_att'].permute([0,2,1])
        #task['y_att_context'] = task['y_att'] * torch.ones(task['x_context'].shape).to(self.device)
        #task['y_att_target'] = task['y_att'] * torch.ones(task['x_target'].shape).to(self.device)
        
        if self.concat_static_features:
            task['y_context'] = torch.cat([task['y_context'],task['y_att_context']],dim=2)
            task['y_target'] = torch.cat([task['y_target'],task['y_att_target']],dim=2)

        task = prep_task(task,
                            context_mask=self.context_mask,
                            target_mask=self.target_mask,
                            dropout_rate=self.dropout_rate,
                            embedding=True,
                            concat_static_features=self.concat_static_features,
                            observe_at_target=self.observe_at_target,
                            device=self.device)

        return task
    