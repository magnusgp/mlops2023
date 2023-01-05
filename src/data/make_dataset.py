# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
from torchvision import transforms

import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def makedata(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        Input only the path to the raw data
    """
    # load the first part of the corrupted mnist dataset
    Xtrain = np.load('{}/train_0.npz'.format(input_filepath))['images']
    Ytrain = np.load('{}/train_0.npz'.format(input_filepath))['labels']
    
    # loop over the corrupted mnist dataset
    for i in range(1,5):
        Xtrain = np.concatenate((Xtrain, np.load('{}/train_{}.npz'.format(input_filepath, i))['images']), axis=0)
        Ytrain = np.concatenate((Ytrain, np.load('{}/train_{}.npz'.format(input_filepath, i))['labels']), axis=0)
    Xtest = np.load('{}/test.npz'.format(input_filepath))['images']
    Ytest = np.load('{}/test.npz'.format(input_filepath))['labels']
    
    # Define a transform to normalize the data with mean 0 and standard deviation 1
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])
    
    # normalize data
    Xtrain = transform(Xtrain)
    Ytrain = transform(Ytrain)
    Xtest = transform(Xtest)
    Ytest = transform(Ytest)
    """    
    
    # combine data and labels
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain).float(), torch.from_numpy(Ytrain).long())
    testset = torch.utils.data.TensorDataset(torch.from_numpy(Xtest).float(), torch.from_numpy(Ytest).long())
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # save trainset and testset into the processed folder (output_filepath)
    torch.save(trainset, '{}/trainset.pt'.format(output_filepath))
    torch.save(testset, '{}/testset.pt'.format(output_filepath))
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    makedata()
