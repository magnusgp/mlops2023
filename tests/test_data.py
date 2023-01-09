import torch
import numpy as np
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists('data/processed/trainset.pt'), reason="Training files not found")
@pytest.mark.skipif(not os.path.exists('data/processed/testset.pt'), reason="Training files not found")
def test_data():
    # load data from data/processed folder with pytorch
    trainset = torch.load('data/processed/trainset.pt')
    testset = torch.load('data/processed/testset.pt')
    # assert len(dataset) == N_train for training and N_test for test
    N_train = 25000
    assert len(trainset) == N_train, "Train dataset did not have the correct number of samples"
    N_test = 5000
    assert len(testset) == N_test, "Test dataset did not have the correct number of samples"
    #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    for i in range(len(trainset)):
        assert trainset[i][0].shape == torch.Size([28,28]), "Trainset datapoint did not have the correct shape"
        
    for i in range(len(testset)):
        assert testset[i][0].shape == torch.Size([28,28]), "Testset datapoint did not have the correct shape"
        
    #assert that all 10 labels are present in both datasets
    train_labels = [trainset[i][1].item() for i in range(len(trainset))]
    test_labels = [testset[i][1].item() for i in range(len(testset))]
    
    assert len(set(train_labels)) == 10, "Trainset did not have all 10 labels"
    assert len(set(test_labels)) == 10, "Testset did not have all 10 labels"
    
    assert len(trainset) == N_train, "Dataset did not have the correct number of samples"

