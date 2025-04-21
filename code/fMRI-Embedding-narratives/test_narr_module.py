import torch
from narr_module import CT_Narrative_Data_Module  # Adjust this if the file name or path is different

def test_data_module():
    # Set parameters
    task = "pieman"
    task_tr = 300  # Total number of TRs
    batch_size = 16
    frames_per = 10
    train_val_test = (0.8, 0.1, 0.1)
    num_workers = 1

    # Initialize data module
    data_module = CT_Narrative_Data_Module(
        task=task,
        task_tr=task_tr,
        batch_size=batch_size,
        num_workers=num_workers,
        train_val_test=train_val_test,
        frames_per=frames_per
    )
    
    # Get DataLoaders
    train_loader, val_loader, test_loader = data_module._setup_dataloaders()
    
    # Define expected sizes based on task_tr and split ratios
    expected_train_len = int(task_tr * train_val_test[0] // frames_per)
    expected_val_len = int(task_tr * train_val_test[1] // frames_per)
    expected_test_len = int(task_tr * train_val_test[2] // frames_per)

    # Check Train DataLoader
    train_batches = 0
    for data, label in train_loader:
        assert data.shape == (batch_size, frames_per, data.shape[2]), \
            f"Train batch shape {data.shape} does not match expected {(batch_size, frames_per, data.shape[2])}"
        train_batches += 1
    assert train_batches * batch_size == expected_train_len, \
        f"Train DataLoader size {train_batches * batch_size} does not match expected {expected_train_len}"

    # Check Validation DataLoader
    val_batches = 0
    for data, label in val_loader:
        assert data.shape == (batch_size, frames_per, data.shape[2]), \
            f"Validation batch shape {data.shape} does not match expected {(batch_size, frames_per, data.shape[2])}"
        val_batches += 1
    assert val_batches * batch_size == expected_val_len, \
        f"Validation DataLoader size {val_batches * batch_size} does not match expected {expected_val_len}"

    # Check Test DataLoader
    test_batches = 0
    for data, label in test_loader:
        assert data.shape[1:] == (frames_per, data.shape[2]), \
            f"Test batch shape {data.shape} does not match expected {(1, frames_per, data.shape[2])}"
        test_batches += 1
    assert test_batches == expected_test_len, \
        f"Test DataLoader size {test_batches} does not match expected {expected_test_len}"

    print("All tests passed!")

# Run the test function
test_data_module()
