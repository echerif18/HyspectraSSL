import os
import glob
from pickle import dump,load
from sklearn.preprocessing import PowerTransformer, StandardScaler

# from torchvision import transforms
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import math
import random
from scipy.signal import savgol_filter

######### Raw data ##########

def read_db(file, sp=False, encoding=None):
    db = pd.read_csv(file, encoding=encoding, low_memory=False)
    db.drop(['Unnamed: 0'], axis=1, inplace=True)
    if (sp):
        features = db.loc[:, "400":"2500"]
        labels = db.drop(features.columns, axis=1)
        return db, features, labels
    else:
        return db

### Apply savgol filter for a wavelength filter, 
def filter_segment(features_noWtab, order=1,der= False):
    #features_noWtab: Segment of the signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
    part1 = features_noWtab.copy()
    if (der):
        fr1 = savgol_filter(part1, 65, 1,deriv=1)
    else:
        fr1 = savgol_filter(part1, 65, order)
    fr1 = pd.DataFrame(data=fr1, columns=part1.columns)
    return fr1


def feature_preparation(features, inval = [1351,1431, 1801, 2051], frmax=2451, order=1,der= False):
    # features: The original reflectance signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
    features.columns = features.columns.astype('int')
    features[features<0] = 0   
    
    #####Substitute high values with the mean of neighbour values
    other = features.copy()
    other[other>1] = np.nan
    other = (other.fillna(method='ffill') + other.fillna(method='bfill'))/2
    other=other.interpolate(method='linear', axis=1).ffill().bfill()
    
    wt_ab = [i for i in range(inval[0],inval[1])]+[i for i in range(inval[2],inval[3])]+[i for i in range(2451,2501)] 

    features_Wtab = other.loc[:,wt_ab]
    features_noWtab=other.drop(wt_ab,axis=1)
    
    fr1 = filter_segment(features_noWtab.loc[:,:inval[0]-1], order = order, der = der)
    fr2 = filter_segment(features_noWtab.loc[:,inval[1]:inval[2]-1], order = order,der = der)
    fr3 = filter_segment(features_noWtab.loc[:,inval[3]:frmax], order = order,der = der)    
    
    
    inter = pd.concat([fr1,fr2,fr3], axis=1, join='inner')
    inter[inter<0]=0
    
    return inter

######## calculate sample weights from meta data #########
def samp_w(w_train, train_x):
    wstr = 100 - 100 * (w_train.loc[train_x.index, :].groupby(['dataset'])['numSamples'].count() /
                        w_train.loc[train_x.index, :].shape[0])
    samp_w_tr = np.array(w_train.loc[train_x.index, 'dataset'].map(dict(wstr)), dtype='float')
    return samp_w_tr


def data_prep_db(db_val_lb, ls_tr, weight_sample=False):
    val_x = feature_preparation(db_val_lb.loc[:, '400':'2500']).loc[:, 400:2450]
    val_x.index = db_val_lb.index
    
    val_y = db_val_lb[ls_tr]
    
    if(weight_sample):
        w_val = samp_w(db_val_lb.iloc[:,:8], db_val_lb)
        return val_x, val_y, w_val
    else:
        return val_x, val_y

def balanceData(db_train, w_train, Traits, random_state=300,percentage=1):
        ### The maximum number of samples within a dataset ##
        mx = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset').numSamples.count().max().max()*percentage
        fill = pd.concat([w_train, db_train], axis=1).groupby('dataset').sample(n=int(mx),random_state = random_state,replace=True)#.reset_index(drop=True)
        return fill


def save_scaler(train_y, save=False, dir_n=None, k=None, standardize=False, scale=False):
    from sklearn.preprocessing import FunctionTransformer, PowerTransformer
    
    if(scale):
        scaler = PowerTransformer(method='box-cox', standardize=standardize).fit(np.array(train_y)) # method='yeo-johnson' box-cox
    else:
        identity = FunctionTransformer(None)
        scaler = identity.fit(np.array(train_y))
    if save:
        if not os.path.exists(dir_n):
            os.mkdir(dir_n)
        dump(scaler, open(dir_n + '/scaler_{}.pkl'.format(k), 'wb')) 
    return scaler



### Datasets ####
def infinite_iter(train_dataset_loader, unlabeled_dataset_loader):
    data_loader_itr = iter(train_dataset_loader)
    data_loader_un_itr = iter(unlabeled_dataset_loader)
    
    while True:
        try:
            # Attempt to fetch the next batch from the unlabeled dataset iterator
            unlabeled_examples = next(data_loader_un_itr)
        except StopIteration:
            # If the unlabeled dataset iterator is exhausted, stop the infinite loop
            # print("Unlabeled dataset fully consumed. Stopping.")
            break
        try:
            # Attempt to fetch the next batch from the labeled dataset iterator
            labeled_examples, labels, _ = next(data_loader_itr)
        except StopIteration:
            # If the labeled dataset iterator is exhausted, reset it
            data_loader_itr = iter(train_dataset_loader)
            labeled_examples, labels, _ = next(data_loader_itr)

        # Create a tensor filled with NaN values for unlabeled data labels
        shape = (len(unlabeled_examples), labels.shape[1])
        nan_tensor = torch.full(shape, float('nan'))
        
        # Concatenate labeled and unlabeled data and labels
        samples = torch.cat([labeled_examples, unlabeled_examples])
        samples_lb = torch.cat([labels, nan_tensor])
        
        # Yield the combined batch
        yield samples, samples_lb


class SpectraDataset(Dataset):
    def __init__(self, X_train, y_train=None, meta_train=None, augmentation=False, aug_prob=0.5, betashift=0.01, slopeshift=0.01, multishift=0.1):
        """
        Args:
            X_train: Input features (spectra).
            y_train: Labels (None if unlabeled).
            meta_train: Metadata (optional).
            augmentation: Whether to apply augmentation.
            aug_prob: Probability of applying augmentation per sample.
            betashift, slopeshift, multishift: Parameters for shift augmentation.
        """
        self.X_train = np.array(X_train)  # Ensure these are NumPy arrays
        self.y_train = None if y_train is None else np.array(y_train)
        self.meta_train = None if meta_train is None else np.array(meta_train.dataset)
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.betashift = betashift  # Reduced parameter for minimal shift
        self.slopeshift = slopeshift
        self.multishift = multishift

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        # Retrieve the corresponding spectra
        x = torch.tensor(self.X_train[idx], dtype=torch.float32)
        
        # Optionally retrieve the label if available
        y = None if self.y_train is None else torch.tensor(self.y_train[idx], dtype=torch.float32)
        
        # Optionally retrieve metadata if available
        meta = None if self.meta_train is None else self.meta_train[idx]
        
        # Optionally apply augmentation with batch-level and sample-level randomness
        if self.augmentation and random.random() < self.aug_prob:
            x = self._apply_augmentation(x)

        # If the dataset is unlabeled, return only the spectra (no labels or metadata)
        if y is None:
            return x  # Unlabeled data, return only spectra
        
        # If labeled, return spectra, labels, and metadata (if available)
        return x, y, meta

    def _apply_augmentation(self, x_tensor):
        """
        Apply one of the augmentation methods to the input spectra.
        We randomly select an augmentation method at batch-level
        and apply it with sample-level probability.
        """
        # Define the augmentation methods to choose from
        augmentation_methods = [self._add_noise, self._shift]
        
        # Randomly pick an augmentation method for the current batch
        aug_method = random.choice(augmentation_methods)
        
        # Apply the chosen augmentation method
        return aug_method(x_tensor)

    def _add_noise(self, x_tensor, std=0.01):
        """Add Gaussian noise to the input tensor."""
        noise = torch.randn_like(x_tensor) * std
        return x_tensor + noise

    def _shift(self, x_tensor):
        """Apply a custom shift to the input tensor based on provided parameters, ensuring positivity."""
        # Calculate the standard deviation of the spectra
        std = torch.std(x_tensor)  # Use the std of this particular sample

        # Generate random shift parameters with a very small beta and slope
        beta = (torch.rand(1) * 2 * self.betashift - self.betashift) * std  # Ensure the shift is minimal
        slope = (torch.rand(1) * 2 * self.slopeshift - self.slopeshift + 1)  # Small variation around 1

        # Calculate axis for shifting based on the size of the input tensor
        axis = torch.arange(x_tensor.shape[0], dtype=torch.float32) / float(x_tensor.shape[0])
        
        # Calculate the offset based on beta and slope
        offset = (slope * axis + beta - axis - slope / 2.0 + 0.5)
        multi = (torch.rand(1) * 2 * self.multishift - self.multishift + 1)

        # Apply the shift by multiplying and adding the offset
        augmented_x = multi * x_tensor + offset * std
        
        # Ensure the augmented signal remains positive (clamp to a minimum of 0)
        augmented_x = torch.clamp(augmented_x, min=0)

        return augmented_x



# ############### unlabeled from multzi csv files ##
class MultiFileAugmentedCSVDataset(Dataset):
    def __init__(self, file_paths, chunk_size=1000, augmentation=False, aug_prob=0.,
                 betashift=0.01, slopeshift=0.01, multishift=0.1, transform=None, scale=False):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.betashift = betashift
        self.slopeshift = slopeshift
        self.multishift = multishift
        self.transform = transform
        self.current_chunk = None
        self.current_index = 0
        self.file_index = 0
        self.chunk_iter = None
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.load_next_file()  # Initialize with the first file
        self.scale = scale
        
        if(self.scale):
            self.fit_scaler()      # Fit scaler on the data

    def fit_scaler(self):
        # Load the full dataset or chunks to compute scaler statistics
        data = []
        for file_path in self.file_paths:
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)
            for chunk in chunk_iter:
                spectra = chunk.drop(['Unnamed: 0'], axis=1).values  # Assuming all columns are part of the spectra
                data.append(spectra)
        full_data = np.vstack(data)
        self.scaler.fit(full_data) #[:,:-1] if needed!!! # Fit scaler on all spectra data

    def scale_data(self, spectra):
        return self.scaler.transform(spectra)

    def __len__(self):
        total_rows = 0
        for file_path in self.file_paths:
            total_rows += sum(1 for _ in open(file_path)) - 1  # Exclude header
        return total_rows

    def reset(self):
        """Reset file and chunk iterators to start from the beginning."""
        self.current_chunk = None
        self.current_index = 0
        self.file_index = 0
        self.chunk_iter = None
        self.load_next_file()

    def load_next_file(self):
        if self.file_index < len(self.file_paths):
            self.chunk_iter = pd.read_csv(self.file_paths[self.file_index], chunksize=self.chunk_size, low_memory=False)
            self.file_index += 1
        else:
            self.file_index = 0
            self.load_next_file()

    def load_next_chunk(self):
        if self.chunk_iter is None:
            return False

        try:
            self.current_chunk = next(self.chunk_iter)

            # Prepare the chunk
            if 'Unnamed: 0' in self.current_chunk.columns:
                self.current_chunk = self.current_chunk.drop(['Unnamed: 0'], axis=1).reset_index(drop=True)

            spectra = self.current_chunk.values
            # spectra = spectra[:,:-1]
            
            if(self.scale):
                spectra = self.scale_data(spectra)  # Apply scaling here
            self.current_chunk = pd.DataFrame(spectra)

            self.current_index = 0

            if self.transform:
                self.current_chunk = self.current_chunk.apply(self.transform, axis=1)

            return True
        except StopIteration:
            self.load_next_file()
            return self.load_next_chunk()

    def __getitem__(self, idx):
        while self.current_chunk is None or self.current_index >= len(self.current_chunk):
            if not self.load_next_chunk():
                self.reset()
    
        row = self.current_chunk.iloc[self.current_index]
        self.current_index += 1
    
        x = torch.tensor(row.values, dtype=torch.float32)
    
        if self.augmentation and self.aug_prob > 0:
            rand_val = random.random()
            if rand_val < self.aug_prob:
                x = self._apply_augmentation(x)
    
        return x

    def _apply_augmentation(self, x_tensor):
        augmentation_methods = [self._add_noise]  # Add other augmentation methods if needed
        aug_method = random.choice(augmentation_methods)
        return aug_method(x_tensor)

    def _add_noise(self, x_tensor, std=0.01):
        noise = torch.randn_like(x_tensor) * std
        return x_tensor + noise
    


################ Splitting of data set into training units : splits ####
def split_csvs_with_proportions_sequential(input_folder, output_folder, num_splits=20, chunk_size=10000):
    """
    Split multiple CSV files into specified number of splits using proportions, processing one dataset at a time.

    Parameters:
        input_folder (str): Path to folder containing input CSV files.
        output_folder (str): Path to folder for saving output CSV files.
        num_splits (int): Number of output files (splits) to create.
        chunk_size (int): Number of rows to process at a time.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Step 1: Calculate total rows and proportions
    total_rows = 0
    file_row_counts = {}
    for file in glob.glob(os.path.join(input_folder, "*.csv")):
        row_count = sum(1 for _ in open(file)) - 1  # Exclude header
        file_row_counts[file] = row_count
        total_rows += row_count

    print(f"Total rows: {total_rows}")
    print(f"File row counts: {file_row_counts}")

    # Calculate target rows per split
    base_split_size = total_rows // num_splits
    remainder = total_rows % num_splits
    split_sizes = [base_split_size + (1 if i < remainder else 0) for i in range(num_splits)]

    print(f"Split sizes (with remainder distributed): {split_sizes}")

    # Step 2: Prepare split files
    split_files = [os.path.join(output_folder, f"split_{i + 1}.csv") for i in range(num_splits)]
    split_counters = [0] * num_splits
    headers_written = [False] * num_splits

    # Step 3: Process each file sequentially
    for file, file_row_count in file_row_counts.items():
        print(f"Processing file: {file}")

        # Calculate the proportion of rows for this file in each split
        file_proportions = [math.floor((file_row_count / total_rows) * size) for size in split_sizes]
        extra_rows = file_row_count - sum(file_proportions)

        # Distribute remaining rows due to rounding
        for i in range(extra_rows):
            file_proportions[i % num_splits] += 1

        print(f"Rows allocated to each split for {file}: {file_proportions}")

        # Distribute rows across splits
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            shuffled_chunk = chunk.sample(frac=1, random_state=42)  # Shuffle the chunk
            chunk_idx = 0

            for i, rows_to_take in enumerate(file_proportions):
                if rows_to_take == 0:
                    continue

                rows_available = len(shuffled_chunk) - chunk_idx
                rows_to_write = min(rows_available, rows_to_take)

                if rows_to_write > 0:
                    buffer = shuffled_chunk.iloc[chunk_idx:chunk_idx + rows_to_write]
                    with open(split_files[i], "a") as f:
                        buffer.to_csv(f, index=False, header=not headers_written[i])
                    headers_written[i] = True
                    split_counters[i] += len(buffer)
                    chunk_idx += rows_to_write
                    file_proportions[i] -= rows_to_write

                if chunk_idx >= len(shuffled_chunk):
                    break

    # Final output
    print("Splitting complete!")
    for i, split_file in enumerate(split_files):
        print(f"Split {i + 1}: {split_counters[i]} rows written to {split_file}")



def split_parquets_with_proportions_sequential(input_folder, output_folder, num_splits=20, chunk_size=10000):
    """
    Split multiple Parquet files into specified number of splits using proportions, processing one dataset at a time.

    Parameters:
        input_folder (str): Path to folder containing input Parquet files.
        output_folder (str): Path to folder for saving output Parquet files.
        num_splits (int): Number of output files (splits) to create.
        chunk_size (int): Number of rows to process at a time.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Step 1: Calculate total rows and proportions
    total_rows = 0
    file_row_counts = {}
    for file in glob.glob(os.path.join(input_folder, "*.parquet")):
        row_count = len(pd.read_parquet(file))
        file_row_counts[file] = row_count
        total_rows += row_count

    print(f"Total rows: {total_rows}")
    print(f"File row counts: {file_row_counts}")

    # Calculate target rows per split
    base_split_size = total_rows // num_splits
    remainder = total_rows % num_splits
    split_sizes = [base_split_size + (1 if i < remainder else 0) for i in range(num_splits)]

    print(f"Split sizes (with remainder distributed): {split_sizes}")

    # Step 2: Prepare split file buffers
    split_data = [[] for _ in range(num_splits)]
    split_counters = [0] * num_splits

    # Step 3: Process each file sequentially
    for file, file_row_count in file_row_counts.items():
        print(f"Processing file: {file}")

        # Calculate the proportion of rows for this file in each split
        file_proportions = [math.floor((file_row_count / total_rows) * size) for size in split_sizes]
        extra_rows = file_row_count - sum(file_proportions)

        # Distribute remaining rows due to rounding
        for i in range(extra_rows):
            file_proportions[i % num_splits] += 1

        print(f"Rows allocated to each split for {file}: {file_proportions}")

        # Distribute rows across splits
        df = pd.read_parquet(file)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        chunk_idx = 0
        for i, rows_to_take in enumerate(file_proportions):
            if rows_to_take == 0:
                continue

            rows_available = len(df) - chunk_idx
            rows_to_write = min(rows_available, rows_to_take)

            if rows_to_write > 0:
                buffer = df.iloc[chunk_idx:chunk_idx + rows_to_write]
                split_data[i].append(buffer)
                split_counters[i] += len(buffer)
                chunk_idx += rows_to_write
                file_proportions[i] -= rows_to_write

            if chunk_idx >= len(df):
                break

    # Step 4: Save splits to disk
    for i, data_chunks in enumerate(split_data):
        if data_chunks:
            full_df = pd.concat(data_chunks, ignore_index=True)
            output_path = os.path.join(output_folder, f"split_{i + 1}.parquet")
            full_df.to_parquet(output_path, index=False)
            print(f"Split {i + 1}: {split_counters[i]} rows written to {output_path}")

    print("Splitting complete!")



def sliding_custom_cv(df, seed=None):
    dataset_ids = sorted(df['dataset'].unique())
    
    if seed is not None:
        random.seed(seed)

    test_sets = []

    # First fold is explicitly defined
    test_sets.append([1, 2, 4, 5, 6])

    # Start building from index 6 (i.e., after dataset 6)
    current = 7
    max_id = max(dataset_ids)

    while current + 3 <= max_id:
        # Collect the next 4 datasets
        chunk = [current, current + 1, current + 2, current + 3]

        # Pick either 2 or 3
        choice = random.choice([2, 3])
        test_sets.append([choice] + chunk)

        # Move the starting index by 4 each time
        current += 4

    # Now yield each fold
    for test_ids in test_sets:
        train_ids = [i for i in dataset_ids if i not in test_ids]

        # Ensure 2 and 3 are not both in training
        if 2 in train_ids and 3 in train_ids:
            continue

        df_train = df[df['dataset'].isin(train_ids)].copy()
        df_test = df[df['dataset'].isin(test_ids)].copy()

        yield df_train, df_test, test_ids



def sliding_custom_dataset(df, seed=None):
    dataset_ids = sorted(df['dataset'].unique())

    if seed is not None:
        random.seed(seed)

    # One dataset as test each time
    for test_id in dataset_ids:
        test_ids = [test_id]
        train_ids = [i for i in dataset_ids if i != test_id]

        df_train = df[df['dataset'].isin(train_ids)].copy()
        df_test = df[df['dataset'].isin(test_ids)].copy()

        yield df_train, df_test, test_ids