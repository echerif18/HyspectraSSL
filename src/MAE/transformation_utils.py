import torch
import torch.nn as nn

class StaticTransformationLayer(nn.Module):
    """
    A static transformation layer that applies a given transformation function
    to the input tensor. This is useful for embedding a fixed (non-learnable)
    transformation into a neural network module.
    
    Args:
        transformation (callable): A function or module to apply on the input.
    """
    def __init__(self, transformation):
        super(StaticTransformationLayer, self).__init__()
        self.transformation = transformation

    def forward(self, x):
        """
        Forward pass: apply the static transformation to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Transformed tensor.
        """
        # Apply the provided transformation
        x_transformed = self.transformation(x)
        return x_transformed


class BoxCoxTransform(nn.Module):
    """
    Applies the Box-Cox transformation to each feature (channel) of the input tensor.
    The Box-Cox transformation is a power transformation that stabilizes variance and 
    makes data more normally distributed. Optionally, it can normalize the transformed 
    data using provided mean and standard deviation.
    
    Args:
        lambda_values (list or tensor): A list/tensor of lambda values for each feature.
        mean (optional, tensor or list): Mean values for normalization after transformation.
        std (optional, tensor or list): Standard deviation values for normalization after transformation.
    """
    def __init__(self, lambda_values, mean=None, std=None):
        super(BoxCoxTransform, self).__init__()
        self.lambda_values = lambda_values
        # Determine the computation device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Forward pass: apply the Box-Cox transformation to each feature of the input tensor.
        If normalization parameters (mean and std) are provided, the output is normalized.
        
        Args:
            x (torch.Tensor): Input tensor (assumed to be 2D: [batch_size, features]).
        
        Returns:
            torch.Tensor: Box-Cox transformed (and optionally normalized) tensor.
        """
        transformed = [] 
        
        # If mean and std are provided, ensure they are on the same device as x
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.clone().detach().to(x.device)
            self.std = self.std.clone().detach().to(x.device)
            
        # Process each feature separately with its corresponding lambda value
        for i, lambda_value in enumerate(self.lambda_values):
            # Ensure lambda_value is a tensor on the correct device
            lambda_value = lambda_value.clone().to(x.device)
            if lambda_value.eq(0):
                # When lambda is 0, use the logarithm transformation
                transformed.append(torch.log(x[:, i:i+1]))  # Assuming x is a 2D tensor
            else:
                # Add a small epsilon for numerical stability
                epsilon = 1e-6
                lambda_value += epsilon
                # Apply the Box-Cox transformation for lambda != 0
                transformed.append(((x[:, i:i+1] ** lambda_value) - 1) / lambda_value)

        # Concatenate the transformed features along the feature dimension
        if self.mean is not None and self.std is not None:
            # Normalize the concatenated result using provided mean and std
            return (torch.cat(transformed, dim=1) - self.mean) / (self.std + 1e-6)
        else:
            return torch.cat(transformed, dim=1).to(x.device)
    
    def inverse(self, y):
        """
        Inverse the Box-Cox transformation to recover the original data.
        If normalization was applied in the forward pass, it is reversed here.
        
        Args:
            y (torch.Tensor): Transformed (and possibly normalized) tensor.
        
        Returns:
            torch.Tensor: Tensor after applying the inverse Box-Cox transformation.
        """
        original = []
        
        # Ensure mean and std are on the same device as y if they exist
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.clone().detach().to(y.device)
            self.std = self.std.clone().detach().to(y.device)
        
        # Process each feature separately with its corresponding lambda value
        for i, lambda_value in enumerate(self.lambda_values):
            lambda_value = lambda_value.clone().to(y.device)
            if lambda_value.eq(0):
                if self.mean is not None and self.std is not None:
                    # Reverse normalization and apply the exponential (inverse of log)
                    ori = y[:, i:i+1] * (torch.tensor(self.std[i]) + 1e-6) + torch.tensor(self.mean[i])
                    original.append(torch.exp(ori))
                else:
                    original.append(torch.exp(y[:, i:i+1]))  # Assuming y is a 2D tensor
            else:
                # Add epsilon for numerical stability
                epsilon = 1e-6
                lambda_value += epsilon
                if self.mean is not None and self.std is not None:
                    # Reverse normalization then apply the inverse Box-Cox transformation for lambda != 0
                    ori = y[:, i:i+1] * (self.std[i] + 1e-6) + self.mean[i]
                    original.append(((lambda_value * ori) + 1) ** (1 / lambda_value))
                else:
                    original.append(((lambda_value * y[:, i:i+1]) + 1) ** (1 / lambda_value))
        
        # Concatenate the original features and move the result to the appropriate device
        return torch.cat(original, dim=1).to(y.device)
