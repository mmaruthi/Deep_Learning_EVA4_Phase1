from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


# # class for Alubmentation Transforms # # 
class AlbumentationTransforms:
      def __init__(self,normalize=False, mean=None, stdev=None):
            
          self.normalize = normalize
          self.mean      = mean 
          self.stdev     = stdev 
      
      
      def test_transforms(self):
          transforms_list = [AP.ToTensor()]
          if (self.normalize):
             transforms_list.append(A.Normalize(self.mean,self.stdev))
          return A.Compose(transforms_list)
          
      # Define a method for train data . It can have multiple transformations other than changing to tensor and normalizing
      # so create your lists accordingly . One for before converting to tensor and normalization 
      # one for after converting it to tensor and normalization
      
      
      def train_transforms(self , before_norm=None, after_norm=None):
          if before_norm:
             transforms_list = before_norm 
             transforms_list.append(AP.ToTensor())
          else:
             transforms_list = [AP.ToTensor()]
             
          if (self.normalize):
             transforms_list.append(A.Normalize(self.mean,self.stdev))
           
          if after_norm:
             transforms_list.extend(after_norm)
           
          return A.Compose(transforms_list)

      
      

# # class for Transformations ## 
class Transforms_custom:
      def __init__(self,normalize=False, mean=None, stdev=None):
      
          self.normalize = normalize
          self.mean      = mean      ## Make sure you pass the meand and stdev whenever normalization is set to true 
          self.stdev     = stdev
      
      
      # Define a method for test data set as it does not need extra transformations.
      # We just need to normalize with mean and stdev
      
      def test_transforms(self):
          transforms_list = [transforms.ToTensor()]
          if (self.normalize):
             transforms_list.append(transforms.Normalize(self.mean,self.stdev))
          return transforms.Compose(transforms_list)
          
      # Define a method for train data . It can have multiple transformations other than changing to tensor and normalizing
      # so create your lists accordingly . One for before converting to tensor and normalization 
      # one for after converting it to tensor and normalization
      
      
      def train_transforms(self , before_norm=None, after_norm=None):
          if before_norm:
             transforms_list = before_norm 
             transforms_list.append(transforms.ToTensor())
          else:
             transforms_list = [transforms.ToTensor()]
             
          if (self.normalize):
             transforms_list.append(transforms.Normalize(self.mean,self.stdev))
           
          if after_norm:
             transforms_list.extend(after_norm)
           
          return transforms.Compose(transforms_list)
