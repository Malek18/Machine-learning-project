
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cecile capponi, AMU
L3 Informatique, 2023/24
"""

"""
Computes a representation of an image from the (gif, png, jpg...) file
-> representation can be (to extend)
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels
other to be defined
--
input = an image (jpg, png, gif)
output = a new representation of the image
"""    
from PIL import Image
import  numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def raw_image_to_representation(image_file, representation):
 
  img = Image.open(image_file)
 
 
  if representation == 'HC':
     
      histogram = img.histogram()
      return histogram
  elif representation == 'PX':
     
      pixels = np.array(img)
      return pixels
  elif representation == 'GC':
     
      gray_img = img.convert('L')
      gray_matrix = np.array(gray_img)
      return gray_matrix
  else:
     
      raise ValueError("Undefined representation")
 

"""
Returns a relevant structure embedding train images described according to the
specified representation and associate each image (name or/and location) to its label.
-> Representation can be (to extend)
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels
other to be defined
--
input = where are the examples, which representation of the data must be produced ?
output = a relevant structure (to be discussed, see below) where all the images of the
directory have been transformed, named and labelled according to the directory they are
stored in (the structure lists all the images, each image is made up of 3 informations,
namely its name, its representation and its label)
This structure will later be used to learn a model (function learn_model_from_dataset)
-- uses function raw_image_to_representation
"""

def load_transform_label_train_dataset(directory, representation):
    donneTab = []

    for root, dirs, files in os.walk(directory):  # Déballez les trois éléments
        for file in files:
            image_file = os.path.join(root, file)
            image_representation = raw_image_to_representation(image_file, representation)
            
            if 'Mer' in root:
                label = 1
            elif 'Ailleurs' in root:
                label = -1
            else:
                raise ValueError("Répertoire non étiqueté")
                
            image_data = {
                'name': file,  
                'representation': image_representation,  
                'label': label  
            }
            
            donneTab.append(image_data)

    return donneTab


 

"""
Returns a relevant structure embedding test images described according to the
specified representation.
-> Representation can be (to extend)
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels
other to be defined
--
input = where are the data, which represenation of the data must be produced ?
output = a relevant structure, preferably the same chosen for function load_transform_label_train_data
-- uses function raw_image_to_representation
-- must be consistant with function load_transform_label_train_dataset
-- while be used later in the project
"""
def load_transform_test_dataset(directory, representation):
   test_dataset = []

   for root,dirs,files in os.walk(directory):
       for file in files:
           image_file = os.path.join(root, file)
           image_representation = raw_image_to_representation(image_file, representation)
           test_data = {
               'name': file,
               'representation': image_representation,
               'label':0
           }
           test_dataset.append(test_data)

   return test_dataset





"""
Learn a model (function) from a pre-computed representation of the dataset, using the algorithm
and its hyper-parameters described in algo_dico
For example, algo_dico could be { algo: 'decision tree', max_depth: 5, min_samples_split: 3 }
or { algo: 'multinomial naive bayes', force_alpha: True }
--
input = transformed labelled dataset, the used learning algo and its hyper-parameters (better a dico)
output =  a model fit with data
"""


def learn_model_from_dataset(train_dataset, algo_dico):
   algo = algo_dico['algo']
   model = None
 
   if algo == 'multinomial naive bayes':

       force_alpha = algo_dico.get('force_alpha', True)
       alpha = algo_dico.get('alpha', 1.0)
       model = MultinomialNB(alpha=alpha,force_alpha=force_alpha)
   elif algo == 'svm':
       kernel = algo_dico.get('kernel', 'rbf')
       C = algo_dico.get('C', 1.0)
       gamma = algo_dico.get('gamma', 'scale')
       model = SVC(kernel=kernel, C=C, gamma=gamma)
   elif algo == 'k_neighbors':
       n_neighbors = algo_dico.get('n_neighbors', 5)
       weights = algo_dico.get('weights', 'uniform')
       model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

   elif algo == 'decision_tree':
       max_depth = algo_dico.get('max_depth', 5)
       min_samples_split = algo_dico.get('min_samples_split', 3)
       model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
   else:
       raise ValueError("Algorithme d'apprentissage non pris en charge")

  
 
 
   X_train = [data['representation'] for data in train_dataset]
   y_train = [data['label'] for data in train_dataset]
   max_length = max(len(rep) for rep in X_train)
   X_train = [rep + [0] * (max_length - len(rep)) for rep in X_train]
   X_train = np.array(X_train)
   y_train = np.array(y_train)
   model.fit(X_train, y_train)

   return model



train_dataset = load_transform_label_train_dataset('/home/codespace/Machine-learning-project/Data','HC')


algo_dico = {
  'algo': 'multinomial naive bayes',
  'force_alpha': True  
}
model = learn_model_from_dataset(train_dataset, algo_dico)
if model is not None:
  print("Le modèle a été correctement entraîné.")
  print("Classes:", model.classes_)
  print("Probabilités a priori des classes:", model.class_log_prior_)
  print("Probabilités conditionnelles des caractéristiques:", model.feature_log_prob_)
else:
  print("Une erreur s'est produite lors de l'apprentissage du modèle.")

"""
Given one example (previously loaded with its name and representation),
computes its class according to a previously learned model.
--
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_dataset
"""


def predict_example_label(example, model):
  representation = example['representation']
  label = model.predict([representation])[0]
   
  if label >= 0:
      return 1
  else:
      return -1
  

"""Computes a structure that computes and stores the label of each example of the dataset,
using a previously learned model.
--
input = a structure embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each identified data (image) of the input dataset
"""
def predict_sample_label(dataset, model):
  labeled_dataset = []

  for data in dataset:
      name = data['name']
      label = predict_example_label(data, model)
      
      labeled_data = {
          'name': name,
          'label': label
      }
      
      labeled_dataset.append(labeled_data)

  return labeled_dataset


"""Save the predictions on dataset to a text file with syntax:

image_name <space> label (either -1 or 1)  
NO ACCENT  
In order to be perfect, the first lines of this file should indicate the details
of the learning methods used, with its hyper-parameters (in order to do so, the signature of
the function must be changed, as well as the signatures of some previous functions in order for
these details to be transmitted along the pipeline.
--
input = where to save the predictions, structure embedding the dataset
output =  OK if the file has been saved, not OK if not
"""
def write_predictions(directory, filename, predictions):
  try:
      with open(os.path.join(directory, filename), 'w') as file:
          file.write("Learning method: {}\n".format(predictions['learning_method']))
          file.write("Hyperparameters: {}\n\n".format(predictions['hyperparameters']))
          
          for prediction in predictions['predictions']:
              file.write("{} {}\n".format(prediction['name'], prediction['label']))
      
      return "OK"
  except Exception as e:
      print("Error:", e)
      return "Not OK"


"""
Estimates the accuracy of a previously learned model using train data,
either through CV or mean hold-out, with k folds.
input = the train labelled data as previously structured, the type of model to be learned
(as in function learn_model_from_data), and the number of split to be used either
in a hold-out or by cross-validation
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random guess)"""


def estimate_model_score(train_dataset, algo_dico, k):
  
  X_train = [data['representation'] for data in train_dataset]
  y_train = [data['label'] for data in train_dataset]
  
  model = learn_model_from_dataset(train_dataset, algo_dico)
  scores = cross_val_score(model, X_train, y_train, cv=k)
  mean_score = scores.mean()
  
  return mean_score 



# Charger les données de test
test_directory = '/home/codespace/Machine-learning-project/Test'
test_representation = 'HC'  # ou toute autre représentation souhaitée
test_dataset = load_transform_test_dataset(test_directory, test_representation)

# Faire des prédictions sur les données de test
predictions = predict_sample_label(test_dataset, model)

# Évaluer la qualité du modèle
# Par exemple, calculer l'exactitude
true_labels = [data['label'] for data in test_dataset]
predicted_labels = [data['label'] for data in predictions]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_labels, predicted_labels)
print("Exactitude du modèle:", accuracy)