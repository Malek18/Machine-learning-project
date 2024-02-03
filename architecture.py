
#hi les filles 


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
"""from PIL import Image
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

def raw_image_to_representation(image_file, representation):
   
    img = Image.open(image_file)
   
    print("test")
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
    
raw_image_to_representation("C:/Users/malak/Downloads/aka70a.jpeg", 'HC')"""



from PIL import Image
import numpy as np

def raw_image_to_representation(image_file, representation):
    try:
        img = Image.open(image_file)
        print("Image ouverte avec succès.")
        
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
            raise ValueError("Représentation non définie.")
    except FileNotFoundError:
        print(f"Fichier '{image_file}' introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")

# Exemple d'appel de fonction
result = raw_image_to_representation("C:/Users/malak/Downloads/aka70a.jpeg", 'HC')
if result is not None:
    print("Résultat :", result)


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
""""
def load_transform_label_train_dataset(directory, representation):
    donneTab = []

    
    for root, dirs, files in os.walk(directory):
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
""""
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
"""def load_transform_test_dataset(directory, representation):
    return None

"""
"""
Learn a model (function) from a pre-computed representation of the dataset, using the algorithm 
and its hyper-parameters described in algo_dico
For example, algo_dico could be { algo: 'decision tree', max_depth: 5, min_samples_split: 3 } 
or { algo: 'multinomial naive bayes', force_alpha: True }
--
input = transformed labelled dataset, the used learning algo and its hyper-parameters (better a dico)
output =  a model fit with data
"""



"""from sklearn.naive_bayes import MultinomialNB

def learn_model_from_dataset(train_dataset, algo_dico):
    algo = algo_dico['algo'] 
    model = None
    
    if algo == 'multinomial naive bayes':
       
        force_alpha = algo_dico.get('force_alpha', False)
    
        model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None, force_alpha=force_alpha)
    else:
        raise ValueError("Algorithme d'apprentissage non pris en charge")
    
    
    X_train = [data['representation'] for data in train_dataset]
    y_train = [data['label'] for data in train_dataset]
    
    
    model.fit(X_train, y_train)
    
    return model
""""""
directory = "/amuhome/a21222384/Bureau/Data"
representation = 'HC'  
algo_dico = {'algo': 'multinomial naive bayes', 'force_alpha': False}
d=load_transform_label_train_dataset(directory,representation)
model = learn_model_from_dataset(d, algo_dico)
print("Modèle entraîné :", model)
"""
"""# Création d'un ensemble de données d'entraînement fictif
train_dataset = [
    {'representation': [1, 2, 3], 'label': 1},
    {'representation': [4, 5, 6], 'label': -1},
    {'representation': [7, 8, 9], 'label': 1},
    {'representation': [10, 11, 12], 'label': -1},
    # Ajoutez d'autres exemples ici
]

# Définition de l'algorithme et de ses hyperparamètres
algo_dico = {'algo': 'multinomial naive bayes', 'force_alpha': False}

# Apprentissage du modèle à partir de l'ensemble de données fictif
model = learn_model_from_dataset(train_dataset, algo_dico)

# Affichage du modèle entraîné
print("Modèle entraîné :", model)
"""
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
     # Si la prédiction est positive ou nulle, retourner +1, sinon -1
    if predicted_label >= 0:
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
    predictions = []
    for data in dataset:
        representation = data['representation']  # Extraction de la représentation de l'exemple
        predicted_label = model.predict([representation])[0]  # Prédiction de l'étiquette avec le modèle
        predictions.append({'name': data['name'], 'label': predicted_label})  # Stockage de la prédiction avec le nom de l'exemple
    return predictions

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
"""def write_predictions(directory, filename, predictions):
    return None

"""
""""
Estimates the accuracy of a previously learned model using train data, 
either through CV or mean hold-out, with k folds.
input = the train labelled data as previously structured, the type of model to be learned
(as in function learn_model_from_data), and the number of split to be used either 
in a hold-out or by cross-validation 
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random guess)"""
"""
def estimate_model_score(train_dataset, algo_dico, k):
    return None"""

    
