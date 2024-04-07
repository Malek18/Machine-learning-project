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
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import cv2
from skimage import feature
from skimage.color import rgb2gray
from scipy import fft
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


def raw_image_to_representation(image_file, representations):
    img = Image.open(image_file)
    img = img.resize((100, 100))
    img = img.convert('RGB')
    img_gray = img.convert('L')

    combined_representation = []

    for representation in representations:
        if representation == 'HC':
            histogram = np.array(img.histogram())
            combined_representation.extend(histogram)
        elif representation == 'PX':
            pixels = np.array(img)
            combined_representation.extend(pixels.reshape(-1))
        elif representation == 'GC':
            gray_matrix = np.array(img_gray)
            combined_representation.extend(gray_matrix.flatten())
        elif representation == 'FOURIER':
            img_gray = rgb2gray(np.array(img))
            fft_image = np.abs(fft.fftshift(fft.fft2(img_gray)))
            combined_representation.extend(fft_image.flatten())
        elif representation == 'HOG':
            hog_feature = feature.hog(np.array(img_gray), pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                      visualize=False)
            combined_representation.extend(hog_feature)
        elif representation == 'SHAPE_MOMENTS':
            moments = cv2.moments(np.array(img_gray))
            hu_moments = cv2.HuMoments(moments).flatten()
            combined_representation.extend(hu_moments)
        elif representation == 'WAVELET':
            coeffs = pywt.dwt2(np.array(img_gray), 'haar')
            cA, (cH, cV, cD) = coeffs
            combined_representation.extend(cA.flatten())
            combined_representation.extend(cH.flatten())
            combined_representation.extend(cV.flatten())
            combined_representation.extend(cD.flatten())
        elif representation == 'HuMoments':
            moments = cv2.moments(np.array(img_gray))
            hu_moments = cv2.HuMoments(moments).flatten()
            combined_representation.extend(hu_moments)
        else:
            raise ValueError("Undefined representation")

    return np.array(combined_representation)


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

    for root, dirs, files in os.walk(directory):
        for file in files:
            image_file = os.path.join(root, file)
            image_representation = raw_image_to_representation(image_file, representation)

            # Vérifie le premier caractère du nom du fichier pour déterminer l'étiquette
            if file[0] == '0':
                label = -1  # Ailleurs
            elif file[0] == '1':
                label = 1  # Mer
            else:
                raise ValueError("Étiquette invalide pour le fichier :", file)

            test_data = {
                'name': file,
                'representation': image_representation,
                'label': label
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
        model = MultinomialNB(alpha=alpha, force_alpha=force_alpha)
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
    elif algo == 'random_forest':
        n_estimators = algo_dico.get('n_estimators', 100)
        max_depth = algo_dico.get('max_depth', None)
        min_samples_split = algo_dico.get('min_samples_split', 2)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split)
    elif algo == 'logistic_regression':
        penalty = algo_dico.get('penalty', 'l2')
        C = algo_dico.get('C', 1.0)
        model = LogisticRegression(penalty=penalty, C=C)
    elif algo == 'mlp':
        hidden_layer_sizes = algo_dico.get('hidden_layer_sizes', (100,))
        activation = algo_dico.get('activation', 'relu')
        solver = algo_dico.get('solver', 'adam')
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)

    else:
        raise ValueError("Algorithme d'apprentissage non pris en charge")

    X_train = np.vstack([data['representation'] for data in train_dataset])
    y_train = np.array([data['label'] for data in train_dataset])

    model.fit(X_train, y_train)

    return model



"""
Dans ce code :

  Nous avons simulé un ensemble de données d'entraînement avec des représentations et des étiquettes.
  Nous avons défini un dictionnaire d'algorithme avec l'algorithme "multinomial naive bayes" et les hyperparamètres associés.
  Nous avons appelé la fonction learn_model_from_dataset avec ces paramètres et vérifié si le modèle retourné n'est pas None, ce qui signifierait que l'apprentissage du modèle a réussi.

5 / 5

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


def write_predictions(directory, filename, predictions, algo_dico, sentence):
    try:
        with open(os.path.join(directory, filename), 'w') as file:
            file.write(sentence)
            file.write("Algorithme utilisé :\n")
            for key, value in algo_dico.items():
                file.write("{}: {}\n".format(key, value))

            for prediction in predictions:
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


# Main
representations = ['PX','GC', 'FOURIER', 'WAVELET']

dataset = load_transform_label_train_dataset('Data',representations)

dataset_test = load_transform_test_dataset('AllTest', representations)

algo_dico1 = {
    'algo': 'svm',
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale'
}
algo_dico2 = {
    'algo': 'multinomial naive bayes',
    'force_alpha': True
}
algo_dico3 = {
    'algo': 'k_neighbors',
    'n_neighbors': 5,
    'weights': 'uniform'
}
algo_dico4 = {
    'algo': 'decision_tree',
    'max_depth': 5,
    'min_samples_split': 3
}
algo_dico5 = {
    'algo': 'random_forest',
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2
}
algo_dico6 = {
    'algo': 'logistic_regression',
    'penalty': 'l2',
    'C': 1.0
}
algo_dico7 = {
    'algo': 'mlp',
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam'
}

# Diviser les données d'entraînement en ensembles d'entraînement et de validation
train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

#calculer le score d'entrainement
score = estimate_model_score(train_dataset, algo_dico1, k=5)
print("Score d'entrainement :", score)

# Calculer le score de validation
validation_score = estimate_model_score(validation_dataset, algo_dico1, k=5)
print("Score de validation:", validation_score)


# Entraîner le modèle sur l'ensemble d'entraînement
best_model = learn_model_from_dataset(train_dataset, algo_dico1)

# Prédire les étiquettes des données de test avec le modèle entraîné
test_predictions = predict_sample_label(dataset_test, best_model)

# Extraire les étiquettes prédites et les étiquettes réelles des données de test
predicted_labels = [data['label'] for data in test_predictions]
true_labels = [data['label'] for data in dataset_test]

# Calculer le score d'exactitude des prédictions par rapport aux étiquettes réelles
test_accuracy = accuracy_score(true_labels, predicted_labels)
print("Score de test:", test_accuracy)

sentence = f"Pour la machine qui détecte la mère à partir des images données, le score est de donnée d'entrainement est: {score},le score est de donnée de validation est: {validation_score},le score est de donnée de test est: {test_accuracy}. L'algorithme de classification utilisé est : {algo_dico1}\n"

# Enregistrer les prédictions dans un fichier
final_result = write_predictions("./", "test.txt", test_predictions, algo_dico1, sentence)
print(final_result)
