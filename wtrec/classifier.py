from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers



def create_baseline_classifier():
    """
    Create GBDT classifier as specified in baseline paper
    
    Args:
        None
    Returns:
        Scikit learn classifier implementing corresponding interface
    """
    return GradientBoostingClassifier(min_samples_leaf=2,
                                      n_estimators=100,
                                      random_state=0)


def create_nn_classifier(freeze_conv=True):
    """
    Create neural network for image rendering approach
    
    Args:
        freeze_conv: Disable training of convolutional layers
    Returns:
        Keras model (compiled) implementing corresponding interface
    """
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='relu'))

    if freeze_conv:
        model.layers[0].trainable = False

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
