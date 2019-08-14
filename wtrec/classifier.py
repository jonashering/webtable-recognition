from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
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


def create_nn_classifier():
    """
    Create neural network for image rendering approach

    Args:
        None
    Returns:
        Keras model (compiled) implementing corresponding interface
    """
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
