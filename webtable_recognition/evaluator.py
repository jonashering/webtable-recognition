from sklearn.metrics import classification_report


def evaluation_report(Y_true, Y_pred, classes=None):
    print(classification_report(Y_true, Y_pred, target_names=classes))
