from sklearn.metrics import classification_report


def evaluation_report(Y_true, Y_pred):
    print(classification_report(Y_true, Y_pred))
