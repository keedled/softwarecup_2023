from sklearn.metrics import precision_score, recall_score, classification_report

def MacroF1(y_true,y_pred,average = None):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    macro_F1 = (2 * precision * recall) / (precision + recall)
    return macro_F1