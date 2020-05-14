import data
import pipeline
import sklearn

from sklearn.metrics import classification_report

if __name__ == '__main__':
    X, y = data.load_data('train')
    print('{}\n{}'.format(X[0], y[0]))
    
    y_pred = pipeline.baseline_pipeline(X, y)
    print(classification_report(y, y_pred))