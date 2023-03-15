import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_FRAME_PATH = './data/spin_ladder/spin_ladder_70_2.csv'
MODEL_SAVE_DIR = './simpleML/spin_ladder/70_2'
MODEL_NAME = 'rf_classifier'

if __name__ == '__main__':

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    df = pd.read_csv(DATA_FRAME_PATH)

    # Take 10% of the data with zero MZM states and all the data with non-zero MZM states
    df = df[df['num_zm'] == 0].sample(frac=0.1).append(df[df['num_zm'] > 0])
    df['zm_exists'] = df['num_zm'] > 0
    print(df.groupby('num_zm').count())

    X = df[['N', 'M', 'delta', 'q', 'mu', 'J', 'delta_q', 't', 'theta']]
    y = df['zm_exists']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    best_acc = 0
    best_depth = 0
    best_n_estimators = 1

    for depth in tqdm(range(1, 30), desc='Model optimization'):
        for n_estimators in range(1, 200, 10):
            model = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_depth = depth
                best_n_estimators = n_estimators
                pickle.dump(model, open(os.path.join(MODEL_SAVE_DIR, MODEL_NAME + '.pkl'), 'wb'))

    print(f'Best depth: {best_depth}')
    print(f'Best n_estimators: {best_n_estimators}')
    print(f'Best accuracy: {best_acc}')

    # Load best model
    model = pickle.load(open(os.path.join(MODEL_SAVE_DIR, MODEL_NAME + '.pkl'), 'rb'))
    y_pred = model.predict(X_train)

    print(f'Balanced accuracy: {balanced_accuracy_score(y_train, y_pred)}')
    print(f'F1 score: {f1_score(y_train, y_pred, average="weighted")}')

    y_pred = model.predict(X_test)

    print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}')
    print(f'F1 score: {f1_score(y_test, y_pred, average="weighted")}')

    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
