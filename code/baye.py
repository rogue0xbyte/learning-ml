import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def NBPredict(data, features):
    X = data[features]
    y = data['In_Game_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    predicted_prices = naive_bayes.predict(X_test)

    accuracy = accuracy_score(y_test, predicted_prices)
    precision = precision_score(y_test, predicted_prices, average='weighted')
    recall = recall_score(y_test, predicted_prices, average='weighted')
    f1 = f1_score(y_test, predicted_prices, average='weighted')

    classification_rep = classification_report(y_test, predicted_prices)
    conf_matrix = confusion_matrix(y_test, predicted_prices)

    predictions_df = pd.DataFrame({'Predicted_In_Game_Price': predicted_prices}, index=X_test.index)
    output_df = pd.concat([X_test, predictions_df], axis=1)

    output_df['Actual_In_Game_Price'] = y_test
    output_df['Correct_Prediction'] = output_df['Actual_In_Game_Price'] == output_df['Predicted_In_Game_Price']
    output_df['Correct_Prediction'] = output_df['Correct_Prediction'].map({True: 'True', False: 'False'})

    output_df.to_csv('predicted_prices_with_actual.csv', index=False)
    output_df.to_csv('predicted_prices.csv', index=False)

    # Save metrics and confusion matrix to a TXT file
    with open('metrics_and_confusion_matrix.txt', 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')
        file.write(f'Precision: {precision}\n')
        file.write(f'Recall: {recall}\n')
        file.write(f'F1 Score: {f1}\n\n')
        
        file.write('Classification Report:\n')
        file.write(classification_rep)
        
        file.write('\n\nConfusion Matrix:\n')
        file.write(np.array2string(conf_matrix, separator=', '))
    
    # print("Predictions and metrics saved.")

    return accuracy
