from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch
import pickle
from Sea_Level_Prediction_Model import LSTMModel
from datetime import datetime
import datetime
from dateutil.relativedelta import relativedelta
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.chdir("C:\\Users\Qing Rui\Desktop\BC3407 Business Transformation\Project\APIs Final\Sealevel Final")

class Optimization:
        """
        Optimization is a helper class that takes model, loss function, optimizer function
        learning scheduler (optional), early stopping (optional) as inputs. In return, it
        provides a framework to train and validate the models, and to predict future values
        based on the models.
        --Attributes--
        model: 
            Model class created for the type of RNN
        loss_fn: torch.nn.modules.Loss
            Loss function to calculate the losses
        optimizer: torch.optim.Optimizer 
            Optimizer function to optimize the loss function
        train_losses: list[float]
            The loss values from the training
        val_losses: list[float]
            The loss values from the validation
        """
        def __init__(self, model, loss_fn, optimizer):
            self.model = model
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.train_losses = []
            self.val_losses = []
            
        def train_step(self, x, y):
            """
            Given the features (x) and the target values (y) tensors, the method completes
            one step of the training. First, it activates the train mode to enable back prop.
            After generating predicted values (yhat) by doing forward propagation, it calculates
            the losses by using the loss function. Then, it computes the gradients by doing
            back propagation and updates the weights by calling step() function.

            --Arguments--
                x: torch.Tensor
                    Tensor for features to train one step
                y: torch.Tensor
                    Tensor for target values to calculate losses

            """
            # Sets model to train mode
            self.model.train()

            # Makes predictions
            yhat = self.model(x)

            # Computes loss
            loss = self.loss_fn(y, yhat)

            # Computes gradients
            loss.backward()

            # Updates parameters and zeroes gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
            """
            The method takes DataLoaders for training and validation datasets, batch size for
            mini-batch training, number of epochs to train, and number of features as inputs.
            Then, it carries out the training by iteratively calling the method train_step for
            n_epochs times. Finally, it saves the model in a designated file path.

            --Arguments--
                train_loader: torch.utils.data.DataLoader
                    DataLoader that stores training data
                val_loader: torch.utils.data.DataLoader
                    DataLoader that stores validation data
                batch_size: int
                    Batch size for mini-batch training
                n_epochs: int 
                    Number of epochs, i.e., train steps, to train
                n_features: int
                    Number of feature columns

            """
            model_path = f'model_lstm'
            
            for epoch in range(1, n_epochs + 1):
                # mini-batch training iteration of training datasets
                batch_losses = []
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                    y_batch = y_batch.to(device)
                    loss = self.train_step(x_batch, y_batch)
                    batch_losses.append(loss)
                    # update training loss value
                training_loss = np.mean(batch_losses)
                self.train_losses.append(training_loss)
                
                with torch.no_grad():
                    # mini-batch training iteration of validation datasets
                    batch_val_losses = []
                    validation = []
                    validation_values = []
                    for x_val, y_val in val_loader:
                        x_val = x_val.view([batch_size, -1, n_features]).to(device)
                        y_val = y_val.to(device)
                        self.model.eval()
                        yhat = self.model(x_val)
                        val_loss = self.loss_fn(y_val, yhat).item()
                        batch_val_losses.append(val_loss)
                        #
                        validation.append(yhat.to(device).detach().numpy())
                        validation_values.append(y_val.to(device).detach().numpy())
                    # update validation loss value
                    validation_loss = np.mean(batch_val_losses)
                    self.val_losses.append(validation_loss)
                
                # print loss value per epoch period
                if (epoch <= 10) | (epoch % 10 == 0):
                    print(
                        f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                    )

            torch.save(self.model.state_dict(), model_path)
            return validation, validation_values

        def evaluate(self, test_loader, batch_size=1, n_features=1):
            """
            The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
            and number of features as inputs. Similar to the model validation, it iteratively
            predicts the target values and calculates losses. Then, it returns two lists that
            hold the predictions and the actual values.

            Note:
                This method assumes that the prediction from the previous step is available at
                the time of the prediction, and only does one-step prediction into the future.

            --Arguments--
                test_loader: torch.utils.data.DataLoader 
                    DataLoader that stores test data
                batch_size: int
                    Batch size for mini-batch training
                n_features: int
                    Number of feature columns

            --Returns--
                predictions: list[float]
                    The values predicted by the model
                values: list[float]
                    The actual values in the test set.
            """
            # mini-batch testing to evaluate data from test dataset
            with torch.no_grad():
                predictions = []
                values = []
                for x_test, y_test in test_loader:
                    x_test = x_test.view([batch_size, -1, n_features]).to(device)
                    y_test = y_test.to(device)
                    self.model.eval()
                    yhat = self.model(x_test)
                    # save model prediction result to list
                    predictions.append(yhat.to(device).detach().numpy())
                    values.append(y_test.to(device).detach().numpy())

            return predictions, values

        def predict(self, future_loader, batch_size=1, n_features=1):
            """
            The method takes DataLoaders for the predicting future dataset, batch size for mini-batch testing,
            and number of features as inputs. 

            --Arguments--
                test_loader: torch.utils.data.DataLoader 
                    DataLoader that stores test data
                batch_size: int
                    Batch size for mini-batch training
                n_features: int
                    Number of feature columns

            --Returns--
                predictions: list[float]
                    The values predicted by the model
            """
            # mini-batch testing to predict data from future dataset
            with torch.no_grad():
                predictions = []
                for x_test in test_loader:
                    x_test = x_test.view([batch_size, -1, n_features]).to(device)
                    self.model.eval()
                    yhat = self.model(x_test)
                    # save model prediction result to list
                    predictions.append(yhat.to(device).detach().numpy())

            return predictions
        
        def plot_losses(self):
            """
            The method plots the calculated loss values for training and validation
            """
            plt.figure(figsize=[8, 5])
            plt.plot(self.train_losses, label="Training loss")
            plt.plot(self.val_losses, label="Validation loss")
            plt.legend()
            plt.title("Losses")
            plt.show()
            plt.close()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Xtrain.pkl', 'rb') as g:
    X_train = pickle.load(g)

with open('yTrain.pkl', 'rb') as h:
    y_train = pickle.load(h)

with open('XTest.pkl', 'rb') as i:
    X_test = pickle.load(i)

with open('yTest.pkl', 'rb') as j:
    y_test = pickle.load(j)

    
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

#method POST means use "body" on Postman, if GET then use params
@app.route("/sealevel", methods = ['POST']) #This adds a "/predict" to the end of the website


def sealevel(): #This is to query for the input from the user
    #Request for input


    def onehot_encode_pd(df, cols):
        for col in cols:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1) #.drop(columns=col)
        return df
    def generate_cyclical_features(df, col_name, period, start_num=0):
        kwargs = {
            f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
            f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
                }
        return df.assign(**kwargs).drop(columns=[col_name])
    def feature_label_split(df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y
    def give_year_name(df, start, end, name):
        i = start
        while i <= end:
            df[f'{name}_{i}'] = 0
            i += 1
        return df
    def get_scaler(scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()
    def inverse_transform(scaler, df, columns):
        for col in columns:
            df[col] = scaler.inverse_transform(df[col])
        return df
    def format_predictions(predictions, values, df_test, scaler):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
        return df_result


    ### Postman Inputs here
    # data = {"date" : "2021-12-23T18:25:43-05:00"}

    data = request.get_json()

    input_time = data['date'] # YYYY-MM-DD
    new_date = input_time.split("T")[0]
    year, month, day = map(int, new_date.split('-'))
    year = int(year) +1
    edate = datetime.date(year, 12, 31)
    sdate = edate - datetime.timedelta(days= 2 * 365)

    # make a dataframe with range of sdate and edate
    df_future = pd.DataFrame(pd.date_range(sdate, edate, freq='d'))
    df_future.columns = ['time']
    df_future['value'] = 0 # it's just here because the function need it as an input
    df_future.set_index('time', inplace=True)


    # make features with datetime object
    df_future = (df_future
                .assign(day = df_future.index.day)
                .assign(month = df_future.index.month)
                .assign(week_of_year = df_future.index.week)
                .assign(year = df_future.index.year)
                )

    # one hot encoding the features
    df_future = onehot_encode_pd(df_future, ['day', 'month', 'week_of_year'])
    df_future = give_year_name(df_future, 1995, 2014, 'year')
    df_future = onehot_encode_pd(df_future, ['year'])

    df_future = generate_cyclical_features(df_future, 'day', 31, 1)
    df_future = generate_cyclical_features(df_future, 'month', 12, 1)
    df_future = generate_cyclical_features(df_future, 'week_of_year', 52, 0)



    X_fut, y_fut = feature_label_split(df_future, 'value')

    batch_size = 32
    #scaling
    scaler = get_scaler('minmax')

    X_train_arr = scaler.fit_transform(X_train)
    X_fut_arr = scaler.transform(X_fut)
    # X_fut_arr = np.array(X_fut)
    y_train_arr = scaler.fit_transform(y_train)
    y_fut_arr = scaler.transform(y_fut)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # convert to tensor
    X_fut_tens = torch.Tensor(X_fut_arr)
    y_fut_tens = torch.Tensor(y_fut_arr)

    # convert to Dataset object (pytorch)
    future = TensorDataset(X_fut_tens, y_fut_tens)
    future_loader = DataLoader(future, batch_size=32, shuffle=False, drop_last=True)


    # predict using evaluate function

    fut_predictions, fut_values = model.evaluate(
        future_loader,
        batch_size= 32,
        n_features=125
    )

    # format the prediction result
    df_future_predict = format_predictions(fut_predictions, fut_values, X_fut, scaler)
    df_future_predict['time'] = df_future_predict.index

    test = df_future_predict.loc[df_future_predict['time'] == new_date] ### YYYY-MM-DD
    x = test['prediction']

    pred_out = pd.cut(x, bins=[0,1600,1800,2000],
        labels = ['Slow Sea Level Rise expected','Moderate Sea Level rise expected','Danger! Extreme Sea Level Rise, Stay away from coastal regions!'])

    pred_label = pred_out[pred_out.transform(type) == str].values[0]

    Sea_Intensity = [f"Predicted {pred_label} at {new_date}"]


    return jsonify(Sea_Intensity)


#Inputs should come in json format
if __name__ == '__main__':
    app.run() #Default port is 5000, can add port = ? to edit the port to your desired port
