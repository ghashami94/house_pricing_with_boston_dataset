from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import losses
from keras import metrics
from keras import optimizers



def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1)) #last layer in regression has no activation function beacouse we don't wanna to limit the output
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae']) #mae : mean absolute error : it's for show   
    return model


def smoth_curve(points,factor = 0.9):
    smothed_points = []
    for point in points:
        if smothed_points:
            previous = smothed_points[-1]
            smothed_points.append(previous*factor+point*(1-factor))
        else:
            smothed_points.append(point)
    return smothed_points


(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

train_data.shape
test_data.shape
train_targets


#Preprocessing the data

#Normalize the data :Z-Score Normalization
mean = train_data.mean(axis =0) #every column : each feature 
train_data -= mean    #mean =0
std = train_data.std(axis = 0)
train_data /= std   # std=1

test_data -= mean
test_data /= std


# k-fold classification
k=4
num_val_samples= len(train_data)//k
num_epochs = 200
all_mae_histories=[]
for i in range(k):
    print("Processing fold #",i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,validation_data = (val_data,val_targets),epochs = num_epochs,batch_size =16,verbose =0)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i]for x in all_mae_histories])for i in range(num_epochs)]
smoth_mae_history = smoth_curve(average_mae_history[10:])
smoth_mae_history = average_mae_history[10:]
plt.plot(range(1,len(smoth_mae_history)+1),smoth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Valication MAE')
plt.show()

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


