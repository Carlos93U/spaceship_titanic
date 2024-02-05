import pickle
import pandas as pd

# to load model
def load_model(path_file):
    with open(path_file, 'rb') as f:
        model = pickle.load(f)
    return model

# to predict values
def to_predict(model, data):
    prediction = model.predict((data,))
    return prediction

if __name__ == "__main__":
    # path of model
    path_file = './modelo_rf.pkl'

    # loading model
    modelo_cargado = load_model(path_file)

    # Data for predict
    mi_obj = pd.Series({
    'Group': '0018',
    'HomePlanet': 2,
    'CryoSleep': 0,
    'Destination': 1,
    'Age': 19.0,
    'VIP': 0,
    'RoomService': 0.0,
    'FoodCourt': 9.0,
    'ShoppingMall': 0.0,
    'Spa': 2823.0,
    'VRDeck': 0.0,
    'DeckName': 2,
    'DeckNumber': 4,
    'DeckSide': 2
})

    # to do prediction
    predict = to_predict(modelo_cargado, mi_obj)

    # view results
    print("predicciones:", predict)
