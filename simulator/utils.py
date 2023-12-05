import joblib
from sklearn.preprocessing import PolynomialFeatures

class CommPredict:
    def __init__(self,num,comm_alo,comm_scale,ib='IB4') -> None:
        self.num=num/1024
        self.comm_alo=comm_alo
        self.comm_scale=comm_scale
        self.ib=ib
        self.prediction=self._get_predict_res()

    def _get_predict_res(self):
        poly_features = PolynomialFeatures(degree=3, include_bias=False)
        X_pred = poly_features.fit_transform([[self.num]])
        model=joblib.load(f'../comm/{self.comm_alo}_{self.comm_scale}_{self.ib}_model.joblib')
        predictions = model.predict(X_pred)/1024

        return predictions


def _get_model_config(self):
    if self._model_size==7:
        self._h=4096
        self._a=32
        self._l=32
    elif self._model_size==13:
        self._h=5120
        self._a=40
        self._l=40
    elif self._model_size==30:
        self._h=6144
        self._a=48
        self._l=60
    else: 
        self._h=8192
        self._a=64
        self._l=80

    return self._h,self._a,self._l