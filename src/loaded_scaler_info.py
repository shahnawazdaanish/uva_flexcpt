import numpy as np
from sklearn.preprocessing import StandardScaler


class LoadedScalerInfo:
    def __init__(self, scaler_state=None):
        self.scaler_x = None
        self.scaler_y = None
        self.scaler_x_ch4 = None
        self.scaler_y_ch4 = None
        self.X_scaler_s = None
        self.Y_scaler_r = None
        
        if scaler_state is None and scaler_state != {}:
            raise ValueError("scaler_state must be provided to initialize LoadedScalerInfo.")

        self.reinstate_scalers(scaler_state)

    def reinstate_scalers(self, scaler_state):
        if scaler_state is None or scaler_state == {}:
            raise ValueError("scaler_state must be provided to reinstate scalers.")

        self.scaler_x = self._restore_scaler_state(StandardScaler(), scaler_state.get("scaler_x"))
        self.scaler_y = self._restore_scaler_state(StandardScaler(), scaler_state.get("scaler_y"))
        self.scaler_x_ch4 = self._restore_scaler_state(StandardScaler(), scaler_state.get("scaler_x_ch4"))
        self.scaler_y_ch4 = self._restore_scaler_state(StandardScaler(), scaler_state.get("scaler_y_ch4"))
        self.X_scaler_s = self._restore_scaler_state(StandardScaler(), scaler_state.get("X_scaler_s"))
        self.Y_scaler_r = self._restore_scaler_state(StandardScaler(), scaler_state.get("Y_scaler_r"))

        print("Scalers reinstated successfully.")

    @staticmethod
    def _restore_scaler_state(scaler_obj, state):
        if state is None or scaler_obj is None:
            return scaler_obj 

        for attr, value in state.items():
            if isinstance(value, np.ndarray):
                setattr(scaler_obj, attr, value.copy())
            else:
                setattr(scaler_obj, attr, value)

        return scaler_obj