import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator

def run_ekf(
    data: pd.DataFrame,
    bat_model: pd.DataFrame=None,
    soc_ocv: pd.DataFrame=None
):
    """
    EKF with parameter estimation for 2-RC model, *without log(C1/C2)*.
    State vector:
        [ SOC, V1, V2, R0, R1, R2, C1, C2 ]
    """

    # -----------------------------
    # Internal EKF hyperparameters
    # -----------------------------
    DeltaT   = 1.0
    Qn_rated = 5.0 * 3600.0
    meas_noise = 5e-3
    deg_ocv = 6
    clip_soc = True

    R_min, R_max = 1e-4, 0.5
    C_min, C_max = 1e2, 2.5e4

    Q_dyn   = 1e-6
    Q_R     = 1e-12
    Q_C     = 1e-14

    # --- inputs / measurement arrays ---
    Current = data['Measured_Current_R'].values
    Vt_meas = data['Measured_Voltage'].values
    N = len(Current)

    # --- initial parameter guesses ---
    if isinstance(bat_model, pd.DataFrame) and {'R0','R1','R2','C1','C2'}.issubset(bat_model.columns):
        R0_init = float(np.nanmedian(bat_model['R0'].values))
        R1_init = float(np.nanmedian(bat_model['R1'].values))
        R2_init = float(np.nanmedian(bat_model['R2'].values))
        C1_init = float(np.nanmedian(bat_model['C1'].values))
        C2_init = float(np.nanmedian(bat_model['C2'].values))
    else:
        R0_init = 0.02
        R1_init = 0.02
        R2_init = 0.02
        C1_init = 2000
        C2_init = 2000

    # clip
    R0_init = float(np.clip(R0_init, R_min, R_max))
    R1_init = float(np.clip(R1_init, R_min, R_max))
    R2_init = float(np.clip(R2_init, R_min, R_max))
    C1_init = float(np.clip(C1_init, C_min, C_max))
    C2_init = float(np.clip(C2_init, C_min, C_max))

    # Initial state:
    X = np.array([
        [1.0],          # SOC
        [0.0],          # V1
        [0.0],          # V2
        [R0_init],
        [R1_init],
        [R2_init],
        [C1_init],      # Linear C1
        [C2_init]       # Linear C2
    ])
    n_x = X.shape[0]

    # Covariances
    P_x = np.eye(n_x) * 0.01
    Q_x = np.diag([
        Q_dyn, Q_dyn, Q_dyn,   # SOC, V1, V2
        Q_R, Q_R, Q_R,         # R0, R1, R2
        Q_C, Q_C               # C1, C2
    ])
    R_x = meas_noise

    # ---- Prepare SOC–OCV data ----
    soc_grid = soc_ocv['SOC'].values.astype(float)
    ocv_grid = soc_ocv['OCV'].values.astype(float)

    # Sort SOC and OCV according to SOC
    sort_idx = np.argsort(soc_grid)
    soc_grid = soc_grid[sort_idx]
    ocv_grid = ocv_grid[sort_idx]

    # Remove duplicates
    unique_soc, unique_idx = np.unique(soc_grid, return_index=True)
    soc_grid = unique_soc
    ocv_grid = ocv_grid[unique_idx]

    # Ensure SOC strictly increases
    if np.any(np.diff(soc_grid) <= 0):
        raise ValueError("SOC must be strictly increasing after sorting and deduplication.")

    # ---- PCHIP Interpolator ----
    ocv_fun = PchipInterpolator(soc_grid, ocv_grid)
    docv_fun = ocv_fun.derivative()

    def ocv_and_deriv(soc):
        soc_cl = np.clip(soc, 0.0, 1.0)
        return float(ocv_fun(soc_cl)), float(docv_fun(soc_cl))



    # storage
    SOC_est = np.zeros(N)
    Vt_est  = np.zeros(N)
    Vt_err  = np.zeros(N)
    Params  = np.zeros((N,5))

    # ------------------------------------
    # EKF LOOP
    # ------------------------------------
    for k in tqdm(range(N)):

        U = Current[k]
        SOC, V1, V2, R0, R1, R2, C1, C2 = X.flatten()

        # enforce bounds
        R0 = np.clip(R0, R_min, R_max)
        R1 = np.clip(R1, R_min, R_max)
        R2 = np.clip(R2, R_min, R_max)
        C1 = np.clip(C1, C_min, C_max)
        C2 = np.clip(C2, C_min, C_max)

        # ---- Prediction ----
        Tau1 = R1 * C1
        Tau2 = R2 * C2
        a1 = np.exp(-DeltaT / Tau1)
        a2 = np.exp(-DeltaT / Tau2)
        b1 = R1 * (1 - a1)
        b2 = R2 * (1 - a2)

        SOC_pred = SOC - (DeltaT / Qn_rated) * U
        if clip_soc: SOC_pred = np.clip(SOC_pred,0,1)

        V1_pred = a1 * V1 + b1 * U
        V2_pred = a2 * V2 + b2 * U

        X_pred = np.array([
            [SOC_pred],
            [V1_pred],
            [V2_pred],
            [R0],
            [R1],
            [R2],
            [C1],
            [C2]
        ])

        # ---- Jacobian A ----
        A = np.eye(n_x)
        A[1,1] = a1
        A[2,2] = a2

        # Derivatives (no logC now)
        t = DeltaT

        da1_dR1 = a1 * t / (C1 * R1**2)
        da1_dC1 = a1 * t / (R1 * C1**2)
        db1_dR1 = (1 - a1) - R1 * da1_dR1
        db1_dC1 = -R1 * da1_dC1

        A[1,4] = da1_dR1 * V1 + db1_dR1 * U
        A[1,6] = da1_dC1 * V1 + db1_dC1 * U

        da2_dR2 = a2 * t / (C2 * R2**2)
        da2_dC2 = a2 * t / (R2 * C2**2)
        db2_dR2 = (1 - a2) - R2 * da2_dR2
        db2_dC2 = -R2 * da2_dC2

        A[2,5] = da2_dR2 * V2 + db2_dR2 * U
        A[2,7] = da2_dC2 * V2 + db2_dC2 * U

        P_pred = A @ P_x @ A.T + Q_x

        # ---- Measurement ----
        ocv_pred, dOCV = ocv_and_deriv(SOC_pred)
        Vt_pred = ocv_pred - V1_pred - V2_pred - U * R0
        innov = Vt_meas[k] - Vt_pred

        # Jacobian C_x
        C_x = np.zeros((1,n_x))
        C_x[0,0] = dOCV
        C_x[0,1] = -1
        C_x[0,2] = -1
        C_x[0,3] = -U

        C_x[0,4] = -(da1_dR1 * V1 + db1_dR1 * U)
        C_x[0,6] = -(da1_dC1 * V1 + db1_dC1 * U)
        C_x[0,5] = -(da2_dR2 * V2 + db2_dR2 * U)
        C_x[0,7] = -(da2_dC2 * V2 + db2_dC2 * U)

        S = float(C_x @ P_pred @ C_x.T + R_x)
        S = max(S, 1e-12)
        K = (P_pred @ C_x.T) / S

        X = X_pred + K * innov
        I = np.eye(n_x)
        P_x = (I - K @ C_x) @ P_pred @ (I - K @ C_x).T + (K * R_x) @ K.T

        # enforce bounds after update
        X[0,0] = np.clip(X[0,0], 0,1)
        X[3,0] = np.clip(X[3,0], R_min, R_max)
        X[4,0] = np.clip(X[4,0], R_min, R_max)
        X[5,0] = np.clip(X[5,0], R_min, R_max)
        X[6,0] = np.clip(X[6,0], C_min, C_max)
        X[7,0] = np.clip(X[7,0], C_min, C_max)

        # store results
        SOC_est[k] = X[0,0]
        C1_u, C2_u = X[6,0], X[7,0]
        ocv_u,_ = ocv_and_deriv(X[0,0])
        Vt_after = ocv_u - X[1,0] - X[2,0] - U * X[3,0]
        Vt_est[k] = Vt_after
        Vt_err[k] = Vt_meas[k] - Vt_after

        Params[k,:] = [X[3,0], X[4,0], X[5,0], C1_u, C2_u]

    return SOC_est, Vt_est, Vt_err, Params





# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from scipy.interpolate import interp1d

# def run_ekf(
#     data: pd.DataFrame,
#     bat_model: pd.DataFrame,
#     soc_ocv: pd.DataFrame,
#     DeltaT: float = 1.0,
#     Qn_rated: float = 5.0 * 3600.0,
#     min_param: float = 1e-6,
#     meas_noise: float = 1e-3,
#     deg_ocv: int = 6,
#     clip_soc: bool = True
# ):
#     """
#     Extended Kalman Filter with online parameter estimation for a 2-RC Thevenin model.
#     State vector: [SOC, V1, V2, R0, R1, R2, C1, C2]

#     Returns:
#       SOC_est (N,), Vt_est (N,), Vt_err (N,), Params (N,5) where Params cols = [R0,R1,R2,C1,C2]
#     """

#     # --- inputs / measurement arrays ---
#     Current = data['Measured_Current_R'].values
#     Vt_meas = data['Measured_Voltage'].values
#     # Temperature is available but we *estimate* parameters online; we may use T only for init.
#     Temperature = data['Measured_Temperature'].values if 'Measured_Temperature' in data.columns else np.zeros_like(Current)

#     N = len(Current)

#     # --- initial state (choose reasonable init) ---
#     # Use bat_model statistics to get initial parameter guesses if available
#     if isinstance(bat_model, pd.DataFrame) and {'R0','R1','R2','C1','C2'}.issubset(set(bat_model.columns)):
#         # Use median as robust starting point
#         R0_init = float(np.nanmedian(bat_model['R0'].values))
#         R1_init = float(np.nanmedian(bat_model['R1'].values))
#         R2_init = float(np.nanmedian(bat_model['R2'].values))
#         C1_init = float(np.nanmedian(bat_model['C1'].values))
#         C2_init = float(np.nanmedian(bat_model['C2'].values))
#     else:
#         # fallback defaults
#         R0_init, R1_init, R2_init = 0.02, 0.02, 0.02
#         C1_init, C2_init = 2000.0, 2000.0

#     SOC0 = 1.0  # if you have better init (e.g. coulomb count) swap here
#     X = np.array([[SOC0], [0.0], [0.0], [R0_init], [R1_init], [R2_init], [C1_init], [C2_init]])
#     n_x = X.shape[0]

#     # --- covariances (tune these) ---
#     P_x = np.eye(n_x) * 0.01
#     # small Q for dynamic states, smaller for parameters (slowly time-varying)
#     Q_x = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8, 1e-6, 1e-6])
#     R_x = meas_noise

#     # OCV fit
#     SOCOCV = np.polyfit(soc_ocv['SOC'].values, soc_ocv['OCV'].values, deg_ocv)
#     dSOCOCV = np.polyder(SOCOCV)

#     def ocv_and_deriv(soc):
#         ocv = np.polyval(SOCOCV, soc)
#         docv = np.polyval(dSOCOCV, soc)
#         return ocv, docv

#     # storage
#     SOC_est = np.zeros(N)
#     Vt_est = np.zeros(N)
#     Vt_err = np.zeros(N)
#     Params = np.zeros((N, 5))  # R0, R1, R2, C1, C2

#     for k in tqdm(range(N)):
#         U = Current[k]  # Ampere (your sign convention)
#         # Unpack state and enforce positivity for parameter usage
#         SOC, V1, V2, R0, R1, R2, C1, C2 = X.flatten()
#         R0 = max(R0, min_param); R1 = max(R1, min_param); R2 = max(R2, min_param)
#         C1 = max(C1, min_param); C2 = max(C2, min_param)

#         # --- Prediction ---
#         Tau1 = max(R1 * C1, min_param)
#         Tau2 = max(R2 * C2, min_param)
#         a1 = np.exp(-DeltaT / Tau1)
#         a2 = np.exp(-DeltaT / Tau2)
#         b1 = R1 * (1 - a1)
#         b2 = R2 * (1 - a2)

#         SOC_pred = SOC - (DeltaT / Qn_rated) * U
#         if clip_soc:
#             SOC_pred = np.clip(SOC_pred, 0.0, 1.0)

#         V1_pred = a1 * V1 + b1 * U
#         V2_pred = a2 * V2 + b2 * U

#         X_pred = np.array([[SOC_pred],
#                            [V1_pred],
#                            [V2_pred],
#                            [R0],
#                            [R1],
#                            [R2],
#                            [C1],
#                            [C2]])

#         # --- Linearized state transition Jacobian A (analytic) ---
#         A = np.eye(n_x)
#         A[0,0] = 1.0
#         A[1,1] = a1
#         A[2,2] = a2

#         # derivatives of a1 and b1 wrt R1 and C1 (safe since we're clipped)
#         t = DeltaT
#         da1_dR1 = a1 * (t) / (R1 * R1 * C1)
#         da1_dC1 = a1 * (t) / (R1 * C1 * C1)
#         db1_dR1 = (1 - a1) + R1 * (-da1_dR1)
#         db1_dC1 = - R1 * da1_dC1
#         A[1,4] = da1_dR1 * V1 + db1_dR1 * U  # dV1/dR1
#         A[1,6] = da1_dC1 * V1 + db1_dC1 * U  # dV1/dC1

#         da2_dR2 = a2 * (t) / (R2 * R2 * C2)
#         da2_dC2 = a2 * (t) / (R2 * C2 * C2)
#         db2_dR2 = (1 - a2) + R2 * (-da2_dR2)
#         db2_dC2 = - R2 * da2_dC2
#         A[2,5] = da2_dR2 * V2 + db2_dR2 * U  # dV2/dR2
#         A[2,7] = da2_dC2 * V2 + db2_dC2 * U  # dV2/dC2

#         P_pred = A @ P_x @ A.T + Q_x

#         # --- Measurement prediction using predicted state ---
#         ocv_pred, dOCV_pred = ocv_and_deriv(SOC_pred)
#         Vt_pred = ocv_pred - V1_pred - V2_pred - U * R0
#         innov = Vt_meas[k] - Vt_pred

#         # --- Measurement Jacobian C (analytic, includes param -> V1/V2 effect one-step) ---
#         C_x = np.zeros((1, n_x))
#         C_x[0,0] = dOCV_pred
#         C_x[0,1] = -1.0
#         C_x[0,2] = -1.0
#         C_x[0,3] = -U

#         # derivatives of V1_pred and V2_pred wrt R/C (computed above)
#         dV1_dR1 = da1_dR1 * V1 + db1_dR1 * U
#         dV1_dC1 = da1_dC1 * V1 + db1_dC1 * U
#         dV2_dR2 = da2_dR2 * V2 + db2_dR2 * U
#         dV2_dC2 = da2_dC2 * V2 + db2_dC2 * U

#         # measurement has -V1_pred - V2_pred, so jacobian entries are negative of partials
#         C_x[0,4] = -dV1_dR1
#         C_x[0,6] = -dV1_dC1
#         C_x[0,5] = -dV2_dR2
#         C_x[0,7] = -dV2_dC2

#         # --- Kalman gain and update (Joseph form for P) ---
#         S = (C_x @ P_pred @ C_x.T).squeeze() + R_x
#         S = max(S, 1e-12)
#         K = (P_pred @ C_x.T) / S  # (n_x,1)

#         X = X_pred + K * innov
#         I = np.eye(n_x)
#         P_x = (I - K @ C_x) @ P_pred @ (I - K @ C_x).T + (K * R_x) @ K.T

#         # --- enforce physical bounds after update ---
#         X[0,0] = np.clip(X[0,0], 0.0, 1.0)
#         X[3,0] = max(X[3,0], min_param)
#         X[4,0] = max(X[4,0], min_param)
#         X[5,0] = max(X[5,0], min_param)
#         X[6,0] = max(X[6,0], min_param)
#         X[7,0] = max(X[7,0], min_param)

#         # recompute terminal voltage after update (for logging)
#         SOC_u, V1_u, V2_u, R0_u, R1_u, R2_u, C1_u, C2_u = X.flatten()
#         ocv_u, _ = ocv_and_deriv(SOC_u)
#         Vt_after = ocv_u - V1_u - V2_u - U * R0_u
#         resid_after = Vt_meas[k] - Vt_after

#         # store
#         SOC_est[k] = X[0,0]
#         Vt_est[k] = Vt_after
#         Vt_err[k] = resid_after
#         Params[k,:] = np.array([X[3,0], X[4,0], X[5,0], X[6,0], X[7,0]])

#     return SOC_est, Vt_est, Vt_err, Params
