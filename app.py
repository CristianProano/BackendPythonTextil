import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ==============================================================
# CONFIGURACIÓN
# ==============================================================

RUTA_DATASET = "data/tiempo_produccion_rows.csv"
RUTA_MODELO = "modelos/modelo_unificado.pkl"

# ==============================================================
# APP
# ==============================================================

app = FastAPI(title="API Predicción Tiempo Producción")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego puedes limitar a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================
# CARGA MODELO UNIFICADO
# ==============================================================

bundle = joblib.load(RUTA_MODELO)

modelo = bundle["modelo"]
scaler_features = bundle["scaler_features"]
scaler_target = bundle["scaler_target"]
label_prenda = bundle["label_prenda"]
label_talla = bundle["label_talla"]
FEATURES = bundle["features"]
anio_min = bundle["anio_min"]
anio_max = bundle["anio_max"]

# ==============================================================
# UTILIDADES
# ==============================================================

def convertir_minutos(x):
    try:
        d, h, m = map(int, str(x).split(":"))
        return d * 1440 + h * 60 + m
    except:
        return np.nan

def minutos_a_dhhmm(total_min):
    total_min = int(round(total_min))
    d = total_min // 1440
    h = (total_min % 1440) // 60
    m = total_min % 60
    return f"{d:02d}:{h:02d}:{m:02d}"

# ==============================================================
# ENDPOINT SALUD
# ==============================================================

@app.get("/")
def health():
    return {"status": "ok"}

# ==============================================================
# ENDPOINT PRINCIPAL
# ==============================================================

@app.get("/predict")
def predict(anio: int = Query(..., description="Año a predecir")):
    """
    Retorna:
    codigo_prenda | talla | prediccion_tiempo
    """

    # ----------------------------------------------------------
    # 1. CARGAR DATA HISTÓRICA
    # ----------------------------------------------------------

    df = pd.read_csv(RUTA_DATASET)
    df.columns = [c.strip().lower() for c in df.columns]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    df["mes"] = df["fecha"].dt.month
    df["año"] = df["fecha"].dt.year

    df["tiempo_minutos"] = df["tiempo"].apply(convertir_minutos)
    df = df.dropna(subset=["tiempo_minutos"])

    # ----------------------------------------------------------
    # 2. AGREGACIÓN MENSUAL
    # ----------------------------------------------------------

    df = (
        df.groupby(["año", "mes", "codigo_prenda", "talla"], as_index=False)
        .agg({
            "cantidad_prenda": "sum",
            "tiempo_minutos": "sum"
        })
        .sort_values(["año", "mes"])
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # 3. ENCODING
    # ----------------------------------------------------------

    df["prenda_enc"] = label_prenda.transform(df["codigo_prenda"].astype(str))
    df["talla_enc"] = label_talla.transform(df["talla"].astype(str))

    # ----------------------------------------------------------
    # 4. FEATURES TEMPORALES
    # ----------------------------------------------------------

    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["trimestre"] = ((df["mes"] - 1) // 3) + 1
    df["año_norm"] = (anio - anio_min) / max(1, anio_max - anio_min)
    df["tiempo"] = (anio - anio_min) * 12 + df["mes"]

    # ----------------------------------------------------------
    # 5. LAGS Y ROLLINGS
    # ----------------------------------------------------------

    grp = df.groupby(["prenda_enc", "talla_enc"])["tiempo_minutos"]

    df["lag_1"] = grp.shift(1)
    df["lag_3"] = grp.shift(3)
    df["lag_6"] = grp.shift(6)
    df["var_pct"] = grp.pct_change()
    df["rolling_3"] = grp.rolling(3, min_periods=1).mean().reset_index(drop=True)
    df["rolling_6"] = grp.rolling(6, min_periods=1).mean().reset_index(drop=True)
    df["rolling_std_3"] = grp.rolling(3, min_periods=1).std().reset_index(drop=True)

    df.fillna(0, inplace=True)

    # ----------------------------------------------------------
    # 6. FEATURE MATRIX
    # ----------------------------------------------------------

    X = df[FEATURES]
    X_scaled = scaler_features.transform(X)

    # ----------------------------------------------------------
    # 7. PREDICCIÓN
    # ----------------------------------------------------------

    pred_log = modelo.predict(X_scaled)
    pred_log_unscaled = scaler_target.inverse_transform(
        pred_log.reshape(-1, 1)
    ).ravel()

    pred_min = np.maximum(0, np.expm1(pred_log_unscaled))

    df_out = pd.DataFrame({
        "codigo_prenda": df["codigo_prenda"],
        "talla": df["talla"],
        "prediccion_tiempo": [minutos_a_dhhmm(x) for x in pred_min]
    })

    # ----------------------------------------------------------
    # 8. RESPUESTA JSON (para dashboard)
    # ----------------------------------------------------------

    return {
        "anio": anio,
        "total_registros": len(df_out),
        "data": df_out.to_dict(orient="records")
    }
