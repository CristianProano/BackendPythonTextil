import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import os

# ==============================================================
# CACHE SIMPLE EN DISCO (NO INVASIVO)
# ==============================================================

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(name: str):
    return os.path.join(CACHE_DIR, name)

def load_cache(name: str):
    path = _cache_path(name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

def save_cache(name: str, data):
    joblib.dump(data, _cache_path(name))
# ==============================================================
# CACHE PERSISTENTE (DATA → CACHE EN ARRANQUE)
# ==============================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PERSISTENT_MAP = {
    "predict_tiempo_2026.joblib": "tiempo_precalculado_2026.joblib",
    "predict_material_2026.joblib": "material_precalculado_2026.joblib",
    "predict_ventas.joblib": "ventas_precalculado.joblib",
}

def preload_cache_from_data():
    for cache_name, data_name in PERSISTENT_MAP.items():
        cache_path = os.path.join(CACHE_DIR, cache_name)
        data_path = os.path.join(DATA_DIR, data_name)

        if not os.path.exists(cache_path) and os.path.exists(data_path):
            try:
                data = joblib.load(data_path)
                joblib.dump(data, cache_path)
                print(f"✔ Cache restaurada desde data: {cache_name}")
            except Exception as e:
                print(f"⚠ Cache persistente inválida ({data_name}), se recalculará. Error: {e}")


preload_cache_from_data()

# Modelo Tiempo
RUTA_DATASET = "data/tiempo_produccion_rows.csv"
RUTA_MODELO = "modelos/modelo_unificado.pkl"
# Modelo Material
RUTA_DATASET_MATERIAL = "data/material_produccion_rows.csv"
RUTA_MODELO_MATERIAL = "modelos/modelo_material_unificado.pkl"

#CARGA MODELO VENTAS (UNIFICADO)

MODELO_VENTAS = joblib.load("modelos/modelo_prediccion_ventas_lightgbm_joblib.pkl")
LABEL_ALMACEN = joblib.load("modelos/labelencoder_almacen.pkl")
SCALER_VENTAS = joblib.load("modelos/scaler_features.pkl")

RUTA_VENTAS = "data/ventas_rows.csv"
RUTA_ALMACEN = "data/almacen_rows.csv"
RUTA_PRENDA = "data/prenda_rows.csv"

FEATURE_COLS = [
    "MES", "AÑO", "ALMACEN_ENC", "PRECIO_TOTAL",
    "MES_SIN", "MES_COS", "TRIMESTRE", "AÑO_NORM",
    "TIEMPO", "VENTA_PREVIA", "VARIACION"
]

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
    cache_key = f"predict_tiempo_{anio}.joblib"
    cache = load_cache(cache_key)
    if cache is not None:
        return cache
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

    response = {
        "anio": anio,
        "total_registros": len(df_out),
        "data": df_out.to_dict(orient="records")
    }

    save_cache(cache_key, response)
    joblib.dump(response, os.path.join(DATA_DIR, "tiempo_precalculado_2026.joblib"))
    return response


    # ----------------------------------------------------------
    #                  MODELO MATERIAL
    # ----------------------------------------------------------

# CARGA MODELO MATERIAL (UNIFICADO)

bundle_material = joblib.load(RUTA_MODELO_MATERIAL)

modelo_material = bundle_material["modelo"]
scaler_features_mat = bundle_material["scaler_features"]
scaler_cant_prenda = bundle_material["scaler_cantidad_prenda"]
label_material = bundle_material["label_material"]
label_talla_mat = bundle_material["label_talla"]
label_unidad = bundle_material["label_unidad"]
FEATURES_MAT = bundle_material["features"]
anio_min_mat = bundle_material["anio_min"]
anio_max_mat = bundle_material["anio_max"]
@app.get("/predict-material")
def predict_material(anio: int = Query(..., description="Año a predecir consumo de material")):
    """
    Retorna:
    codigo_prenda | talla | codigo_material | unidad | prediccion_cantidad_material_anual
    """
    cache_key = f"predict_material_{anio}.joblib"
    cache = load_cache(cache_key)
    if cache is not None:
        return cache
    # ----------------------------------------------------------
    # 1. CARGAR DATA HISTÓRICA
    # ----------------------------------------------------------

    df = pd.read_csv(RUTA_DATASET_MATERIAL)
    df.columns = [c.strip().lower() for c in df.columns]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    df["mes"] = df["fecha"].dt.month
    df["año"] = df["fecha"].dt.year

    # ----------------------------------------------------------
    # 2. AGREGACIÓN (IGUAL AL ENTRENAMIENTO)
    # ----------------------------------------------------------

    df = df.groupby(
        ["año", "mes", "codigo_material", "codigo_prenda", "talla", "unidad"],
        as_index=False
    ).agg({
        "cantidad_material": "sum",
        "cantidad_prenda": "sum"
    }).sort_values(["año", "mes"]).reset_index(drop=True)

    # ----------------------------------------------------------
    # 3. ENCODING
    # ----------------------------------------------------------

    df["material_enc"] = label_material.transform(df["codigo_material"].astype(str))
    df["talla_enc"] = label_talla_mat.transform(df["talla"].astype(str))
    df["unidad_enc"] = label_unidad.transform(df["unidad"].astype(str))

    # ----------------------------------------------------------
    # 4. FEATURES TEMPORALES
    # ----------------------------------------------------------

    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["trimestre"] = ((df["mes"] - 1) // 3) + 1
    df["año_norm"] = (df["año"] - anio_min_mat) / max(1, anio_max_mat - anio_min_mat)
    df["tiempo"] = (df["año"] - anio_min_mat) * 12 + df["mes"]

    df["cantidad_prenda_norm"] = scaler_cant_prenda.transform(
        df[["cantidad_prenda"]]
    )

    # ----------------------------------------------------------
    # 5. LAGS Y ROLLINGS
    # ----------------------------------------------------------

    df["grupo"] = (
        df["material_enc"].astype(str) + "_" +
        df["codigo_prenda"].astype(str) + "_" +
        df["talla"].astype(str)
    )

    grp = df.groupby("grupo")["cantidad_material"]

    df["lag_1"] = grp.shift(1)
    df["lag_2"] = grp.shift(2)
    df["lag_3"] = grp.shift(3)
    df["roll_3"] = grp.rolling(3, min_periods=1).mean().reset_index(drop=True)
    df["roll_6"] = grp.rolling(6, min_periods=1).mean().reset_index(drop=True)
    df["var_pct"] = grp.pct_change()

    df.fillna(0, inplace=True)

    # ----------------------------------------------------------
    # 6. ÚLTIMO ESTADO POR SERIE
    # ----------------------------------------------------------

    ultimos = df.groupby("grupo", as_index=False).tail(1)

    resultados = []

    for _, row in ultimos.iterrows():
        mensual = []
        lag, l1, l2 = row["cantidad_material"], row["lag_1"], row["lag_2"]
        r3, r6 = row["roll_3"], row["roll_6"]

        for mes in range(1, 13):
            feat = {
                "mes": mes,
                "año": anio,
                "material_enc": row["material_enc"],
                "talla_enc": row["talla_enc"],
                "cantidad_prenda_norm": row["cantidad_prenda_norm"],
                "mes_sin": np.sin(2 * np.pi * mes / 12),
                "mes_cos": np.cos(2 * np.pi * mes / 12),
                "trimestre": ((mes - 1) // 3) + 1,
                "año_norm": (anio - anio_min_mat) / max(1, anio_max_mat - anio_min_mat),
                "tiempo": (anio - anio_min_mat) * 12 + mes,
                "lag_1": lag,
                "lag_2": l1,
                "lag_3": l2,
                "roll_3": r3,
                "roll_6": r6,
                "var_pct": 0,
                "unidad_enc": row["unidad_enc"]
            }

            Xf = pd.DataFrame([feat])[FEATURES_MAT]
            Xf = scaler_features_mat.transform(Xf)

            pred = modelo_material.predict(Xf)[0]
            mensual.append(pred)

            l2, l1 = l1, lag
            lag = 0.5 * lag + 0.5 * pred
            r3 = (r3 * 2 + pred) / 3
            r6 = (r6 * 5 + pred) / 6

        resultados.append({
            "codigo_prenda": row["codigo_prenda"],
            "talla": row["talla"],
            "codigo_material": row["codigo_material"],
            "unidad": row["unidad"],
            "prediccion_cantidad_material_anual": float(np.sum(mensual))
        })

    response = {
        "anio": anio,
        "total_registros": len(resultados),
        "data": resultados
    }

    save_cache(cache_key, response)
    joblib.dump(response, os.path.join(DATA_DIR, "material_precalculado_2026.joblib"))
    return response

    # ----------------------------------------------------------
    #                  MODELO VENTAS
    # ----------------------------------------------------------
@app.get("/predict-ventas")
def predict_ventas():
    cache_key = "predict_ventas.joblib"
    cache = load_cache(cache_key)
    if cache is not None:
        return cache
    # ==============================
    # 1. CARGAR CSV
    # ==============================
    df = pd.read_csv(RUTA_VENTAS)
    df.columns = [c.strip().upper() for c in df.columns]

    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FECHA"])

    df["MES"] = df["FECHA"].dt.month
    df["AÑO"] = df["FECHA"].dt.year

    # ==============================
    # 2. AGRUPAR (IGUAL ENTRENAMIENTO)
    # ==============================
    df_grouped = df.groupby(
        ["MES", "AÑO", "CODIGO_ALMACEN"],
        as_index=False
    ).agg({
        "CANTIDAD": "sum",
        "PRECIO_TOTAL": "sum"
    }).sort_values(["AÑO", "MES"]).reset_index(drop=True)

    # ==============================
    # 3. FEATURES
    # ==============================
    df_grouped["MES_SIN"] = np.sin(2*np.pi*df_grouped["MES"]/12)
    df_grouped["MES_COS"] = np.cos(2*np.pi*df_grouped["MES"]/12)
    df_grouped["TRIMESTRE"] = ((df_grouped["MES"]-1)//3)+1

    anio_min = df_grouped["AÑO"].min()
    anio_max = df_grouped["AÑO"].max()

    df_grouped["AÑO_NORM"] = (
        (df_grouped["AÑO"]-anio_min) /
        max(1,(anio_max-anio_min))
    )

    df_grouped["TIEMPO"] = (df_grouped["AÑO"]-anio_min)*12 + df_grouped["MES"]

    df_grouped["CODIGO_ALMACEN"] = df_grouped["CODIGO_ALMACEN"].astype(str)
    df_grouped["ALMACEN_ENC"] = LABEL_ALMACEN.transform(df_grouped["CODIGO_ALMACEN"])

    df_grouped["VENTA_PREVIA"] = (
        df_grouped.groupby("ALMACEN_ENC")["CANTIDAD"]
        .shift(1).fillna(0)
    )

    df_grouped["VARIACION"] = (
        df_grouped.groupby("ALMACEN_ENC")["CANTIDAD"]
        .pct_change().fillna(0)
    )

    # ==============================
    # 4. PREDICCIÓN (MISMO LOOP)
    # ==============================
    ultimos = df_grouped.groupby("ALMACEN_ENC").tail(1)
    predicciones = []

    for _, row in ultimos.iterrows():

        almacen_enc = row["ALMACEN_ENC"]
        codigo_almacen = row["CODIGO_ALMACEN"]
        venta_previa = row["CANTIDAD"]
        precio_base = row["PRECIO_TOTAL"]
        tiempo_inicio = int(row["TIEMPO"])

        for step in range(1, 13):

            tiempo_fut = tiempo_inicio + step
            año_fut = anio_min + (tiempo_fut - 1) // 12
            mes_fut = ((tiempo_fut - 1) % 12) + 1

            X_future = pd.DataFrame([{
                "MES": mes_fut,
                "AÑO": año_fut,
                "ALMACEN_ENC": almacen_enc,
                "PRECIO_TOTAL": precio_base,
                "MES_SIN": np.sin(2*np.pi*mes_fut/12),
                "MES_COS": np.cos(2*np.pi*mes_fut/12),
                "TRIMESTRE": ((mes_fut-1)//3)+1,
                "AÑO_NORM": (año_fut-anio_min)/max(1,(anio_max-anio_min)),
                "TIEMPO": tiempo_fut,
                "VENTA_PREVIA": venta_previa,
                "VARIACION": 0.0
            }])[FEATURE_COLS]

            X_scaled = SCALER_VENTAS.transform(X_future)
            pred = float(MODELO_VENTAS.predict(X_scaled)[0])

            venta_previa = pred

            predicciones.append({
                "codigo_almacen": codigo_almacen,
                "cantidad_predicha": pred
            })

    # ==============================
    # 5. AGREGAR POR ALMACÉN
    # ==============================
    df_pred = pd.DataFrame(predicciones)

    df_final = (
        df_pred.groupby("codigo_almacen", as_index=False)
        ["cantidad_predicha"].sum()
    )

    # ==============================
    # 6. INDEXADO (ARCHIVOS EXTRA)
    # ==============================
    df_almacen = pd.read_csv(RUTA_ALMACEN)
    df_prenda = pd.read_csv(RUTA_PRENDA)

    df_almacen.columns = df_almacen.columns.str.lower()
    df_prenda.columns = df_prenda.columns.str.lower()
    df_final.columns = df_final.columns.str.lower()

    df_final["codigo_almacen"] = df_final["codigo_almacen"].astype(str)
    df_almacen["codigo_almacen"] = df_almacen["codigo_almacen"].astype(str)
    df_prenda["codigo_prenda"] = df_prenda["codigo_prenda"].astype(str)

    df_merge1 = df_final.merge(df_almacen, on="codigo_almacen", how="left")
    df_final_enriquecido = df_merge1.merge(df_prenda, on="codigo_prenda", how="left")

    df_final_enriquecido = df_final_enriquecido[[
        "codigo_almacen",
        "codigo_prenda",
        "nombre_prenda",
        "talla",
        "cantidad_predicha"
    ]]

    # ==============================
    # 7. RESPUESTA
    # ==============================
    response = {
        "anio_prediccion": int(anio_max + 1),
        "total_registros": len(df_final_enriquecido),
        "data": df_final_enriquecido.to_dict(orient="records")
    }

    save_cache(cache_key, response)
    joblib.dump(response, os.path.join(DATA_DIR, "ventas_precalculado.joblib"))
    return response

