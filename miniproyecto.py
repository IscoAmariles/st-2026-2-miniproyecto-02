"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. El informe,
docs/informe.md, cita las cifras y las figuras que produce este guion.

Serie del grupo: uso mensual de tarjetas débito en el comercio minorista de
Islandia, enero de 2000 a agosto de 2013 (164 meses). Ficha en datos/README.md.
El modelo a superar es el mejor del miniproyecto 1: SARIMAX(2,1,0)(0,1,1)12
con los días hábiles del mes como exógena.
"""
import time
import warnings
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# Estilo de las figuras: trazos delgados, cuadrícula tenue y los colores de
# las series siempre en el mismo orden (azul, naranja, aqua).
AZUL, NARANJA, AQUA = "#257de0", "#f47230", "#00b894"
TINTA, TINTA_SUAVE = "#191919", "#666666"
plt.rcParams.update({
    "axes.prop_cycle": matplotlib.cycler(color=[AZUL, NARANJA, AQUA, "#f2a600"]),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": TINTA,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.8,
    "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE,
    "lines.linewidth": 1.8, "legend.frameon": False, "font.size": 9,
})

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# La misma serie del miniproyecto 1, con su ficha en datos/README.md.
ARCHIVO = "datos/debito_islandia.csv"
FECHA = "fecha"
VALOR = "uso_debito"
PERIODO = 12

# Umbral del miniproyecto 1 para marcar un mes como atípico: residuo de la
# STL robusta mayor que UMBRAL veces la desviación robusta (1.4826 * MAD).
UMBRAL = 3.0


def detectar_atipicos(serie):
    """Meses cuyo residuo de la STL robusta supera UMBRAL desviaciones robustas."""
    residuo = STL(serie, period=PERIODO, robust=True).fit().resid
    escala = 1.4826 * np.median(np.abs(residuo - np.median(residuo)))
    z = (residuo - np.median(residuo)) / escala
    return pd.DataFrame({"valor": serie[np.abs(z) > UMBRAL],
                         "residuo": residuo[np.abs(z) > UMBRAL],
                         "z": z[np.abs(z) > UMBRAL]})


def preparar(serie):
    """La misma decisión del miniproyecto 1 sobre huecos y datos atípicos.

    - Huecos: un mes vacío se rellena por interpolación lineal en el tiempo.
      La serie de Islandia no trae ninguno, pero la regla queda en el código.
    - Atípicos: se detectan con el residuo de una STL robusta y se imprimen,
      pero NO se modifican: son meses reales (la crisis bancaria de 2008 y
      meses con más o menos días hábiles), no errores de captura.
    Devuelve una serie sin NaN.
    """
    vacios = serie.isna().sum()
    serie = serie.interpolate(method="time").bfill().ffill()
    print(f"  Huecos interpolados: {vacios}")
    atipicos = detectar_atipicos(serie)
    print(f"  Atípicos (|z| > {UMBRAL:g} sobre el residuo STL robusto), "
          f"se conservan: {len(atipicos)}")
    if len(atipicos):
        tabla = atipicos.copy()
        tabla.index = tabla.index.strftime("%Y-%m")
        print(tabla.round(3).to_string())
    atipicos.to_csv(RESULTADOS / "atipicos.csv", index_label="fecha")
    return serie


# ---------- Días hábiles de Islandia ----------
# La exógena del miniproyecto 1 son los días hábiles del mes: de lunes a
# viernes, menos los festivos oficiales de Islandia. En el miniproyecto 1 los
# festivos salían de la librería `holidays`, que aquí no se permite, así que
# se calculan con el calendario: dependen solo de la fecha, se conocen con
# años de anticipación y no usan ningún dato de la serie.

def pascua(anio):
    """Domingo de Pascua (algoritmo gregoriano anónimo de Meeus, Jones y Butcher)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(anio, mes, dia + 1)


@lru_cache(maxsize=None)
def festivos_islandia(anio):
    """Los 14 festivos oficiales de Islandia en un año, los mismos que daba la
    librería `holidays` en el miniproyecto 1. Nochebuena y Nochevieja son
    medio día hábil y `holidays` no las cuenta, así que aquí tampoco."""
    domingo = pascua(anio)
    fijos = [pd.Timestamp(anio, mes, dia) for mes, dia in
             [(1, 1),     # Año Nuevo
              (5, 1),     # Día del Trabajo
              (6, 17),    # Día Nacional
              (12, 25),   # Navidad
              (12, 26)]]  # San Esteban
    # Jueves y Viernes Santo, Pascua, lunes de Pascua, Ascensión,
    # Pentecostés y lunes de Pentecostés.
    moviles = [domingo + pd.Timedelta(days=n) for n in (-3, -2, 0, 1, 39, 49, 50)]
    abril_19, agosto_1 = pd.Timestamp(anio, 4, 19), pd.Timestamp(anio, 8, 1)
    verano = abril_19 + pd.Timedelta(days=(3 - abril_19.dayofweek) % 7)   # 1.er jueves después del 18 de abril
    comercio = agosto_1 + pd.Timedelta(days=(0 - agosto_1.dayofweek) % 7)  # 1.er lunes de agosto
    return tuple(sorted(fijos + moviles + [verano, comercio]))


def dias_habiles(fechas):
    """Días hábiles de cada mes (lunes a viernes no festivos), centrados en 21
    para que un coeficiente se lea como el efecto de un día hábil más."""
    fechas = pd.DatetimeIndex(fechas)
    festivos = np.array([f for anio in range(fechas.year.min(), fechas.year.max() + 1)
                         for f in festivos_islandia(anio)], dtype="datetime64[D]")
    inicio = fechas.to_numpy().astype("datetime64[D]")
    fin = (fechas + pd.offsets.MonthBegin(1)).to_numpy().astype("datetime64[D]")
    return pd.Series(np.busday_count(inicio, fin, holidays=festivos) - 21,
                     index=fechas, name="dias_habiles", dtype=float)


def exogenas(serie):
    """Días hábiles de la serie y de los 12 meses siguientes, como en el
    miniproyecto 1."""
    fechas = pd.date_range(serie.index[0], periods=len(serie) + PERIODO, freq="MS")
    return dias_habiles(fechas).to_frame()


# ---------- El modelo a superar ----------
ORDEN, ORDEN_ESTACIONAL = (2, 1, 0), (0, 1, 1, PERIODO)
NOMBRE_CLASICO = "SARIMAX(2,1,0)(0,1,1)12 + días hábiles"
# Sus métricas en el backtest del miniproyecto 1 (resultados/metricas.csv de
# ese repositorio): la fila de referencia tiene que dar lo mismo.
METRICAS_MP1 = {"RMSE": 0.64, "MAE": 0.54, "MAPE": 2.56}


def modelo_clasico(serie):
    """modelo(entrenamiento, h) del SARIMAX del miniproyecto 1.

    En cada origen se vuelve a estimar con los datos hasta el origen. Ajusta
    con los días hábiles de esos meses y pronostica con los de los h meses
    siguientes, que salen del calendario y no de la serie.
    """
    x = exogenas(serie)

    def modelo(entrenamiento, h):
        ajuste = SARIMAX(entrenamiento, exog=x.loc[entrenamiento.index], order=ORDEN,
                         seasonal_order=ORDEN_ESTACIONAL).fit(disp=False)
        ULTIMO["modelo"] = ajuste
        return ajuste.forecast(h, exog=x.loc[fechas_siguientes(entrenamiento, h)]).to_numpy()
    return modelo


def comparar_con_mp1(fila_clasico):
    """Imprime si la fila de referencia coincide con el miniproyecto 1."""
    diferencias = {m: round(fila_clasico[m] - v, 2) for m, v in METRICAS_MP1.items()}
    if all(d == 0 for d in diferencias.values()):
        print("  Coincide con el miniproyecto 1: RMSE {RMSE}, MAE {MAE}, MAPE {MAPE}"
              .format(**METRICAS_MP1))
    else:
        print(f"  Difiere del miniproyecto 1 {METRICAS_MP1}: diferencias {diferencias}")


def graficar_serie(serie):
    atipicos = detectar_atipicos(serie)
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.axvspan(pd.Timestamp("2008-10-01"), pd.Timestamp("2009-12-01"),
                color="#efefef", label="crisis bancaria 2008-2009")
    eje.plot(serie.index, serie.to_numpy(), color=AZUL, label="uso de tarjetas débito")
    eje.scatter(atipicos.index, atipicos["valor"], s=36, color=NARANJA, zorder=3,
                edgecolor="white", linewidth=1, label=f"atípicos (|z| > {UMBRAL:g}), se conservan")
    eje.axvline(serie.index[-N_PRUEBA], color=TINTA_SUAVE, linestyle="--", linewidth=1)
    eje.text(serie.index[-N_PRUEBA], 1.01, " meses del backtest →", color=TINTA_SUAVE,
             transform=eje.get_xaxis_transform(), va="bottom")
    eje.set_title("Uso mensual de tarjetas débito en Islandia (millones de ISK)")
    eje.set_ylabel(VALOR)
    eje.legend(loc="upper left")
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
# Se elige "dif12". La serie crece (la tendencia se multiplica por 2.7 entre
# 2000 y 2013) y su máximo, diciembre de 2012, cae dentro de los meses del
# backtest: un árbol que aprende el nivel no puede pronosticar por encima de lo
# que vio al entrenar. La diferencia a 12 quita la estacionalidad (la misma
# D = 1 del SARIMA) y deja una cantidad que sí se repite en el pasado.
# justificar_objetivo imprime los datos que sostienen la decisión.
OBJETIVO = "dif12"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes t con lo que se sabe antes de que t ocurra.

    - rezago_1 ... rezago_12: z(t-1) ... z(t-12), con shift(k).
    - mes, mes_sin, mes_cos: el mes del año, como número y en el círculo
      (para que diciembre y enero queden cerca en KNN, SVR y las redes).
    - media_3, media_12, desv_12: ventanas móviles sobre z corrida un mes,
      es decir, sobre z(t-1) hacia atrás.
    - dias_habiles, dif_dias_habiles: los días hábiles de t y su diferencia
      con los de t-12. Salen del calendario, no de la serie: se conocen de
      antemano, como la exógena del SARIMAX.
    Ninguna columna de la fila t usa z(t).
    """
    pasado = z.shift(1)
    columnas = {f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)}
    mes = z.index.month.to_numpy()
    columnas["mes"] = mes
    columnas["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    columnas["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    columnas["media_3"] = pasado.rolling(3).mean()
    columnas["media_12"] = pasado.rolling(12).mean()
    columnas["desv_12"] = pasado.rolling(12).std()
    habiles = dias_habiles(z.index).to_numpy()
    columnas["dias_habiles"] = habiles
    columnas["dif_dias_habiles"] = habiles - dias_habiles(z.index - pd.DateOffset(years=1)).to_numpy()
    return pd.DataFrame(columnas, index=z.index)


def filas_iguales(a, b):
    return np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def comprobar_sin_fuga(serie, tabla=tabla_supervisada):
    """Prueba en el código que ninguna fila ve el mes que predice.

    Para cada mes t se suma 1000 a y(t), se rearma la tabla y se exige que la
    fila t quede idéntica. Lanza AssertionError si alguna fila cambia. Para
    mostrar que la prueba sí detecta algo, cuenta además en cuántos meses la
    fila t+1 cambia (su rezago_1 lee el mes alterado). Devuelve ese conteo.
    """
    base = tabla(objetivo(serie))
    cambia_siguiente = 0
    for t in range(len(serie)):
        alterada = serie.copy()
        alterada.iloc[t] += 1000.0
        nueva = tabla(objetivo(alterada))
        if not filas_iguales(nueva.iloc[t], base.iloc[t]):
            raise AssertionError(f"la fila de {serie.index[t]:%Y-%m} cambia al alterar "
                                 "su propio mes: la tabla ve el mes que predice")
        if t + 1 < len(serie) and not filas_iguales(nueva.iloc[t + 1], base.iloc[t + 1]):
            cambia_siguiente += 1
    return cambia_siguiente


def tabla_con_fuga(z):
    """Control negativo: la misma tabla con una ventana que olvida shift(1)."""
    return tabla_supervisada(z).assign(media_3=z.rolling(3).mean())


def probar_tabla(serie):
    """Corre la prueba de fuga sobre la tabla real y sobre el control negativo."""
    cambia_siguiente = comprobar_sin_fuga(serie)
    print(f"  Prueba de fuga: se alteró cada uno de los {len(serie)} meses y ninguna "
          "fila cambió al alterar su propio mes.")
    print(f"  La fila siguiente sí cambió en {cambia_siguiente} de {len(serie) - 1} casos "
          f"(en los primeros {PERIODO} meses z(t) no existe: la diferencia a 12 necesita "
          "el año anterior).")
    try:
        comprobar_sin_fuga(serie, tabla_con_fuga)
    except AssertionError as error:
        print(f"  Control negativo: con media_3 sin shift(1) la prueba falla, como debe: {error}")
    else:
        raise RuntimeError("la prueba de fuga no detectó una ventana sin shift(1)")


def describir_tabla(serie):
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ultimo, _ = filas_completas(serie.iloc[:-PERIODO])
    print(f"  {X.shape[1]} columnas: {', '.join(X.columns)}")
    print(f"  Filas completas para entrenar: {len(X)} en el primer origen del backtest "
          f"({X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}) y {len(X_ultimo)} en el último. "
          f"Se pierden los primeros {len(serie.iloc[:-N_PRUEBA]) - len(X)} meses: "
          f"{PERIODO} por la diferencia a 12 y {REZAGOS} por los rezagos.")
    print(f"  Objetivo z = dif12: media {z.mean():.3f}, desviación {z.std():.3f}, "
          f"de {z.min():.3f} a {z.max():.3f}")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses.

    Para cada mes siguiente: se agrega a y como NaN, se arma su fila (que solo
    mira meses anteriores), se pronostica z, se lleva al nivel con
    y(t) = y(t-12) + z(t) y ese pronóstico se escribe como si fuera dato, para
    que entre como rezago del mes que sigue.
    """
    y = y.copy()
    for fecha in fechas_siguientes(y, h):
        extendida = pd.concat([y, pd.Series([np.nan], index=[fecha])])
        fila = tabla_supervisada(objetivo(extendida)).iloc[[-1]]
        z_hat = float(np.ravel(ajustado.predict(fila))[0])
        y.loc[fecha] = a_nivel(y, z_hat)   # y aún no tiene la fecha: y.iloc[-12] es y(t-12)
    return y.iloc[-h:].to_numpy()


def justificar_objetivo(serie):
    """Los datos que sostienen OBJETIVO = "dif12", sin tocar el backtest.

    1. Cuántos meses del backtest superan el máximo visto antes de él.
    2. Un experimento con los meses anteriores al backtest: se entrena hasta
       12 meses antes del primer origen y se pronostican, en modo recursivo,
       esos 12 meses con Random Forest y XGBoost, aprendiendo el nivel y la
       diferencia a 12.
    """
    global OBJETIVO
    antes, prueba = serie.iloc[:-N_PRUEBA], serie.iloc[-N_PRUEBA:]
    print(f"  Máximo antes del backtest: {antes.max():.3f} ({antes.idxmax():%Y-%m}); "
          f"en los meses del backtest: {prueba.max():.3f} ({prueba.idxmax():%Y-%m}). "
          f"{int((prueba > antes.max()).sum())} de los {N_PRUEBA} meses del backtest "
          "superan todo lo visto antes.")
    entrenamiento, validacion = antes.iloc[:-PERIODO], antes.iloc[-PERIODO:]
    elegido, filas_exp = OBJETIVO, []
    try:
        for opcion in ("nivel", "dif12"):
            OBJETIVO = opcion
            for nombre, fabrica in (("Random Forest", bosque), ("XGBoost", xgboost)):
                pronostico = con_tabla(fabrica)(entrenamiento, PERIODO)
                filas_exp.append({"objetivo": opcion, "modelo": nombre,
                                  "RMSE": round(rmse(validacion, pronostico), 3)})
    finally:
        OBJETIVO = elegido
    experimento = pd.DataFrame(filas_exp).pivot(index="modelo", columns="objetivo", values="RMSE")
    print(f"  Experimento: entrenar hasta {entrenamiento.index[-1]:%Y-%m} y pronosticar "
          f"{validacion.index[0]:%Y-%m} a {validacion.index[-1]:%Y-%m} (RMSE a 12 meses):")
    print(experimento.to_string())


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


# Dos a cuatro valores por hiperparámetro. KNN: cuántos meses parecidos se
# promedian. SVR: C (cuánto castiga los errores) y epsilon (el margen que no
# castiga, en unidades de z). Bosque y XGBoost: la profundidad de los árboles,
# que limita cuántas interacciones aprenden con ~116 filas; el mínimo de filas
# por hoja y la tasa de aprendizaje.
REJILLAS = {
    "KNN": (knn, {"n_neighbors": [3, 5, 7, 10]}),
    "SVR": (svr, {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.3]}),
    "Random Forest": (bosque, {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1, 0.3]}),
}
N_PLIEGUES = 5


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """La combinación de menor RMSE en validación temporal para cada modelo.

    La tabla se arma con serie.iloc[:-N_PRUEBA]: ninguno de los 24 meses del
    backtest participa. TimeSeriesSplit parte esas filas en pliegues que
    siempre validan con meses posteriores a los de entrenamiento. Guarda
    resultados/rejillas.csv con todas las combinaciones.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    validacion = TimeSeriesSplit(n_splits=N_PLIEGUES)
    pliegues = list(validacion.split(X))
    print(f"  TimeSeriesSplit({N_PLIEGUES}) sobre las {len(X)} filas anteriores al backtest "
          f"({X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}): entrenamiento de "
          f"{len(pliegues[0][0])} a {len(pliegues[-1][0])} filas, validación de "
          f"{len(pliegues[0][1])} filas en cada pliegue; el último valida "
          f"{X.index[pliegues[-1][1][0]]:%Y-%m} a {X.index[pliegues[-1][1][-1]]:%Y-%m}.")
    print("  RMSE de validación a un paso, en unidades de z (millones de ISK):")
    elegidos, filas_rejilla = {}, []
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        modelo = fabrica()
        prefijo = modelo.steps[-1][0] + "__" if hasattr(modelo, "steps") else ""
        busqueda = GridSearchCV(modelo, {prefijo + clave: valores for clave, valores in rejilla.items()},
                                cv=validacion, scoring="neg_root_mean_squared_error", n_jobs=1)
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        resultados = busqueda.cv_results_
        for parametros, media, desv, puesto in zip(resultados["params"], resultados["mean_test_score"],
                                                   resultados["std_test_score"],
                                                   resultados["rank_test_score"]):
            filas_rejilla.append({"modelo": nombre, "hiperparametros": str(sin_prefijo(parametros)),
                                  "RMSE_validacion": round(-media, 4), "desviacion": round(desv, 4),
                                  "elegido": puesto == 1})
        print(f"    {nombre}: {elegidos[nombre]}, RMSE {-busqueda.best_score_:.3f} "
              f"(de {len(resultados['params'])} combinaciones)")
    pd.DataFrame(filas_rejilla).to_csv(RESULTADOS / "rejillas.csv", index=False)
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost, ajustados con
    lo elegido sobre las filas anteriores al backtest. Guarda
    figuras/importancia.png y resultados/importancia.csv."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    importancias = pd.DataFrame({
        "Random Forest": bosque(**elegidos["Random Forest"]).fit(X, z).feature_importances_,
        "XGBoost": xgboost(**elegidos["XGBoost"]).fit(X, z).feature_importances_,
    }, index=X.columns)
    importancias = importancias.loc[importancias.mean(axis=1).sort_values().index]
    importancias[::-1].round(4).to_csv(RESULTADOS / "importancia.csv", index_label="variable")
    for nombre in importancias.columns:
        primeras = importancias[nombre].sort_values(ascending=False).head(5)
        print(f"  Importancia, {nombre}: "
              + ", ".join(f"{v} {p:.3f}" for v, p in primeras.items()))

    figura, ejes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    medidas = {"Random Forest": "reducción de impureza (suma 1)",
               "XGBoost": "ganancia promedio por división (suma 1)"}
    for eje, nombre in zip(ejes, importancias.columns):
        eje.barh(importancias.index, importancias[nombre], height=0.6, color=AZUL)
        eje.set_title(nombre)
        eje.set_xlabel(medidas[nombre])
        eje.grid(axis="y", visible=False)
    figura.suptitle("Qué variables usan los ensambles (ordenadas por la importancia promedio)",
                    x=0.01, ha="left", fontweight="bold", fontsize=11)
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    plt.close(figura)


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01
# En Colab con GPU la red y sus tensores van a cuda; sin GPU, a la CPU. Red se
# encarga de los dos: las clases de cada red no tienen que mover nada.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """Una capa nn.LSTM sobre la secuencia (lote, 12, 1), del rezago más viejo
    al más reciente, y una capa lineal que lee el último estado oculto."""

    def __init__(self, entradas=1, ocultas=16):
        super().__init__()
        self.lstm = nn.LSTM(entradas, ocultas, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        salida, _ = self.lstm(x)
        return self.salida(salida[:, -1, :])


# La tercera red es una Conv1D. Recibe la misma secuencia que la LSTM, pero con
# 8 filtros de 3 meses tiene unos 110 parámetros, diez veces menos que la
# LSTM, para entrenar con ~116 filas; y un filtro de 3 meses mira la misma
# dinámica de corto plazo que el AR(2) del SARIMAX.
class Conv1D(nn.Module):
    """nn.Conv1d con `filtros` filtros de tamaño `kernel` sobre los 12 rezagos,
    ReLU, y una capa lineal sobre todas las salidas de los filtros."""

    def __init__(self, entradas=1, filtros=8, kernel=3):
        super().__init__()
        self.capas = nn.Sequential(nn.Conv1d(entradas, filtros, kernel), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(filtros * (REZAGOS - kernel + 1), 1))

    def forward(self, x):
        # Red entrega (lote, 12, 1); Conv1d espera (lote, canales, pasos).
        return self.capas(x.transpose(1, 2))


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (16 ocultas)": (lambda n: LSTM(n, 16), True),
    "Conv1D (8 filtros)": (lambda n: Conv1D(n, 8, 3), True),
}


def validar_redes(serie):
    """Las épocas de cada red, elegidas con validación temporal.

    La tabla se arma con serie.iloc[:-N_PRUEBA], así que ninguno de los 24
    meses del backtest participa, y sus últimos 12 meses son la validación.
    Cada candidata se entrena EPOCAS_MAXIMAS épocas y sus épocas son las de
    menor pérdida de validación. De los dos MLP queda el de menor pérdida de
    validación. Guarda figuras/perdida.png y resultados/validacion_redes.csv,
    y devuelve {nombre: épocas} del mejor MLP, la LSTM y la Conv1D.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    X_ent, z_ent = X.iloc[:-PERIODO], z.iloc[:-PERIODO]
    X_val, z_val = X.iloc[-PERIODO:], z.iloc[-PERIODO:]
    print(f"  Validación: entrenamiento con {len(X_ent)} filas ({X_ent.index[0]:%Y-%m} a "
          f"{X_ent.index[-1]:%Y-%m}), validación con {len(X_val)} ({X_val.index[0]:%Y-%m} a "
          f"{X_val.index[-1]:%Y-%m}); hasta {EPOCAS_MAXIMAS} épocas, pérdida MSE en "
          "unidades estandarizadas de z.")
    curvas, filas_val = {}, []
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        ajustada = Red(construir, secuencia).fit(X_ent, z_ent, X_val, z_val)
        perdidas = np.array(ajustada.perdidas)
        epocas = int(np.argmin(perdidas[:, 1])) + 1
        curvas[nombre] = (perdidas, epocas)
        filas_val.append({"red": nombre,
                          "parametros": sum(p.numel() for p in ajustada.red.parameters()),
                          "epocas": epocas,
                          "perdida_entrenamiento": round(perdidas[epocas - 1, 0], 4),
                          "perdida_validacion": round(perdidas[epocas - 1, 1], 4),
                          "perdida_validacion_final": round(perdidas[-1, 1], 4)})
    tabla = pd.DataFrame(filas_val)
    tabla.to_csv(RESULTADOS / "validacion_redes.csv", index=False)
    for f in filas_val:
        print(f"    {f['red']}: {f['parametros']} parámetros, {f['epocas']} épocas; pérdida de "
              f"validación {f['perdida_validacion']:.3f} (en la época {EPOCAS_MAXIMAS}: "
              f"{f['perdida_validacion_final']:.3f})")

    mlps = tabla[tabla["red"].str.startswith("MLP")]
    mejor_mlp = mlps.loc[mlps["perdida_validacion"].idxmin(), "red"]
    print(f"  MLP elegido: {mejor_mlp}")
    graficar_perdidas(curvas)
    return {nombre: curvas[nombre][1] for nombre in CANDIDATAS
            if not nombre.startswith("MLP") or nombre == mejor_mlp}


def graficar_perdidas(curvas):
    """Un panel por candidata con las pérdidas de entrenamiento y validación por
    época, y una línea vertical en la época elegida."""
    # Cada panel con su propia escala: el MLP grande baja la pérdida de
    # entrenamiento varios órdenes más que los demás y aplastaría sus curvas.
    figura, ejes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    for eje, (nombre, (perdidas, epocas)) in zip(ejes.ravel(), curvas.items()):
        x = np.arange(1, len(perdidas) + 1)
        eje.plot(x, perdidas[:, 0], color=AZUL, label="entrenamiento")
        eje.plot(x, perdidas[:, 1], color=NARANJA, label="validación")
        eje.axvline(epocas, color=TINTA_SUAVE, linestyle="--", linewidth=1)
        eje.text(epocas, 0.98, f" {epocas} épocas", color=TINTA_SUAVE, va="top",
                 transform=eje.get_xaxis_transform())
        eje.set_yscale("log")
        eje.set_title(nombre)
    for eje in ejes[-1]:
        eje.set_xlabel("época")
    for eje in ejes[:, 0]:
        eje.set_ylabel("MSE (z estandarizada)")
    ejes[0, 0].legend(loc="lower left")
    figura.suptitle("Curvas de pérdida: la línea marca la época de menor pérdida de validación",
                    x=0.01, ha="left", fontweight="bold", fontsize=11)
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


SEMILLAS = range(2026, 2031)


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest completo.

    Guarda resultados/semillas.csv con una fila por semilla, imprime el RMSE
    de cada una y devuelve la media y la desviación (ddof=1) del RMSE.
    """
    filas_semillas = []
    for semilla in SEMILLAS:
        inicio = time.perf_counter()
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, semilla)))
        filas_semillas.append({"red": nombre, "epocas": epocas, "semilla": semilla,
                               "RMSE": round(promedio["RMSE"], 4), "MAE": round(promedio["MAE"], 4),
                               "MAPE": round(promedio["MAPE"], 4),
                               "segundos": round(time.perf_counter() - inicio, 1)})
    tabla = pd.DataFrame(filas_semillas)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    media, desv = tabla["RMSE"].mean(), tabla["RMSE"].std(ddof=1)
    print(f"  {nombre}, {epocas} épocas, con {len(tabla)} semillas: RMSE "
          + ", ".join(f"{f['RMSE']:.3f} ({f['semilla']})" for f in filas_semillas))
    print(f"  Media {media:.3f}, desviación {desv:.3f} (ddof=1)")
    return media, desv


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    inicio = time.perf_counter()
    _, promedio = backtest(serie, modelo)
    segundos = time.perf_counter() - inicio
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    print("1. La serie y el modelo a superar")
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"  Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos; mínimo "
          f"{cruda.min():.3f} ({cruda.idxmin():%Y-%m}), máximo {cruda.max():.3f} "
          f"({cruda.idxmax():%Y-%m})")
    serie = preparar(cruda)
    graficar_serie(serie)
    habiles = dias_habiles(serie.index) + 21
    print(f"  Días hábiles por mes (calendario de Islandia): de {habiles.min():.0f} "
          f"a {habiles.max():.0f}")
    primero = len(serie) - N_PRUEBA
    print(f"  Backtest: {N_PRUEBA - PERIODO + 1} orígenes, de "
          f"{serie.index[primero - 1]:%Y-%m} a {serie.index[-PERIODO - 1]:%Y-%m}; "
          f"meses pronosticados: {serie.index[primero]:%Y-%m} a {serie.index[-1]:%Y-%m}")

    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        comparar_con_mp1(filas[-1])
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        describir_tabla(serie)
        probar_tabla(serie)
        justificar_objetivo(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            media, desv = repetir_con_semillas(serie, mejor, epocas[mejor])
            rmse_clasico = next((f["RMSE"] for f in filas if f["modelo"] == NOMBRE_CLASICO), None)
            if rmse_clasico is not None:
                distancia = media - rmse_clasico
                lectura = ("mayor que la desviación entre semillas ({:.3f}): la diferencia no se "
                           "explica por la semilla" if abs(distancia) > desv else
                           "menor que la desviación entre semillas ({:.3f}): la diferencia cabe "
                           "en lo que varía la red con la semilla").format(desv)
                print(f"  Distancia con el SARIMAX: {media:.3f} - {rmse_clasico:.2f} = "
                      f"{distancia:+.3f}, {lectura}.")
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
