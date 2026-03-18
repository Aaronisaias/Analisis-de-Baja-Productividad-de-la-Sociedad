import pandas as pd


def leer_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Lee el archivo CSV y lo carga en la variable principal dt.
    """
    dt_local = pd.read_csv(ruta_csv)
    return dt_local


def limpiar_y_traducir_columnas(dt: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas básicas de limpieza y renombra columnas al español.
    """
    dt_limpio = dt.copy()

    # Eliminar filas con valores nulos
    dt_limpio = dt_limpio.dropna()

    # Renombrar columnas de inglés a español para mayor claridad
    columnas_renombradas = {
        "user_id": "id_usuario",
        "age": "edad",
        "gender": "genero",
        "occupation": "profesion",
        "daily_screen_time_hours": "horas_pantalla_diarias",
        "phone_usage_before_sleep_minutes": "minutos_movil_antes_dormir",
        "sleep_duration_hours": "horas_sueno",
        "sleep_quality_score": "puntaje_calidad_sueno",
        "stress_level": "nivel_estres",
        "caffeine_intake_cups": "tazas_cafeina",
        "physical_activity_minutes": "minutos_actividad_fisica",
        "notifications_received_per_day": "notificaciones_por_dia",
        "mental_fatigue_score": "puntaje_fatiga_mental",
    }

    dt_limpio = dt_limpio.rename(columns=columnas_renombradas)

    return dt_limpio


def crear_rangos_edad(dt: pd.DataFrame) -> pd.DataFrame:
    """
    Crea rangos de edad para los análisis por grupo etario.
    """
    bins_edad = [0, 24, 34, 44, 54, 64, 120]
    labels_edad = [
        "0-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
    ]
    dt["rango_edad"] = pd.cut(dt["edad"], bins=bins_edad, labels=labels_edad, right=True)
    return dt


def crear_rangos_horas_sueno(dt: pd.DataFrame) -> pd.DataFrame:
    """
    Crea rangos de horas de sueño para relacionar duración con otros indicadores.
    """
    bins_sueno = [0, 5, 7, 9, 24]
    labels_sueno = [
        "0-5h",
        "5-7h",
        "7-9h",
        "9h+",
    ]
    dt["rango_horas_sueno"] = pd.cut(
        dt["horas_sueno"], bins=bins_sueno, labels=labels_sueno, right=True
    )
    return dt


def analisis_1_promedio_horas_sueno(dt: pd.DataFrame) -> float:
    """
    1. ¿Cuál es el promedio de sueño de las personas en general?
    """
    return dt["horas_sueno"].mean()


def analisis_2_porcentaje_uso_dispositivos(dt: pd.DataFrame) -> float:
    """
    2. ¿Cuál es el % del uso de los dispositivos?

    Suposición:
    Interpretamos el "% de uso de los dispositivos" como el
    porcentaje promedio del día que las personas pasan frente
    a pantallas: (horas_pantalla_diarias / 24) * 100.
    """
    porcentaje_diario = (dt["horas_pantalla_diarias"] / 24.0) * 100.0
    return porcentaje_diario.mean()


def analisis_3_promedio_uso_movil_antes_dormir(dt: pd.DataFrame) -> float:
    """
    3. ¿Cuál es el promedio del uso del móvil antes de dormir (minutos)?
    """
    return dt["minutos_movil_antes_dormir"].mean()


def analisis_4_profesion_mas_uso_dispositivos(dt: pd.DataFrame) -> pd.Series:
    """
    4. ¿Qué profesión tiene más uso de los dispositivos?

    Usamos el promedio de horas_pantalla_diarias por profesión.
    """
    promedio_por_profesion = (
        dt.groupby("profesion")["horas_pantalla_diarias"].mean().sort_values(ascending=False)
    )
    return promedio_por_profesion


def analisis_5_promedio_por_rango_edad_pantalla_y_sueno(dt: pd.DataFrame) -> pd.DataFrame:
    """
    5. ¿Cuál es el promedio por rango de edad que más uso de la pantalla tiene
       y menos horas de sueño?
    """
    tabla = (
        dt.groupby("rango_edad")[["horas_pantalla_diarias", "horas_sueno"]]
        .mean()
        .sort_values(by=["horas_pantalla_diarias", "horas_sueno"], ascending=[False, True])
    )
    return tabla


def analisis_6_estres_por_rango_edad(dt: pd.DataFrame) -> pd.DataFrame:
    """
    6. ¿Cuál es el porcentaje que más tiene nivel de estrés por rango de edad?

    Definimos "alto estrés" como nivel_estres >= 8 (en escala original del dataset).
    Calculamos:
    - porcentaje de personas con alto estrés por rango_edad
    - nivel de estrés promedio por rango_edad
    """
    dt_local = dt.copy()
    dt_local["alto_estres"] = dt_local["nivel_estres"] >= 8

    resumen = (
        dt_local.groupby("rango_edad")
        .agg(
            cantidad=("nivel_estres", "count"),
            alto_estres_cantidad=("alto_estres", "sum"),
            nivel_estres_promedio=("nivel_estres", "mean"),
        )
    )
    resumen["porcentaje_alto_estres"] = (
        resumen["alto_estres_cantidad"] / resumen["cantidad"] * 100.0
    )
    resumen = resumen.sort_values(
        by=["porcentaje_alto_estres", "nivel_estres_promedio"], ascending=[False, False]
    )

    return resumen


def analisis_7_profesion_mayor_estres(dt: pd.DataFrame) -> pd.Series:
    """
    7. ¿Cuál es la profesión que más nivel de estrés tiene en promedio?
    """
    promedio_estres_profesion = (
        dt.groupby("profesion")["nivel_estres"].mean().sort_values(ascending=False)
    )
    return promedio_estres_profesion


def analisis_8_puntaje_sueno_por_rango_duracion(dt: pd.DataFrame) -> pd.DataFrame:
    """
    8. ¿Cuál es el promedio de puntuación de sueño por rango de duración de horas de sueño?
    """
    tabla = (
        dt.groupby("rango_horas_sueno")["puntaje_calidad_sueno"]
        .mean()
        .sort_values(ascending=False)
    )
    return tabla


def analisis_9_cafeina_por_rango_sueno(dt: pd.DataFrame) -> pd.DataFrame:
    """
    9. ¿Cuál es el promedio de las tazas de cafeína por rangos de horas diarias de sueño?
    """
    tabla = (
        dt.groupby("rango_horas_sueno")["tazas_cafeina"]
        .mean()
        .sort_values(ascending=False)
    )
    return tabla


def analisis_10_profesion_mayor_productividad(dt: pd.DataFrame) -> pd.Series:
    """
    10. ¿Qué profesión tiene mayor productividad?

    Suposición:
    No existe una columna directa de "productividad".
    Creamos un índice de productividad basado en:
    - mayor puntaje_calidad_sueno (mejor descanso)
    - menor puntaje_fatiga_mental (menos cansancio mental)
    - menor nivel_estres (menos estrés)

    Normalizamos de forma simple (z-score aproximado con estandarización lineal)
    y combinamos en un índice.
    """
    dt_local = dt.copy()

    for col in ["puntaje_calidad_sueno", "puntaje_fatiga_mental", "nivel_estres"]:
        media = dt_local[col].mean()
        std = dt_local[col].std()
        if std == 0:
            dt_local[f"z_{col}"] = 0.0
        else:
            dt_local[f"z_{col}"] = (dt_local[col] - media) / std

    # Índice de productividad (mayor es mejor)
    # + calidad de sueño, - fatiga, - estrés
    dt_local["indice_productividad"] = (
        dt_local["z_puntaje_calidad_sueno"]
        - dt_local["z_puntaje_fatiga_mental"]
        - dt_local["z_nivel_estres"]
    )

    productividad_por_profesion = (
        dt_local.groupby("profesion")["indice_productividad"]
        .mean()
        .sort_values(ascending=False)
    )

    return productividad_por_profesion


def exportar_excel(dt: pd.DataFrame, ruta_salida: str) -> None:
    """
    Exporta la tabla principal ya limpia a un archivo XLSX.
    """
    dt.to_excel(ruta_salida, index=False)


def main():
    ruta_csv = "sleep_mobile_stress_dataset_15000.csv"
    ruta_xlsx = "Data_analysis.xlsx"

    print("=== LECTURA Y LIMPIEZA DE DATOS ===")
    global dt
    dt = leer_datos(ruta_csv)
    dt = limpiar_y_traducir_columnas(dt)
    dt = crear_rangos_edad(dt)
    dt = crear_rangos_horas_sueno(dt)

    print("Filas después de limpieza:", len(dt))
    print("Columnas:", list(dt.columns))
    print()

    # 1. Promedio de horas de sueño
    print("1) PROMEDIO DE SUEÑO (HORAS)")
    promedio_sueno = analisis_1_promedio_horas_sueno(dt)
    print(promedio_sueno)
    print()

    # 2. % de uso de dispositivos (porcentaje del día frente a pantallas)
    print("2) % PROMEDIO DEL DÍA USANDO DISPOSITIVOS (PANTALLA)")
    porcentaje_uso = analisis_2_porcentaje_uso_dispositivos(dt)
    print(porcentaje_uso)
    print()

    # 3. Promedio de minutos de móvil antes de dormir
    print("3) PROMEDIO MINUTOS DE MOVIL ANTES DE DORMIR")
    promedio_minutos_movil = analisis_3_promedio_uso_movil_antes_dormir(dt)
    print(promedio_minutos_movil)
    print()

    # 4. Profesión con más uso de dispositivos
    print("4) PROFESIÓN CON MÁS USO DE DISPOSITIVOS (HORAS DE PANTALLA DIARIAS PROMEDIO)")
    profesion_pantalla = analisis_4_profesion_mas_uso_dispositivos(dt)
    print(profesion_pantalla)
    print()

    # 5. Promedio por rango de edad: uso de pantalla y horas de sueño
    print("5) PROMEDIO POR RANGO DE EDAD: HORAS DE PANTALLA Y HORAS DE SUEÑO")
    edad_pantalla_sueno = analisis_5_promedio_por_rango_edad_pantalla_y_sueno(dt)
    print(edad_pantalla_sueno)
    print()

    # 6. Porcentaje con mayor nivel de estrés por rango de edad
    print("6) PORCENTAJE CON MAYOR NIVEL DE ESTRÉS POR RANGO DE EDAD")
    estres_rango_edad = analisis_6_estres_por_rango_edad(dt)
    print(estres_rango_edad)
    print()

    # 7. Profesión con mayor nivel de estrés promedio
    print("7) PROFESIÓN CON MAYOR NIVEL DE ESTRÉS (PROMEDIO)")
    profesion_estres = analisis_7_profesion_mayor_estres(dt)
    print(profesion_estres)
    print()

    # 8. Promedio de puntaje de sueño por rango de horas de sueño
    print("8) PROMEDIO DE PUNTAJE DE CALIDAD DE SUEÑO POR RANGO DE HORAS DE SUEÑO")
    puntaje_por_rango_sueno = analisis_8_puntaje_sueno_por_rango_duracion(dt)
    print(puntaje_por_rango_sueno)
    print()

    # 9. Promedio de tazas de cafeína por rango de horas de sueño
    print("9) PROMEDIO DE TAZAS DE CAFEÍNA POR RANGO DE HORAS DE SUEÑO")
    cafeina_por_rango_sueno = analisis_9_cafeina_por_rango_sueno(dt)
    print(cafeina_por_rango_sueno)
    print()

    # 10. Profesión con mayor productividad (índice compuesto)
    print("10) PROFESIÓN CON MAYOR PRODUCTIVIDAD (ÍNDICE COMPUESTO)")
    productividad_por_profesion = analisis_10_profesion_mayor_productividad(dt)
    print(productividad_por_profesion)
    print()

    # Exportar archivo limpio a Excel
    exportar_excel(dt, ruta_xlsx)
    print(f"Archivo limpio exportado a: {ruta_xlsx}")


if __name__ == "__main__":
    main()

