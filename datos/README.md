# Datos

La serie del grupo está en esta carpeta, en CSV: la misma del miniproyecto 1.

| Campo | Valor |
|---|---|
| Archivo | `debito_islandia.csv` |
| Qué mide | Uso mensual de tarjetas débito en el comercio minorista de Islandia: valor total de las compras hechas con tarjeta débito en el mes, en millones de coronas islandesas (ISK) a precios corrientes, según la documentación de la fuente |
| Fuente | Statistics Iceland (Hagstofa Íslands), <https://statice.is>. Se tomó de la copia que distribuye el paquete de R `fpp2` (Hyndman, *Forecasting: Principles and Practice*, 2.ª ed., conjunto `debitcards`), publicada en CSV por el repositorio Rdatasets: <https://github.com/vincentarelbundock/Rdatasets/blob/master/csv/fpp2/debitcards.csv> |
| Licencia | Datos de Statistics Iceland: Creative Commons Atribución 4.0 Internacional (CC BY 4.0), con crédito a Statistics Iceland (<https://statice.is/publications/open-data-access>). Las copias de `fpp2` y Rdatasets se distribuyen con licencia GPL-3 |
| Frecuencia | mensual |
| Periodo | 2000-01 a 2013-08 (164 observaciones) |
| Columna de fecha | `fecha` (primer día del mes, formato `AAAA-MM-DD`) |
| Columna que se pronostica | `uso_debito` (millones de ISK) |

## Cómo se construyó el CSV

El archivo original de Rdatasets tiene las columnas `rownames`, `time` (año
decimal: 2000, 2000.083, …) y `value`. Se convirtió `time` a la fecha del
primer día de cada mes, empezando en enero de 2000, y `value` se renombró
`uso_debito`. No se cambió ningún valor.

## Huecos y datos atípicos

- **Huecos:** ninguno; los 164 meses tienen dato. `preparar` en
  `miniproyecto.py` de todas formas interpola linealmente cualquier mes vacío.
- **Atípicos:** el guion marca 10 meses cuyo residuo de la STL robusta supera
  3 desviaciones robustas (3 × 1.4826 × MAD), concentrados en 2008, el año de
  la crisis bancaria islandesa. Se conservan sin cambios porque son meses
  reales, no errores de captura. La lista queda en `resultados/atipicos.csv`.

## Escala

Los valores van de 7.2 a 26.7, una escala cómoda para estimar los modelos,
así que no se reescalaron.

## Días hábiles

La variable exógena del SARIMAX y dos columnas de la tabla supervisada
(`dias_habiles` y `dif_dias_habiles`) son los días hábiles de cada mes: de
lunes a viernes, menos los festivos oficiales de Islandia. No están en el CSV:
el guion los calcula a partir del calendario, así que se conocen de antemano y
no usan ningún dato de la serie.

- En el miniproyecto 1 los festivos salían de la librería `holidays`, que el
  miniproyecto 2 no permite. Aquí se calculan en Python puro
  (`festivos_islandia` en `miniproyecto.py`): los 5 festivos de fecha fija
  (Año Nuevo, Día del Trabajo, Día Nacional, Navidad y San Esteban), los 7
  que dependen de la Pascua (Jueves y Viernes Santo, Pascua, lunes de Pascua,
  Ascensión, Pentecostés y lunes de Pentecostés), el primer día del verano
  (primer jueves después del 18 de abril) y el día del comercio (primer lunes
  de agosto). La Pascua sale del algoritmo gregoriano de Meeus, Jones y
  Butcher.
- Nochebuena y Nochevieja son medio día hábil y `holidays` no las cuenta como
  festivo, así que aquí tampoco.
- Se comprobó contra `holidays` mes a mes: los 176 meses (los 164 de la serie
  y los 12 siguientes) dan el mismo número de días hábiles.
- Van de 16 a 23 días por mes. En el modelo entran centrados en 21, para que
  el coeficiente se lea como el efecto de un día hábil más.
