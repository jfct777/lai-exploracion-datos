# M27C: auditoría dirigida de gVCF en chr22

## Para qué sirve

M27C responde una pregunta que M27B no podía resolver: cuántas posiciones del modelo Gnomix están realmente evaluadas en NAMBR aunque no aparezcan en el VCF conjunto que contiene solo variantes. Un registro ausente en ese VCF puede significar que las 128 muestras son homocigotas para REF o que la región no tiene evidencia suficiente; los bloques de referencia de los gVCF permiten distinguir ambos casos.

Esta auditoría no ejecuta Gnomix, no simula cromosomas, no entrena modelos y no evalúa si las variantes raras mejoran LAI. Su salida solo decide si vale la pena pasar a la auditoría separada de donantes, parentesco y canal raro.

## Qué contará como posición lista

No basta con que una posición esté cubierta. Para contar dentro del 80% exigido por el modelo, debe cumplir cuatro condiciones:

1. el bloque o registro del gVCF cubre la posición;
2. el genotipo pasa la política de calidad;
3. REF/ALT son compatibles con el marcador congelado;
4. si el genotipo es heterocigoto, su fase está respaldada por una llamada exacta y concordante en el scaffold existente. Los homocigotos no necesitan ordenar haplotipos.

La fracción se calcula siempre sobre los 110.074 marcadores del modelo. El 80% publicado por Gnomix se refiere a posiciones compartidas con el modelo. Aquí le agrego controles más estrictos de calidad y fase para decidir si los 128 candidatos merecen pasar a la selección de donantes. Por eso un resultado favorable todavía no significa que el panel final esté listo: la fracción deberá recalcularse después de retirar relacionados y fijar qué individuos quedarán como donantes.

No se reemplazará el 80% por una cobertura ponderada por frecuencia ni por heterocigosidad.

Además, REF se comprobará contra el FASTA GRCh38 fijado por el proyecto. Se leerá únicamente chr22 mediante su índice; no hace falta transferir ni recorrer los 3,4 GB del genoma completo.

## Calidad y análisis de sensibilidad

La política principal será `GQ≥20`, profundidad efectiva `≥10` y al menos 95% de genotipos evaluables por marcador. En un bloque homocigoto de referencia se usará `MIN_DP`, porque representa la profundidad mínima observada dentro del bloque; en un registro de variante se usará `DP`.

Estos cortes tienen antecedentes en filtros de genotipos WGS, pero no fueron calibrados específicamente en NAMBR. Por eso se mantendrá un ancla fija y se cambiará una sola condición por vez:

- profundidad mínima: 8 y 15;
- GQ mínimo: 10 y 30;
- call rate: 90%, 99% y 100%. El último valor muestra el caso más estricto, sin ningún genotipo faltante entre los 128 candidatos.

No se elegirá después la combinación que conserve más marcadores. Se informará primero el resultado principal y luego todas las sensibilidades. Si la política principal pasa pero algún valor razonable cruza el 80%, el resultado se marcará como sensible al corte y deberá revisarse antes de gastar en otra corrida; no se cambiará silenciosamente de umbral. También se mostrará el techo estructural sin filtros para separar falta de cobertura de baja calidad.

## Información ancestral

Una posición lista para el programa puede aportar poca o ninguna información para distinguir ancestrías. Por eso, usando los 78 donantes congelados —26 AFR, 26 EUR y 26 NAM—, se calcularán por marcador las frecuencias alélicas de cada grupo, su incertidumbre y la diferencia observada entre grupos.

Se informará por separado cuántos marcadores son monomórficos, cuántos presentan diferencias entre ancestrías y cómo se distribuyen a lo largo de las ventanas reales del modelo. Esta descripción no cambia el gate del 80% y un `PASS` técnico no se presentará como prueba de utilidad biológica.

El scaffold contiene tanto muestras NAMBR como donantes usados por el baseline. Por eso, la fase que recuperemos desde ese panel es utilizable para preparar una entrada, pero no es validación independiente. El resultado separará los marcadores cuya fase es trivial por ser homocigotos de los que dependen del scaffold.

## Validaciones antes de aceptar el resultado

El parser tendrá casos sintéticos para bordes de bloques, huecos, alelos distintos, multialélicos, campos faltantes y fase. En datos reales deberá recuperar las 128 identidades de M27B y reproducir, en los marcadores compartidos con el scaffold, al menos 19.535 llamadas conjuntas y una concordancia mínima por muestra de 0,99.

También se comprobará explícitamente un caso dentro de un bloque `0/0` y un hueco real. Así evitamos volver a convertir automáticamente una ausencia en homocigoto de referencia.

## Ejecución y costo

Los gVCF están en `southamerica-east1`, mientras que el datalake del proyecto está en `us-central1`. Leerlos desde la región actual cuesta US$0,14 por GiB transferido. La corrida se enviará por eso a São Paulo y montará el bucket en solo lectura mediante Cloud Storage FUSE.

Los objetos están en clase Coldline. Aunque la lectura se haga en la misma región, Google cobra US$0,02 por GiB recuperado, además de las operaciones de lectura. Como control previo, los índices TBI de ocho muestras delimitan 0,523 GiB comprimidos para chr22 en conjunto. No es todavía el consumo facturado, pero permite reservar US$0,80 para recuperación, operaciones, imagen y los insumos pequeños que vienen de `us-central1`, además del cómputo.

Se usará una sola tarea con un máximo de 8 vCPU, 32 GB de RAM y hasta ocho lectores de gVCF en paralelo. Esto evita crear 128 máquinas y descargar 128 veces la misma imagen. Dentro del mismo smoke se compararán 1, 4 y 8 lectores sobre exactamente los mismos archivos; se escogerá la configuración más pequeña que quede a menos de 20% del mejor tiempo y mantenga la memoria por debajo de 70% de lo solicitado. Este ajuste solo decide recursos, no filtros científicos.

La estimación preliminar para smoke y corrida completa es de US$0,30 a US$1,50. No se lanzará la corrida completa si el smoke proyecta más de US$2 o más de tres horas, ni si el método empieza a copiar gVCF completos o a leerlos desde Norteamérica.

## Regla de parada

Si los inputs o el parser no pasan, se detiene. Si la fracción lista bajo la política principal queda por debajo del 80%, se cierra el uso del modelo congelado con estos activos. Si pasa pero depende del corte, primero se revisa esa sensibilidad. Si pasa de forma estable, el único paso autorizado es diseñar la auditoría de donantes, parentesco con PC-Relate sin KING y canal raro; todavía no se autoriza simulación, Gnomix ni entrenamiento.
