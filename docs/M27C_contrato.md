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

Los gVCF están en `southamerica-east1`, mientras que el datalake del proyecto está en `us-central1`. Leerlos desde la región actual cuesta US$0,14 por GiB transferido. La corrida se enviará por eso a São Paulo y accederá a los gVCF mediante Cloud Storage FUSE. Aunque la integración del sistema expone el montaje como escribible, el proceso abre estos archivos únicamente para lectura y no modifica los originales.

Los objetos están en clase Coldline. Aunque la lectura se haga en la misma región, Google cobra US$0,02 por GiB recuperado, además de las operaciones de lectura. Como control previo, los índices TBI de ocho muestras delimitan 0,523 GiB comprimidos para chr22 en conjunto. No es todavía el consumo facturado, pero permite reservar US$0,80 para recuperación, operaciones, imagen y los insumos pequeños que vienen de `us-central1`, además del cómputo.

Se usará una sola tarea con un máximo de 8 vCPU, 32 GB de RAM y hasta ocho lectores de gVCF en paralelo. Esto evita crear 128 máquinas y descargar 128 veces la misma imagen. Dentro del mismo smoke se compararán 1, 4 y 8 lectores sobre exactamente los mismos archivos; se escogerá la configuración más pequeña que quede a menos de 20% del mejor tiempo y mantenga la memoria por debajo de 70% de lo solicitado. Este ajuste solo decide recursos, no filtros científicos.

La estimación preliminar para smoke y corrida completa es de US$0,30 a US$1,50. No se lanzará la corrida completa si el smoke proyecta más de US$2 o más de tres horas, ni si el método empieza a copiar gVCF completos o a leerlos desde Norteamérica.

## Regla de parada

Si los inputs o el parser no pasan, se detiene. Si la fracción lista bajo la política principal queda por debajo del 80%, se cierra el uso del modelo congelado con estos activos. Si pasa pero depende del corte, primero se revisa esa sensibilidad. Si pasa de forma estable, el único paso autorizado es diseñar la auditoría de donantes, parentesco con PC-Relate sin KING y canal raro; todavía no se autoriza simulación, Gnomix ni entrenamiento.

## Resultado de la corrida

La corrida final `m27c-targeted-gvcf-chr22-20260814b` consultó las 110.074 posiciones de chr22 en los 128 gVCF, sin descargar archivos completos. Antes hice un control de recursos con ocho gVCF: uno, cuatro y ocho lectores tardaron 572,4, 166,4 y 158,9 segundos, respectivamente. Elegí cuatro lectores porque quedó a menos de 5% del mejor tiempo usando la mitad de los vCPU. La corrida completa tardó 79,8 minutos. Fue más lenta que la proyección inicial de 44 minutos porque las primeras ocho muestras ordenadas por nombre no representaban bien la distribución de tamaños y tiempos de todos los gVCF. En una corrida parecida, el control deberá tomar archivos pequeños, medianos y grandes, no simplemente los primeros de la lista.

El control de identidad pasó en las 128 muestras. Cada gVCF compartió entre 43.004 y 43.025 genotipos con el scaffold y la concordancia de dosis estuvo entre 0,9911 y 0,99995, por encima del mínimo preregistrado de 0,99. Esto confirma que se consultaron las muestras esperadas y que la codificación de genotipos coincide; no demuestra todavía que sean donantes parentales adecuados.

La cobertura estructural fue completa en las posiciones consultadas: no hubo sitios fuera de bloques o registros ni genotipos ausentes. De las 14.089.472 combinaciones muestra×marcador, 12.187.287 provinieron de bloques homocigotos de referencia, 1.901.531 de registros exactos de variante, 456 tuvieron alelos incompatibles y 198 correspondieron a otra ALT pero eran homocigotas para REF. Esta separación es importante porque permite distinguir una posición realmente evaluada de una ausencia en un VCF que solo guarda variantes.

Con la política principal —profundidad efectiva ≥10, GQ≥20 y al menos 122 de 128 muestras evaluables—, 108.274 marcadores (98,36%) pasaron calidad. Después de exigir además fase respaldada para todos los heterocigotos de alta calidad, quedaron 92.790 marcadores listos, equivalentes a 84,30% del modelo. Aquí “listo” significa utilizable para construir la entrada candidata; no significa que el marcador sea informativo para ancestría ni que el panel parental esté validado. Los homocigotos aportaron 13.081.813 llamadas con fase trivial, 668.867 heterocigotos tuvieron fase respaldada y 251.904 heterocigotos no pudieron respaldarse con el scaffold. Por eso la pérdida principal aparece al exigir fase, no por falta de cobertura.

### Sensibilidad a los filtros

| Política | Marcadores listos | Fracción | ¿Supera 80%? |
|---|---:|---:|:---:|
| Principal: DP≥10, GQ≥20, call rate≥95% | 92.790 | 84,30% | Sí |
| DP≥8 | 93.391 | 84,84% | Sí |
| DP≥15 | 87.855 | 79,81% | No |
| GQ≥10 | 92.954 | 84,45% | Sí |
| GQ≥30 | 92.142 | 83,71% | Sí |
| Call rate≥90% | 93.754 | 85,17% | Sí |
| Call rate≥99% | 81.595 | 74,13% | No |
| Call rate=100% | 60.810 | 55,24% | No |

El resultado principal pasa, pero no es robusto a todos los valores razonables: DP≥15 queda apenas 205 marcadores por debajo del mínimo de 88.060, mientras que exigir 127 o 128 muestras por posición reduce bastante la fracción. No voy a escoger el filtro que más convenga después de ver estas cifras. Mantengo la política preregistrada como principal y clasifico el resultado como `PASS_THRESHOLD_SENSITIVE`, es decir, factible pero sensible al corte.

El 84,30% global tampoco está repartido de manera uniforme. El modelo tiene 370 ventanas reales; 24,86% de ellas queda por debajo de 80% de marcadores listos y el mayor hueco sin un marcador listo alcanza 772.792 bp, equivalentes a 2,1845 cM. Esto no invalida la auditoría, pero obliga a conservar el diagnóstico por ventana: para detectar bordes de ancestría local importa dónde están los huecos, no solo cuántos marcadores sobreviven en todo el cromosoma.

Entre los 78 donantes congelados del baseline, 79.770 marcadores mostraron alguna diferencia de frecuencia observada entre AFR, EUR y NAM; 64.378 coincidieron además con el conjunto listo. Esta etiqueta solo significa que la mayor y la menor frecuencia observadas no fueron idénticas. No es una prueba estadística, no impone un tamaño mínimo de diferencia y no constituye validación biológica independiente.

## Corrección del control de encabezados

La salida completa marcó inicialmente C0 como `FAIL`, aunque C1–C3 pasaron. La causa fue un error del auditor: buscaba líneas que empezaran por `chr22`, por lo que un contig alternativo como `chr22_KI…` podía sobrescribir el contig canónico durante la lectura del encabezado. Corregí la comparación para aceptar únicamente el ID exacto `chr22`, añadí pruebas de regresión y ejecuté una auditoría de encabezados sobre el mismo manifiesto de 128 gVCF. Los 128 contienen `chr22` con longitud 50.818.468, una sola muestra y todos los campos requeridos.

La reconciliación conserva intactos C1–C3 de la corrida completa y sustituye solo C0 con esa comprobación dirigida. Los hashes del resumen, el contrato de entrada y el manifiesto de gVCF quedaron incluidos en el recibo, de modo que no se están mezclando corridas ni muestras. El resultado reconciliado deja C0–C3 en `PASS`, pero mantiene explícitamente `final_donor_panel_certified=false`.

## Interpretación y siguiente paso permitido

M27C resuelve la duda técnica que dejó M27B: los 128 gVCF sí permiten recuperar más del 80% de las posiciones necesarias bajo la política principal. No prueba que NAMBR sea una población indígena parental pura, que los individuos sean independientes, que cada marcador distinga ancestrías ni que las variantes raras mejoren LAI.

El siguiente paso es pequeño y separado: auditar el conjunto real de candidatos, construir PCs de ancestría que no estén dominados por familias, ejecutar PC-Relate sin KING y comprobar disjunción con el baseline. Después se recalcularán la fracción global, la fase y los huecos usando solo los donantes finales. Si ese panel cae por debajo de 80%, no queda bien separado o no permite estimar parentesco sin circularidad, el piloto se detiene. Hasta pasar esos controles siguen bloqueados Gnomix, la simulación de mosaicos, el entrenamiento y TEST.

La secuencia completa —control de recursos, intento fallido corto, corrida final y auditoría de encabezados— se mantiene por debajo del techo autorizado de US$2. El costo exacto depende de la recuperación Coldline y aparecerá después en facturación; con los tiempos y volúmenes observados, la estimación razonable es aproximadamente US$0,7–1,2.
