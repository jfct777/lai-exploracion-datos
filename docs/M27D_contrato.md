# M27D: parentesco y disjunción de donantes sin KING

**Estado:** preparación y benchmark técnico cerrados el 14 de agosto de 2026. La auditoría de
parentesco completa todavía no está implementada ni autorizada. Los valores que pueden cambiar la
conclusión científica no se elegirán después de mirar qué configuración conserva más donantes.

## Para qué sirve

M27D responde una pregunta concreta antes de simular: cuántos candidatos parentales realmente
independientes quedan después de comprobar parentesco reciente y solapamiento con los 78 donantes del
baseline. No evalúa todavía si las variantes raras mejoran el LAI, no ejecuta Gnomix y no entrena ningún
modelo.

La estimación de parentesco se hará con SNPs comunes de los 22 autosomas. Ese detalle es intencional:
los alelos raros son el canal que queremos estudiar después, pero no son una base neutral para separar
parentesco reciente de la similitud producida por drift, endogamia o historia compartida dentro de una
población indígena. PC-Relate usa componentes principales para retirar esa estructura poblacional antes
de estimar el parentesco familiar reciente.

## Qué datos entran

El universo principal es el panel faseado oficial usado por M27:

`gs://projects-usp/nam-diversity/shapeit/phased/natwgs.1000G.sgdp.hgdp.andamanese.hg38.<chr>.norm.PHASED.vcf.gz`

Son 3.685 muestras y 22 VCF autosómicos que suman 4,31 GiB. La metadata es:

`gs://projects-usp/nam-diversity/natwgs.1000G.sgdp.hgdp.hg38/metadata_complete_revised.txt`

También entran los 22 VCF del baseline:

`gs://projects-usp/dna-do-brasil/dnabr-lai-gnomix/vcf_fixed/dnabr.refpop.fixed.chr<1-22>.vcf.gz`

Estos últimos suman 219 MiB y contienen 78 donantes: 26 AFR, 26 EUR y 26 NAM. Setenta y siete ya
aparecen por identidad resuelta dentro del panel oficial. Esas copias no se duplicarán; el donante que
falta se añadirá al universo de parentesco usando los marcadores autosómicos compatibles. Antes de
aceptarlo se comprobarán REF/ALT, identidad de las 77 coincidencias y tasa de genotipo del donante
añadido. Cada coincidencia deberá tener al menos 10.000 genotipos autosómicos conjuntos y concordancia
de dosis de 0,99 o más; el donante añadido deberá superar 98% de genotipos evaluables en el panel común.

M27 dejó 173 candidatos NAM presentes, no excluidos y sin identidad administrativa con el baseline.
Ese es un conteo previo a parentesco, no 173 unidades independientes. M27D conservará y reportará por
separado cuatro grupos: los 173 candidatos NAM, los candidatos con metadata de Brasil, las 128 muestras
con gVCF evaluadas por M27C y la intersección entre esos grupos. País, población, fuente, proporción `Q`
y las banderas históricas `Maximum_unrelated_dataset` se usarán para describir y auditar, no para
declarar independencia.

## Preparación de marcadores

Para PCA y PC-Relate se conservarán únicamente SNPs autosómicos bialélicos con MAF global al menos 5%,
call rate al menos 98%, alelos compatibles entre archivos y fuera de las regiones de LD extendido ya
fijadas por el proyecto. Se usarán genotipos 0/1/2; la fase no es necesaria para estimar parentesco.

Después se hará poda por desequilibrio de ligamiento. El ancla será una ventana física de 1 Mb y
`r²=0,20`; la sensibilidad más estricta usará la misma ventana y `r²=0,10`. La ventana se expresa en
bases para que no dependa de la densidad desigual de variantes del WGS. El inicio del algoritmo será
determinista y quedará registrado. El número de SNPs retenidos será un resultado del panel, no una cifra
forzada. La referencia habitual de aproximadamente 200–300 mil SNPs podados sirve como control de orden
de magnitud, no como un gate que obligue a modificar filtros hasta alcanzar ese número.

MAF 5% y call rate 98% quedan fijos porque definen un panel común y técnicamente confiable para estimar
estructura. No se hará un barrido indiscriminado de cada constante. La conclusión sí puede depender del
número de componentes y de la poda LD; por eso esos dos elementos tendrán sensibilidad explícita.

## Ruta sin KING

El script histórico `bin/pcrelate_kinship.R` no se reutilizará: primero ejecuta KING-robust y PC-AiR.
M27D tendrá una ruta separada con PCs externos y `training.set`, ambos admitidos directamente por
GENESIS.

1. Se calcula una PCA provisional con los SNPs comunes podados y todas las muestras no excluidas.
2. Se ejecuta una primera pasada de PC-Relate con 8 PCs y todas las muestras como conjunto de ajuste.
   GENESIS permite este inicio cuando todavía no existe un conjunto independiente conocido. Esta salida
   es provisional y no certifica donantes.
3. Con las aristas `phi≥0,0442` se construye un conjunto independiente maximal por inclusión: no se
   puede añadir otra muestra sin crear una relación. La selección será determinista, priorizará call
   rate y resolverá empates con un hash estable del identificador. No usará `Q`, país, población ni una
   bandera histórica como sustituto de parentesco. La representación de cada grupo se revisará después;
   si algún grupo desaparece, se reportará el problema en vez de cambiar el orden para recuperarlo.
4. Se recalcula la PCA únicamente en ese conjunto no relacionado y se proyectan las demás muestras con
   los mismos loadings. Así los familiares no cambian los ejes usados para retirar ancestría.
5. Se ejecuta la pasada final de PC-Relate con ese `training.set`, `scale="overall"`, corrección de
   muestra pequeña y el límite interno de MAF 0,01 de GENESIS.

Para evitar una ambigüedad operativa, la primera pasada corresponde únicamente a la configuración
principal de 8 PCs y `r²=0,20`. De ella se obtiene un solo `training.set`, que se mantiene fijo en las
cuatro configuraciones finales. Así, las sensibilidades cambian solo el número de PCs o la poda LD; no
cambian también el conjunto usado para ajustar PC-Relate.

La primera y la segunda pasada deben conservarse completas. Si cambian los pares relevantes o la
composición de candidatos, no se esconderá la discrepancia: se marcará como inestabilidad del diseño.

## Barrido de sensibilidad preregistrado

La configuración principal usa 8 PCs y `r²=0,20`. Se añaden tres comparaciones de un factor por vez:

| Configuración | PCs | Poda LD | Pregunta |
|---|---:|---:|---|
| Principal | 8 | `r²=0,20` | Ancla histórica del proyecto con poda habitual |
| PCs bajos | 4 | `r²=0,20` | ¿Quedó estructura poblacional sin retirar? |
| PCs altos | 12 | `r²=0,20` | ¿La estructura más fina cambia el parentesco? |
| LD estricto | 8 | `r²=0,10` | ¿Los bloques correlacionados están dominando? |

No se escogerá después la configuración que deje más donantes. Para la lista conservadora, una pareja
se considerará relacionada si `phi≥0,0442` en cualquiera de las cuatro configuraciones. En paralelo se
reportarán `phi≥0,0221` y `phi≥0,0884` como sensibilidades de cuarto y segundo grado, pero esos cortes no
reemplazarán silenciosamente la política principal.

Ocho PCs es un ancla, no una verdad universal: fue el valor usado en la corrida histórica de PC-Relate
del proyecto. Cuatro y doce permiten detectar, respectivamente, ajuste demasiado grueso y estructura
más fina sin abrir una búsqueda amplia. Además de los conteos, se mostrarán los PCs por fuente,
población y ancestría, la varianza explicada y la contribución de componentes familiares. Si los ejes
están dominados por una familia o por el origen técnico de los datos, el resultado no pasa aunque el
número final de muestras parezca conveniente.

## Cómo se forma la lista de candidatos

La lista principal se obtiene en este orden:

1. resolver identidades y alias de forma unívoca;
2. retirar las 78 identidades del baseline;
3. retirar cualquier candidato con `phi≥0,0442` respecto de un donante del baseline en al menos una
   configuración;
4. sobre el grafo unión de relaciones entre candidatos, construir de forma determinista un conjunto
   independiente maximal por inclusión y sin aristas internas;
5. mantener por separado las etiquetas NAM, Brasil y WGS enlazado a M27C.

“Maximal por inclusión” significa que no se puede añadir otro candidato sin introducir una relación
bajo la política fijada; no garantiza la mayor cantidad matemática posible, no es la única solución y
no implica que los individuos representen una población parental pura. Como control se repetirá el
algoritmo con órdenes deterministas alternativos, sin elegir el que retenga más personas. Se informará
cuánto cambia la lista y qué candidatos cruzan el umbral solo en una sensibilidad.

La disjunción tiene dos niveles: identidad, que evita reutilizar exactamente al mismo individuo, y
parentesco, que evita usar familiares de los donantes del baseline como una evaluación aparentemente
independiente. Ambos deben pasar.

## Resultados y controles obligatorios

La salida mínima incluirá:

- contrato y hash de todos los inputs, software, contenedor y parámetros;
- número de muestras y SNPs después de cada filtro y por cromosoma;
- concordancia de las 77 identidades compartidas y estado del donante del baseline añadido;
- PCA provisional, PCA recalculada y proyección de todas las muestras;
- pares y componentes por cada configuración y por los tres umbrales reportados;
- comparación de `phi` entre pasadas y entre sensibilidades;
- lista privada de candidatos conservadores y una tabla pública agregada por fuente, población y grupo;
- conteos separados para NAM total, Brasil y las 128 muestras con gVCF;
- selección determinista y auditoría de empates;
- tiempo, CPU, memoria, disco, bytes leídos y costo estimado;
- manifiesto SHA-256 y recibo de gates.

La implementación tendrá además datos sintéticos con poblaciones estructuradas y pares de parentesco
conocido. Esos tests comprobarán orden de muestras, proyección de PCA, equivalencia entre cálculo por
bloques y monolítico, unión conservadora de aristas y ausencia de cualquier llamada a KING. Son controles
del código; no reemplazan la sensibilidad en los datos reales.

No se publicarán identificadores individuales en documentos o artefactos públicos. La lista necesaria
para reproducir la corrida quedará en el área privada del datalake con acceso controlado.

## Gates y regla de parada

M27D se detiene si ocurre cualquiera de estos casos:

1. no se reconcilian muestra, build, cromosoma, REF/ALT o los 78 donantes del baseline;
2. el panel común queda demasiado pobre o concentrado para obtener PCs de ancestría defendibles;
3. los PCs usados por PC-Relate están dominados por familias o fuente técnica;
4. la segunda pasada no reduce la dependencia familiar de los PCs;
5. la clasificación de candidatos cruza `phi=0,0442` entre configuraciones y no puede resolverse con
   la regla conservadora de unión;
6. no queda un conjunto NAM disjunto del baseline y sin relaciones internas bajo la política fijada;
7. al recalcular M27C únicamente sobre los candidatos finales, la preparación cae por debajo de 80% o
   depende de una política de calidad no defendible.

No se fija todavía un número “mínimo” de donantes para la simulación. Ese número depende del efecto
mínimo que se quiera detectar, de cuántos mosaicos compartan donantes y del error Monte Carlo. M27D
entregará el número real de unidades disponibles; el análisis de poder posterior dirá si alcanza. Pocos
donantes pueden bastar para una prueba técnica, pero no para una conclusión biológica general.

Un PASS de M27D habilita únicamente recalcular M27C en el panel final y diseñar el poder de la
simulación. No habilita por sí mismo Gnomix, mosaicos, entrenamiento ni TEST.

## Ejecución y costo

La ejecución se implementará como un workflow Nextflow separado. La preparación por cromosoma podrá
paralelizarse con un límite de tareas, mientras que cada PC-Relate usará paralelismo interno controlado.
Las cuatro configuraciones finales podrán correr en paralelo después de compartir la misma preparación.
Todas las máquinas creadas por Batch llevarán `team=frank`, se ejecutarán en `us-central1`, no dejarán
discos persistentes y publicarán únicamente resultados finales en Cloud Storage.

Antes de la corrida completa habrá un smoke con dos brazos para separar el costo cuadrático del número
de muestras del costo aproximadamente lineal del número de SNPs. El primer brazo usará todo el universo
de hasta 3.686 muestras y 10.000 SNPs distribuidos entre los 22 autosomas; allí se compararán 4, 8 y 16
hilos. El segundo usará 1.000 muestras estratificadas por fuente y ancestría y hasta 50.000 SNPs, con los
hilos elegidos en el primer brazo. Ninguno de estos subconjuntos producirá una conclusión científica.
Se elegirá el menor recurso que quede a menos de 20% del mejor tiempo y use menos de 70% de la RAM
solicitada. Este ajuste decide infraestructura, no el resultado científico.

Como los 4,53 GiB de entrada están en la misma región, no se espera costo de egreso. El diseño inicial
fijó un techo de US$2 y una hora para el smoke monolítico; la enmienda posterior se explica abajo. La
corrida completa no se lanzará si el smoke proyecta más de US$10 o seis horas sin nueva revisión. La
cifra inicial razonable es US$1–5, pero seguirá siendo una estimación hasta medir PC-Relate con este
panel. Si la paralelización reduce tiempo a costa de lecturas o máquinas duplicadas, se comparará costo
total y no solo reloj.

## Enmienda operativa después del primer smoke

La corrida `m27d-resource-smoke-20260814b` respetó el límite de una hora, pero terminó por timeout antes
de empezar PC-Relate. El log muestra que sí completó la importación de 3.558.958 SNP bialélicos, el
filtro común y las dos podas LD. En ese intento se observaron 220.742 SNP con `r²=0,20` y 141.249 con
`r²=0,10`. Estos dos conteos son preliminares: como el proceso no terminó, los archivos y sus hashes no
se publicaron y todavía no los considero resultados cerrados.

El problema fue operativo. Conversión, poda y benchmark estaban dentro de una sola tarea, por lo que la
preparación consumió el tiempo disponible antes de responder la pregunta de recursos de PC-Relate. No
hubo falta de memoria registrada, no se ejecutó parentesco y no cambió ninguna decisión biológica.

Para no repetir ese costo en cada configuración, la preparación pasa a ser un proceso persistente y
auditable. Este proceso publicará el GDS, las dos listas de SNP podados, los conteos por filtro y un
manifiesto SHA-256. Un segundo proceso reutilizará esos archivos para comparar 4, 8 y 16 hilos. La
preparación tendrá hasta 75 minutos, que añade alrededor de 50% de margen al tiempo observado, y el
benchmark conservará su límite de una hora. El presupuesto adicional conjunto será como máximo US$2 y
el acumulado conservador de los intentos M27D no deberá superar US$3.

Las dos fases se lanzan por separado. `prepare` no puede iniciar el benchmark; `benchmark` exige las
rutas explícitas y ya revisadas del GDS, las dos listas podadas, la tabla privada de estratos y el
manifiesto de preparación. Esta pausa evita que una salida incompleta avance automáticamente.

Esta enmienda no modifica MAF, call rate, ventana física, `r²`, número de PCs, umbrales de parentesco,
panel de muestras ni regla conservadora. Tampoco habilita la corrida completa. Primero debe terminar la
preparación con hashes; después se revisan memoria, costo y limpieza de la nube antes de lanzar el
benchmark.

## Resultado cerrado de la preparación y del benchmark

La preparación final `m27d-marker-preparation-20260814c` terminó correctamente y ya no necesita
repetirse. El GDS contiene 3.685 muestras y 3.558.958 SNP bialélicos autosómicos. Después de exigir
MAF≥5% y call rate≥98% quedaron 3.331.146; después de excluir además las regiones de LD extensa
quedaron 3.298.309. La poda `r²=0,20` retuvo 220.742 marcadores y la poda `r²=0,10`, 141.249. Los
archivos persistentes y el manifiesto fueron verificados por SHA-256.

El benchmark `m27d-pcrelate-smoke-20260814d` reutilizó esos archivos y pasó primero un control de
integridad que recalculó sus hashes. Con 3.685 muestras y 10.000 SNP, los tiempos combinados de PCA y
PC-Relate fueron:

| Hilos | Tiempo | Diferencia frente al más rápido |
|---:|---:|---:|
| 4 | 189,397 s | 3,95% |
| 8 | 186,599 s | 2,41% |
| 16 | 182,203 s | referencia |

Las tres corridas produjeron exactamente 6.787.770 pares. El control adicional de 1.000 muestras y
50.000 SNP, ejecutado con 4 hilos, produjo 499.500 pares en 34,894 s. El proceso completo tuvo un pico
de 11,2 GB de RAM y terminó sin reintentos. No guardó pares individuales, no ejecutó KING y no produjo
una conclusión de parentesco.

De acuerdo con la regla fijada antes de la corrida, se eligen **4 hilos**: quedan muy por debajo del
margen de 20% y evitan pagar por 8 o 16 hilos que casi no reducen el tiempo. Se reservan **32 GiB por
tarea PC-Relate**. Una máquina de 16 GiB quedaría alrededor del límite de 70% con el pico ya observado
y daría poco margen para la carga completa; 64 GiB no se justifican con estos datos.

La extrapolación desde el brazo con todas las muestras sugiere unos 66 minutos por pasada con 220.742
SNP y unos 42 minutos con 141.249 SNP si el tiempo crece aproximadamente de forma lineal con los
marcadores. Es una estimación de capacidad, no un resultado garantizado. El primer `pass0` completo
será el punto de control real: se detendrá si supera 90 minutos, usa 22,4 GiB o más, no produce
6.787.770 pares o eleva la proyección total por encima de 6 horas o US$10.

Antes de ese `pass0` faltan tres tareas: resolver las 35 correspondencias ambiguas y 17 ausentes de la
metadata; implementar la ruta productiva de Nextflow con pruebas sintéticas; y revisar un DAG que
contenga solo M27D. La auditoría completa necesitará una autorización explícita después de mostrar su
comando, costo actualizado y controles.

## Resolución de identidades del panel

La correspondencia entre las 3.685 muestras del panel y la tabla de metadata se resuelve con una
política general, sin listas de identificadores en el código. El orden es fijo y no depende de la
posición de las filas:

1. Si una sola fila es alcanzable por cualquier alias, esa fila gana.
2. Si hay varias, se descartan las marcadas `Exclude`, pero solo mientras quede al menos una: la
   exclusión es una preferencia, no un filtro duro.
3. Se descartan sin excepción las filas sin genotipos. Una fila que no puede aportar genotipos no
   puede ser el miembro del panel, y un `N_genotypes` ausente cuenta como cero, no como permiso.
4. Se prefieren las filas cuyo `IID` coincide directamente con el identificador del panel frente a
   las alcanzables solo por una columna de alias.
5. La muestra se resuelve únicamente si sobrevive exactamente una fila. Cualquier otro caso detiene
   la etapa.

Sobre el panel real esto deja **3.640 `DIRECT_UNIQUE`, 35 `RESOLVED_ACTIVE_GENOTYPED_IID`, 10
`UNMATCHED` y 0 `AMBIGUOUS_FAIL_CLOSED`**. Las 35 colisiones que quedaban pendientes tienen todas la
misma forma: dos filas candidatas, una excluida y sin genotipos alcanzable solo por alias, y otra
activa, genotipada y con coincidencia directa de `IID`. La regla las resuelve sin recurrir al orden
de aparición ni a la población esperada.

### Las muestras sin metadata eran diez, no diecisiete

El recuento anterior de 17 mezclaba dos cosas distintas. Siete de esas muestras sí están en la
metadata: sus identificadores contienen un guion bajo interno, de modo que el identificador doble de
PLINK toma la forma `A_B_A_B`, y la normalización previa solo colapsaba el caso `X_X`. Corregida la
normalización, quedan **diez** muestras sin ninguna fila de metadata.

El origen de esas diez está demostrado por diferencia de conjuntos contra el panel anterior
(`gs://projects-usp/nam-diversity/nat.163wgs.1000G.sgdp.hgdp.hg38/nat.163wgs.1000G.sgdp.hgdp.hg38.fam`,
95.213 bytes, 3.710 filas):

```
3.710 (panel anterior) − 35 (retiradas) + 10 (añadidas) = 3.685 (panel actual)
```

Las diez añadidas llevan prefijos `ONG` y `JAR` y no aparecen en ninguna tabla del bucket. Su
procedencia operativa está probada; su población **no**. La única evidencia poblacional es el prefijo
de su propio identificador, y un prefijo no es una anotación autoritativa, así que no se les asigna
población. Se quedan en el PCA y en PC-Relate, porque quitar a alguien de una auditoría de parentesco
por un hueco administrativo sería peor, y quedan marcadas como no interpretables: no entran en
resúmenes estratificados ni pueden ser seleccionadas como donantes.

Hay además un efecto colateral que conviene tener presente al leer los conteos de donantes: **las 35
muestras retiradas del panel son todas `Source=PSI` y `Ancestry=Native_American`**. La reconstrucción
del panel no fue una suma limpia, y el universo NAM disponible es menor que el del panel de 2022.

La normalización se implementó en el resolutor de M27D y no en el ayudante compartido con M27B: los
artefactos publicados de M27B hashean contra esos bytes exactos y reescribirlos rompería la
reproducibilidad de una corrida terminada.

## Ruta productiva implementada

El flujo tiene cuatro fases que se lanzan por separado (`prepare`, `benchmark`, `strata`, `audit`).
La fase `audit` exige además `--donor_kinship_smoke_only false` y una autorización humana explícita
`--donor_kinship_full_run_authorized true`; sin ella el workflow se detiene antes de crear ninguna
tarea.

| Proceso | Qué hace |
|---|---|
| `RESOLVE_DONOR_KINSHIP_STRATA` | Resuelve identidades y publica la tabla privada y el resumen agregado |
| `RUN_DONOR_KINSHIP_PASS0` | PCA provisional, PC-Relate sobre todas las muestras elegibles y construcción del `training.set` |
| `AUDIT_BASELINE_DONOR_IDENTITY` | Reconcilia los donantes del baseline por concordancia de dosis, no por nombre |
| `FIT_DONOR_KINSHIP_PCA` | Reajusta la PCA solo sobre el `training.set` y proyecta al resto |
| `RUN_DONOR_KINSHIP_CONFIGURATION` | Ejecuta las cuatro configuraciones preregistradas |
| `SELECT_DONOR_KINSHIP_CANDIDATES` | Grafo unión, disjunción, conjunto independiente y recibo de gates |

Se ajusta **una PCA por conjunto de marcadores LD**, no una por configuración. El número de
componentes es un corte de un único ajuste, así que una configuración que solo cambia los PCs cambia
de verdad un solo factor; cambiar la poda LD sí exige su propio ajuste, y ese es precisamente el
factor que varía la configuración estricta. Las cuatro configuraciones se leen del preregistro y no
se repiten en el código de Nextflow, para que contrato e implementación no puedan divergir.

## Determinismo

`snpgdsPCA(algorithm = "randomized")` sortea una matriz de prueba aleatoria. Medido sobre el fixture
sintético, dos corridas sobre entradas idénticas produjeron autovectores que diferían hasta en
**0,97**; con `set.seed()` la diferencia fue exactamente **0**. Esos puntajes llegan a PC-Relate, al
`training.set` y a la lista de candidatos, de modo que la semilla se fija en el preregistro
(`determinism.random_seed = 20260814`) y se escribe en el resumen de cada etapa.

Con la semilla fijada, dos corridas completas del flujo sobre el fixture produjeron **20 de 20
salidas byte-idénticas**, incluidas las de PC-Relate. El contrato solo exigía invariantes y
tolerancias numéricas; en la práctica se obtuvo reproducibilidad exacta, pero esa afirmación está
verificada sobre el fixture y todavía no sobre el panel real con cuatro trabajadores.

También se comprueba un supuesto que GENESIS degrada en silencio: la corrección de muestra pequeña
solo se aplica cuando toda la cohorte cabe en un bloque de muestras, y si no cabe, la librería la
desactiva con una simple advertencia. Con 3.685 muestras y un bloque de 5.000 se aplica, pero la
condición se verifica en el código en vez de darse por supuesta.

## Verificación sobre datos sintéticos

El fixture construye 90 muestras en dos grupos ancestrales, seis tríos padre-madre-hijo, colisiones
de alias con la misma forma que las reales, muestras sin metadata, muestras excluidas y un baseline
que comparte siete donantes con el panel y deja uno fuera. Todo lo que la auditoría afirma tiene ahí
una respuesta correcta conocida de antemano.

El resultado más informativo no es que los tests pasen, sino esto:

| Pasada | φ mediana en pares padre-hijo (verdad 0,25) | Falsos positivos a φ≥0,0442 |
|---|---:|---:|
| pass0, `training.set` = todos | 0,4173 | 5 |
| `anchor_pc8_r2_020` | 0,2796 | 0 |
| `pc8_r2_010` | 0,2716 | 1 |
| `pc_high_12_r2_020` | 0,3021 | 0 |
| `pc_low_4_r2_020` | 0,2676 | 0 |

pass0 ajusta las frecuencias alélicas sobre un conjunto que todavía contiene parientes y por eso
sobreestima. El reajuste sobre el conjunto independiente cierra buena parte de esa brecha y elimina
los falsos positivos. Esa es la razón de que el diseño tenga dos pasadas; si el reajuste dejara de
corregir, el test de integración falla.

Los doce pares emparentados se recuperan en las cuatro configuraciones, ninguno sobrevive dentro del
`training.set`, y la identidad del baseline se confirma con concordancia de dosis 1,000 frente a un
mejor impostor de 0,585.

## Memoria de PC-Relate

La preocupación operativa principal era si 32 GiB alcanzan al pasar de 10.000 a 220.742 marcadores.
No escalan: dentro de `.pcrelate`, GENESIS recorre los bloques de SNP con
`bpiterate(..., REDUCE = .matListCombine, ...)`, que reduce de forma incremental en lugar de
acumular los resultados de cada bloque. Las matrices acumuladoras son de tamaño n×n y dependen solo
del número de muestras.

Medido sobre el panel sintético multiplicando por diez el número de bloques:

| SNP | Bloques | Pico de memoria | Tiempo |
|---:|---:|---:|---:|
| 400 | 2 | 470,9 MB | 0,92 s |
| 1.000 | 5 | 461,6 MB | 1,06 s |
| 2.000 | 10 | 476,5 MB | 1,45 s |
| 3.958 | 20 | 482,1 MB | 2,10 s |

La memoria es plana y solo el tiempo crece. El pico de 11,2 GB observado con 10.000 SNP y 3.685
muestras debería sostenerse con 220.742, de modo que 32 GiB dejan alrededor del triple de margen.
El pass0 real sigue siendo el punto de control: si el pico llega a 22,4 GiB, se detiene.

## Corrección de la lectura del benchmark

El texto anterior justificaba los 4 hilos diciendo que quedaron «a 3,95% del mejor
tiempo». Esa lectura no se sostiene y conviene decirlo con claridad, porque de ella
cuelga el dimensionamiento.

El brazo de 10.000 SNP corrió con un solo bloque de marcadores: `GenotypeBlockIterator`
usa bloques de 10.000 por defecto, de modo que `bpiterate` entregó un único elemento y
sólo un trabajador tuvo trabajo. En esas condiciones el número de hilos no puede afectar
a PC-Relate, y la diferencia del 3,95% entre 4, 8 y 16 hilos mide únicamente el escalado
de la PCA. Se eligió el parámetro que gobierna el riesgo de memoria en un experimento
donde ese parámetro no tenía efecto.

Medido después sobre un panel sintético con la dimensión real de muestras (3.685
individuos, 6.787.770 pares, cuatro bloques de marcadores en vuelo):

| Trabajadores | Memoria máxima de R | RSS del árbol | Segundos |
|---:|---:|---:|---:|
| 1 | 4.467 MB | 2.210 MB | 218,3 |
| 4 | 5.122 MB | 2.309 MB | 194,9 |

Pasar de uno a cuatro trabajadores sube la memoria un 14,7%, no cuatro veces, y el
tiempo baja un 11%. La conclusión operativa —4 hilos y 32 GiB— sobrevive, pero ahora
descansa en una medida del eje correcto y no en un contraste que no podía discriminar.
El tamaño de bloque se fija explícitamente en el preregistro para que el número de
bloques en vuelo sea una cantidad declarada y no un valor por defecto de la librería.

## Tiempo: el coste fijo domina

La proyección de unos 66 minutos por pasada salía de multiplicar por 22 un único punto
de 10.000 SNP. Ese método supone que todo el tiempo escala con los marcadores, y no es
así: después de recorrer los bloques, GENESIS construye en el maestro y en un solo hilo
las tablas de 6.787.770 filas, un coste que depende del número de muestras y no del de
marcadores.

Medido sobre el mismo panel sintético de 3.685 muestras:

| SNP | Segundos |
|---:|---:|
| 2.500 | 167,4 |
| 5.000 | 180,0 |
| 10.000 | 168,9 |
| 20.000 | 194,9 |

El ajuste `t = 165,4 + 0,001325·m` da unos **7,6 minutos** para 220.742 marcadores y
unos **5,9** para 141.249, frente a los 66 y 42 de la extrapolación lineal pura. El
punto de 10.000 SNP reproduce además el tiempo del benchmark real (168,9 s medidos aquí
frente a 179,8 s observados en la nube), lo que sugiere que el panel sintético es un
buen sustituto para dimensionar.

Sigue siendo una estimación: los datos reales tienen genotipos ausentes y estructura de
LD que el panel sintético no reproduce, y ambas cosas añaden trabajo. Por eso la
stop-rule de 90 minutos se mantiene sin tocar, ahora con un margen mucho mayor.

## Compuertas y artefactos añadidos

El recibo emite una fila por cada compuerta preregistrada, con tres estados posibles:
`PASS`, `FAIL` y `NOT_EVALUATED`. Un recibo que omite las compuertas que esta etapa no
puede decidir se lee como si todas hubieran pasado. `G0` admite además
`PASS_WITH_BLIND_SPOT`, que es la lectura honesta cuando un donante del baseline no
tiene gemelo en el panel: su parentesco con los candidatos no puede comprobarse, y eso
es un punto ciego real y no un detalle de redondeo.

`G2` se adjudica en la etapa de PCA y detiene la corrida allí. Dejar que fallara sólo en
la selección obligaba a pagar antes las cuatro pasadas de PC-Relate.

Se persisten dos objetos que el código anterior calculaba y tiraba:

- el coeficiente de endogamia por individuo, que GENESIS devuelve en la misma llamada.
  Es el único observable del módulo capaz de separar un pedigrí reciente de la deriva
  dentro de una población pequeña, y recuperarlo después costaría otra pasada completa;
- la cobertura por estrato del conjunto de entrenamiento y de los candidatos, con el
  denominador al lado del superviviente. Contar sólo supervivientes esconde justo el
  fallo que importa: que una población entera desaparezca porque todos sus miembros
  están emparentados entre sí.

## `pass0` es lanzable por separado

El contrato llama a `pass0` «el punto de control real», así que ahora es una fase propia.
Lanzar `audit` compromete el grafo entero y paga las cuatro configuraciones antes de que
nadie haya leído cuántos pares emparentados encontró la primera pasada. La fase `pass0`
se detiene justo después del conjunto de entrenamiento.

El número de pares se compara contra el valor absoluto fijado en el preregistro
(`pass0_checkpoint.expected_pairs = 6787770`) y no sólo contra `n(n-1)/2` recalculado a
partir de la misma `n`, que es autoconsistente y no puede detectar que el universo
elegible cambió. Comprobado sobre la metadata real: ninguna de las 3.685 muestras del
panel tiene `Exclude=TRUE`, de modo que el universo elegible es 3.685 y el invariante de
6.787.770 pares es correcto.
