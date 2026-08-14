# M27D: parentesco y disjunción de donantes sin KING

**Estado:** contrato científico fijado antes del piloto técnico del 14 de agosto de 2026 y enmienda
operativa registrada después del primer timeout. Los valores que pueden cambiar la conclusión
científica no se elegirán después de mirar qué configuración conserva más donantes.

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
