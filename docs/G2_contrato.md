# Contrato G2: variantes raras como apoyo a la inferencia de ancestría local

Estado: reconstruido y actualizado el 13 de agosto de 2026. Este documento define qué tendría que
demostrarse para afirmar que un canal de variantes raras mejora el LAI. No autoriza por sí solo una
corrida.

## Pregunta científica

La pregunta de G2 es si la información de variantes raras puede corregir etiquetas o precisar bordes
de ancestría local por encima del mismo baseline basado en variantes comunes. La comparación debe
aislar el aporte incremental del canal raro: no basta con mostrar que las raras recuperan ancestría
global, carga rara, cohorte o geografía.

El primer piloto queda acotado al cromosoma 22. Su alcance es metodológico y no pretende representar
por sí solo el comportamiento de los 22 autosomas.

## 0. Identificabilidad

Una mejora de LAI solo es identificable si se conoce la ancestría verdadera en cada posición. En los
individuos reales de DNABR esa verdad no está disponible. Gnomix es el baseline que se quiere evaluar,
no puede ser su propio juez; las etiquetas M14 proceden de las mismas variantes raras y tampoco son una
verdad independiente. NAMBR sirve como fuente potencial de haplotipos, pero su ancestría global no
proporciona etiquetas verdaderas por locus.

Por eso, una evaluación definitiva necesita cromosomas mosaico simulados a partir de haplotipos
parentales faseados y un mapa de recombinación. La simulación conserva la etiqueta de origen de cada
segmento y permite conocer tanto la ancestría por posición como los bordes reales.

La simulación solo sería válida si se cumplen estas condiciones:

- los donantes usados para generar mosaicos no aparecen en el baseline ni en su ajuste;
- el parentesco se comprueba con PC-Relate y los componentes familiares se tratan como una sola unidad;
- cada población parental está definida por evidencia genómica y metadata, no solo por un umbral de
  ancestría global;
- los haplotipos, alelos, coordenadas hg38 y mapa genético son compatibles y trazables;
- el canal raro conserva fase auditable o, si se usa una formulación sin fase, esa decisión queda
  declarada como un diseño distinto.

No se usará KING. Tampoco se llamará “parental puro” a NAMBR: es un conjunto brasileño útil para explorar
el eje indígena americano, pero contiene historia demográfica y mezcla reales.

## 1. Estimando

El estimando primario es la reducción absoluta y pareada del error de dosis de ancestría por posición,
promediada de forma equilibrada entre AFR, EUR y NAM:

`Δerror = error(baseline común) − error(baseline común + canal raro)`.

“Pareada” significa que ambos modelos se evalúan sobre los mismos mosaicos, posiciones y particiones.
Un valor positivo favorece el canal raro.

El estimando secundario es el cambio en el error de localización de bordes, medido en distancia genética
y de forma invariante a intercambios arbitrarios entre los dos haplotipos. Se usa distancia genética
porque un mismo número de bases no representa la misma cantidad de recombinación en todo el cromosoma.

La primera comparación prevista es deliberadamente sencilla: el mismo corrector multinomial con
regularización L2, una vez con las salidas del baseline y otra con el canal raro añadido. La
regularización L2 penaliza coeficientes grandes para reducir inestabilidad. Una red neuronal solo se
consideraría si el canal raro muestra utilidad incremental con este comparador y quedan suficientes
unidades independientes.

## 2. Unidad de análisis y partición

La observación elemental es la posición genómica del mosaico, pero la unidad independiente para dividir
datos, estimar incertidumbre y hacer bootstrap es el paquete completo de un donante o componente
familiar. Las posiciones de un mismo cromosoma y los mosaicos derivados de los mismos donantes no son
réplicas independientes.

La selección de variantes raras y cualquier normalización se ajustan solo con TRAIN. En cada partición,
“rara” significa alelo menor con MAC al menos 2 y MAF menor de 1 % dentro del conjunto de ajuste. La
orientación es al alelo menor, no a ALT por defecto. El requisito mínimo de dos unidades portadoras
independientes evita que una familia cuente varias veces como evidencia.

No se heredan automáticamente las ventanas de 250 kb, 500 kb o 1 Mb usadas en M25. Esas escalas
respondían a una matriz marginal individuo×ventana, no a la resolución de LAI. Si el canal necesita
agregación local, su escala se fijará después de medir soporte raro y resolución en TRAIN/VALIDATION y
se expresará también en cM. No se elegirá mirando TEST.

## 3. Juez o verdad de referencia

El juez principal es la ancestría por posición guardada por el simulador. Debe incluir la etiqueta de
cada segmento, sus bordes y el identificador del paquete de donantes que lo generó.

Los siguientes elementos pueden usarse como controles o descripciones, pero no como verdad por locus:

- Gnomix, FLARE o RFMix sobre DNABR real;
- proporciones globales de ancestría `Q`;
- macroclados de mtDNA o chrY;
- Refined-IBD;
- comunidades M14/M16.5;
- pertenencia nominal a NAMBR.

Los macroclados uniparentales siguen siendo útiles para una comprobación observacional de concordancia,
pero validan linajes maternos o paternos, no segmentos autosómicos. Esa ruta histórica no permite afirmar
que el LAI mejoró.

## 4. Hipótesis nula y controles

La hipótesis nula es que, fuera de muestra, añadir el canal raro no reduce el error posicional ni el
error de bordes respecto al mismo baseline común.

Los controles mínimos son:

- ancestría global `Q`, para detectar si el canal solo reproduce composición continental;
- carga rara global, para detectar si el resultado se explica por cuántos alelos raros porta cada
  individuo;
- cohorte y origen de los datos, para vigilar efectos de reclutamiento o procesamiento;
- callability y regiones de baja complejidad/mappability, para separar ausencia biológica de falta de
  observación o mapeo difícil;
- MAC/MAF y contexto mutacional, cuando se comparen variantes con probabilidades distintas de recurrencia;
- análisis por ancestría y, en especial, por tramos NAM, porque una métrica global puede ocultar el fallo
  de la clase minoritaria.

La longitud de haplotipo compartido puede entrar como sensibilidad etiquetada, pero no como null limpio:
usa parte del mismo reloj genealógico que se intenta evaluar. Tampoco se usará Madsen–Browning como una
solución automática; en el rango raro de DNABR su cambio de peso es modesto y no elimina por sí solo el
dominio de los alelos de menor frecuencia.

## 5. Poder, incertidumbre y regla de parada

Antes de fijar el número de mosaicos se debe declarar un efecto mínimo científicamente relevante
(`SESOI`): la mejora más pequeña que justificaría añadir el canal raro. Después, un smoke de simulación
estima la correlación entre mosaicos que comparten donantes y el error de Monte Carlo. Esas dos cantidades
determinan el número efectivo de unidades y cuántas réplicas hacen falta. Generar miles de mosaicos a
partir de pocos donantes no crea miles de observaciones independientes.

La incertidumbre se estima con bootstrap pareado sobre paquetes completos de donantes o componentes
familiares. Las métricas se reportan por ancestría además del promedio macro. TEST permanece cerrado hasta
congelar datos, particiones, modelo, métricas, SESOI y regla de decisión.

El piloto se detiene antes de simular o entrenar si falla cualquiera de estos puntos:

1. contrato de archivos, hg38, cromosoma, alelos, orden o mapa genético;
2. donantes disjuntos y componentes familiares verificables;
3. parentales defendibles para las tres ancestrías;
4. compatibilidad del baseline sin rellenar o sustituir marcadores de forma no registrada;
5. soporte raro definido solo con TRAIN y presente en al menos dos unidades independientes;
6. fase o representación sin fase con procedencia auditable;
7. poder suficiente para detectar el SESOI sin reutilizar TEST.

Un resultado negativo obtenido con un panel común empobrecido en variantes raras no refuta la hipótesis
WGS. Solo evalúa esa representación y ese conjunto de activos.

## Estado operativo después de M27

M27 auditó la posibilidad de ejecutar el baseline Gnomix chr22 congelado. No probó la hipótesis de G2.

| Gate | Resultado observado | Consecuencia |
|---|---|---|
| G0: archivos y coordenadas | PASS | El contrato técnico básico de chr22/hg38 fue consistente. |
| G1: identidad, independencia y parentales | no resuelto | 77/78 IDs del baseline reaparecen en el panel externo; falta independencia confirmada con PC-Relate. |
| G2: compatibilidad del baseline | FAIL | Coinciden 19.535/110.074 marcadores exactos, 17,75 % frente al 80 % preregistrado. |
| G3: soporte raro y fase | omitido | El workflow se detuvo antes de leer o seleccionar el canal raro. |
| G4: poder | omitido | No se fijó tamaño de simulación ni se abrió TEST. |

No se bajará el umbral después de observar el fallo ni se rellenarán los 90.539 marcadores ausentes como
si fueran homocigotos de referencia. Si se decide construir otro baseline, será un experimento nuevo con
su propio contrato, no una modificación silenciosa de M27.

La cifra histórica de 182 corresponde a filas con una bandera de metadata, no a 182 donantes
independientes. El cruce estricto actual deja 173 candidatos NAM presentes y disjuntos del baseline;
161 tienen `Maximum_unrelated_dataset` y 146 una segunda marca, pero esos conteos todavía no sustituyen
PC-Relate.

También se verificó que los 128 NatWGS aparecen en un scaffold común de chr22 faseado. El WGS raro crudo
es un activo separado. La presencia de una muestra en ambos archivos no demuestra que sus variantes raras
estén conservadas ni faseadas en el scaffold.

## Resultado de M27B y siguiente paso permitido

M27B fue una auditoría read-only del puente entre:

- WGS NatWGS chr22 crudo:
  `gs://projects-usp/nambr/chr/joint_germline_recalibrated.normalized.chr22.vcf.gz`;
- scaffold común faseado:
  `gs://projects-usp/nam-diversity/shapeit/phased/natwgs.1000G.sgdp.hgdp.andamanese.hg38.22.norm.PHASED.vcf.gz`;
- referencia del baseline:
  `gs://projects-usp/dna-do-brasil/dnabr-lai-gnomix/vcf_fixed/dnabr.refpop.fixed.chr22.vcf.gz`.

La corrida final resolvió 128/128 identidades, encontró 14.058 sitios raros de alelo menor en chr22,
968 sitios exactos en el scaffold y 954 con puente directo de fase. B4 quedó `FAIL` bajo su definición
histórica: el VCF conjunto, que contiene únicamente posiciones variantes, expuso 34.856 de los 110.074
registros del modelo congelado. Ese resultado y sus artefactos no se modifican.

La ausencia de una posición en ese VCF no permite distinguir un genotipo 0/0 de una posición no
llamable. Por eso M27C será una auditoría nueva, no una reinterpretación de M27B. Consultará mediante
índice las 110.074 posiciones en los gVCF individuales y separará dos preguntas:

1. **Genotipabilidad:** cuántas posiciones tienen un genotipo respaldado por los bloques de referencia
   y los criterios de calidad declarados. El 80 % de Gnomix sigue siendo un requisito de posiciones,
   no un umbral ponderado por frecuencia.
2. **Información ancestral:** usando los donantes AFR/EUR/NAM del baseline, qué diferenciación conserva
   la intersección genotipable y cómo se distribuye a lo largo del mapa genético. Esta parte es
   descriptiva; no sustituye la prueba final con verdad de ancestría por posición.

Superar la primera pregunta solo dirá que el baseline puede ejecutarse. No demostrará que sea útil ni
que las variantes raras mejoren LAI. Antes de entrenar siguen pendientes los parentales, la disjunción,
una ruta de PC-Relate que no ejecute KING y la simulación con verdad por segmento.

## Estado después de M27C y del benchmark M27D

M27C comprobó que los gVCF permiten recuperar 92.790/110.074 marcadores listos bajo la política
principal, es decir, 84,30%. Como el resultado cae a 79,81% con DP≥15 y a 74,13% con call rate≥99%,
esto es factibilidad técnica sensible al filtro, no certificación de un panel parental.

M27D ya cerró la preparación de SNP comunes para auditar parentesco: 3.298.309 marcadores quedaron
después de calidad y exclusión de LD extensa, y las dos podas retuvieron 220.742 y 141.249. El benchmark
de PC-Relate confirmó los pares esperados, no ejecutó KING y eligió 4 hilos con 32 GiB por tarea. Esta
elección solo fija infraestructura; todavía no existen resultados de parentesco ni una lista final de
donantes.

Antes de ejecutar la auditoría completa faltan resolver 35 correspondencias ambiguas y 17 sin match en
la metadata, implementar las dos pasadas de PC-Relate y probar la selección determinista. La primera
pasada usará únicamente 8 PCs y `r²=0,20`; el conjunto independiente derivado de ella se mantendrá fijo
en las cuatro configuraciones finales. Solo después se podrá recalcular M27C en los donantes retenidos.

## Regla para parámetros e hiperparámetros

No se fijará un valor sensible por costumbre o intuición. Si existe un requisito externo que mide
exactamente la misma cantidad —por ejemplo, el 80 % de posiciones del modelo indicado por Gnomix— se
conserva como ancla y no se reajusta después de observar el resultado. Cuando no exista un valor
directamente transferible, se hará lo siguiente antes de abrir TEST:

- definir un rango razonable a partir de literatura, propiedades de DNABR y resultados previos;
- declarar qué valor es el ancla y cuáles son sensibilidades;
- evaluar el rango únicamente en TRAIN/VALIDATION, respetando paquetes de donantes o familias;
- reportar estabilidad, incertidumbre y todas las configuraciones, no solo la mejor;
- fijar de antemano la regla de selección y el resultado que detiene la línea.

El barrido debe ser proporcional al riesgo. Una constante técnica no necesita una búsqueda artificial;
una escala de agregación, penalización o hiperparámetro capaz de cambiar la conclusión sí necesita una
sensibilidad o búsqueda controlada. Las ventanas de 250 kb, 500 kb y 1 Mb de M25 no se heredan como
valores de LAI: si M27C permite continuar, la escala local se evaluará con un rango definido sobre
TRAIN/VALIDATION y una resolución coherente con el mapa genético.

## Alcance respecto de M23 y M25

M23 mostró que la matriz marginal rara cruda no añadió capacidad predictiva útil al baseline C bajo
Elastic Net y ese pipeline. M25–M25C mostraron que la representación individuo×ventana de chr22 tenía una
señal lineal pequeña, dominada por carga y dependiente de ventanas poco informadas; por eso no se avanzó a
NMF ni autoencoder. Ninguno de esos resultados invalida una representación posicional evaluada contra
verdad de LAI, pero sí obliga a demostrar utilidad con un comparador sencillo antes de aumentar la
complejidad.

## Artefactos que fijan este contrato

- `conf/m27_lai_pilot_preflight_preregistration.json`;
- `bin/audit_lai_pilot_preflight.py`;
- corrida `m27-lai-preflight-20260811a`;
- `m27_lai_pilot_preflight_summary.json`, SHA-256
  `b04f861370a7142b6cbb3337c71bbedee27f74a98916de92bee833cf8f2cada8`;
- decisión canónica de M27 del 13 de agosto de 2026.
