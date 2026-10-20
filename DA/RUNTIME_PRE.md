# PRE de ingeniería, ejecución y procedencia — STRUCTURE y preparación LAI

**Fecha:** 2026-10-19. **Alcance:** revisión previa; no se modificó código, no se leyó ningún arreglo de genotipos y no se lanzó inferencia, entrenamiento ni evaluación. Consejo Élfico 0.4.1: `UNQUALIFIED`; este texto es auditoría automática adversarial, no juicio humano independiente. El `PRE_SNAPSHOT.json` compartido tiene SHA-256 `1a973e0cd52927a17e0b54b02fb9df2c03d8f379ce96fea56f1fb7260c8f5200`.

## Hechos comprobados: fallo de STRUCTURE

La versión 6 terminó `FAILED_PRODUCTS_AND_CACHE_RETAINED`; `training_started=false` y `evaluate_opened=false`. El fallo está en `PREPARE_SHARED_INDEX`: `r02_structure_person_batch_builder.py:budget_tiles` llama al lector congelado y recibe `map-expanded source chunk upper bound cannot fit one core under max_window_sites`.

El primer fallo reproducible usando sólo `manifest.json`, `event_inputs.json` y `map.tsv` es el contexto de atención local de 0.2 cM y profundidad 2, hash `5b9a93d7…`, cromosoma 11. Tras ocho fragmentos, el núcleo candidato `[46,258,094, 46,305,748]` bp necesita el canal de variantes raras en `[45,558,414, 56,290,769]` bp. Interseca 201 bloques: la cota superior es 102,912 filas y 199 bloques interiores ya contienen 101,888; agregando al menos una posición de cada bloque frontera, el mínimo real es 101,890. Por tanto, el exceso sobre `max_window_sites=100000` es real, no sólo redondeo por bloques. La subdivisión hasta un único punto tampoco resuelve todos los núcleos.

Las cotas superiores conservadoras máximas para una ventana centrada en un punto son 72,704 filas (0.05 cM, chr11), 109,568 (0.2 cM, chr11) y 187,392 (1.0 cM, chr3). Son sumas de bloques, no conteos exactos. No justifican elevar el límite: `genomic_context_runtime.LocalAttention.forward` construye una máscara densa y la repite por 8 personas × 4 cabezas. A 100,000 posiciones sólo esa máscara requiere aproximadamente 320 GB; a 187,392, aproximadamente 1.12 TB, antes de puntuaciones, activaciones y gradientes.

## Corrección propuesta y prueba previa obligatoria

1. Separar dos límites: filas que el lector puede recorrer y número máximo de elementos de atención simultáneos. Calcular el conteo exacto de una ventana mediante sumas de `n_sites` para bloques interiores y búsqueda binaria de posiciones en los dos bloques frontera; esto elimina sobreconteo sin tocar genotipos.
2. Reemplazar la máscara global de tamaño $N \times N$ por atención local exacta por bloques de consultas. Para cada bloque de consultas, localizar con `searchsorted` las claves comprendidas en el mismo radio genético; calcular puntuaciones, máscara y `softmax` sólo para ese rectángulo consulta×clave. Conservar dos capas y, por ello, el halo fuente de dos radios. No truncar, muestrear ni cambiar 0.05/0.2/1.0 cM.
3. Verificar igualdad numérica con la implementación densa en un ejemplo pequeño y en un ejemplo adverso con baja recombinación: mismas salidas y gradientes dentro de una tolerancia declarada, incluida una consulta sin vecinos y límites de cromosoma. Después medir tiempo, lectura y RSS máximo para 8 personas en el fragmento real más denso de 1.0 cM. Fijar tamaños de bloque a partir de esa medición y comprobar propagación en configuración, Nextflow y contenedor. Una medición que exceda 2 GiB detiene la preparación; no autoriza aumentar memoria ni iniciar entrenamiento.

Los invariantes científicos son: catálogo y alelo `source_minor` fijos de 2,723 personas; FIT/SELECT/EVALUATE sólo dividen personas; orden, fase, faltantes, coordenadas, mapa y exclusiones sin cambio; seis contextos exactos; ninguna apertura de EVALUATE ni conclusión biológica.

## Recuperación compatible

Hay cuatro archivos completos en el directorio de trabajo de la versión 6. Los tres Deep Sets ya registrados tienen SHA-256 `3116333f…`, `ad7491e2…` y `55e2f6f9…`. Además, el contexto de atención 0.05 cM `c531477a…tiles.json` tiene SHA-256 `0c22d6d7…`, 334 fragmentos y los 22 cromosomas; sus conteos coinciden con el recálculo por metadatos. No existen `index.json`, `PREPARED.json`, `FINAL.json` ni `ALL_TRAINER_ENTRIES.json`, así que la etapa no terminó.

Una versión 7 puede recuperar esos cuatro archivos si un recibo nuevo autentica el paquete v6 (`7e639172…`), solicitudes, hash de contexto, esquema, 22 cromosomas, continuidad y lista exacta recalculada bajo el algoritmo anterior `a0f580a3…`. Si cambia el constructor, aceptar ese hash anterior sólo mediante una regla explícita y exigir igualdad completa; producir únicamente 0.2 y 1.0 cM. La caché operacional de v6 comprobaba tamaño, miembros ZIP, formas y estabilidad del archivo durante la lectura, pero no ligaba criptográficamente los bytes actuales al SHA de restauración. Entrenamiento futuro exige SHA estricto o instantánea/generación autenticada.

## PRE de preparación LAI

**Dictamen:** apto sólo para implementar y probar preparación de datos. La entrada nueva debe fijar `REF_SCOPE/DECISION.json` (`d9b1ef63…`), `changes_v2/receipt.json` (`0d922deb…`) y cada TSV por `task_id`; se prohíbe descubrir la versión más reciente. Para cada uno de 12 casos debe retirar de BASE exactamente la clave `(cromosoma, posición, REF, ALT)` añadida a EVENT, conservar GT, fase, faltantes, personas y orden, y producir los conteos chr1 BASE=275,918/EVENT=3,610 y chr22 BASE=42,696/EVENT=330. Los artefactos L1 y los catálogos anteriores permanecen inmutables; la revisión usa nombres, estados y recibos nuevos.

La reutilización de FULL sigue **bloqueada**. El registro debe contener 12 casos × 3 generaciones = 36 entradas explícitas y autenticar VCF e índice, simulación, eje y orden de personas, panel, dirección, rol, versión/JAR/contenedor de FLARE, parámetros, semilla y hashes originales. Cualquier diferencia obliga a marcar `RECOMPUTE_REQUIRED`; no se crea un alias. La implementación debe extraerse en un directorio vacío y comprobar importaciones, dependencias transitivas, digest del contenedor y propagación de parámetros antes de una prueba real mínima. Esta admisión no permite ejecutar FLARE, entrenar, seleccionar ni evaluar.

## Estado operativo y fuentes abiertas

El paquete v6 declara imagen por digest `sha256:c03864…351ee99`, Nextflow 26.04.6, un proceso de 2 GB y servicio de 3 GiB; no se pudo inspeccionar el daemon Docker desde este entorno. `Batch.list` fue denegado por IAM, no por autenticación caducada. La identidad exacta de VM/disco, etiqueta `team=frank`, servicio vivo y generaciones GCS requieren la comprobación externa separada antes de cualquier ejecución.

Se abrieron: `AGENTS.md`, `.claude/handoff.md`, controles Obsidia, las skills canónicas `consejo-elfico`, `provenance-red-team`, `graphify` y `nextflow-clean-engineering`; `PRE_SNAPSHOT.json`; `CONTEXT_RECEIPT.json`; `CONTINUITY_V6_STATE.json`; `console.log`; paquete v6, manifiesto y lectores congelados; `genomic_context_runtime.py`; metadatos de 22 almacenes; cuatro archivos de contexto; `DECISION.json`; `changes_v2/receipt.json`; `r02_lai_flare_material.py` y `r02_lai_training_inputs_plan.py`. CodeGraph estaba atrasado respecto del repositorio y cada símbolo se confirmó en el archivo real.
