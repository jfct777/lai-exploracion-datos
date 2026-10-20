# R02 STRUCTURE — revisión de procedencia posterior a la implementación

Fecha: 2026-10-19 UTC. Consejo Élfico 0.4.1, lente de procedencia y operación, **UNQUALIFIED**. Esta revisión documenta preparación de entradas; no constituye entrenamiento, evaluación ni evidencia biológica.

## Resultado implementado y probado

`bin/r02_structure_reader_limits.py` recorrió sólo `manifest.json` y `map.tsv` de los 22 cromosomas. `READER_LIMITS_V1.json` (`d5fb2201e26f41d3ddc29a25fd93af7cfe4fd2cbb4042650b46e994dc9a463f8`) fija 109568 filas para atención local de 0.2 cM y 187392 para 1.0 cM; no abrió fragmentos de genotipos. `r02_event_mmap_reader.MappedEventChromosome` recibe ese límite por solicitud sin modificar los almacenes congelados ni el límite de memoria de atención.

`r02_structure_person_batch_builder.budget_tiles` conserva el cálculo exacto anterior mientras cabe y busca un extremo físico menor cuando un intervalo candidato excede el límite autenticado. Se conservaron las 8 personas por lote, 22 cromosomas, 88 vistas, 6 contextos, radios, profundidades, papeles FIT/SELECT y exclusiones. EVALUATE permanece cerrado.

Pasaron cinco pruebas unitarias, compilación de cinco módulos, `git diff --check`, análisis `nextflow -preview`, extracción en un directorio vacío, comprobación de 28 hashes e importaciones dentro del contenedor fijado. El preflight real `/mnt/disks/r02-neural-cx/structure-cz/preflight-v7-20261019T2215Z/PREFLIGHT.json` (`32b127315bfd5de2fd04fd7c2f3f99f03514f2b750625a490547e357cce4929f`) comprobó 22 almacenes, 88 conjuntos FIT/SELECT, 528 identidades y 6 contextos sin leer matrices.

## Incidencia v7 preservada

La primera ejecución v7 fue detenida intencionalmente con SIGTERM después de 494.005 s de servicio y 471.680 s del comando. No fue OOM, deadline ni fallo de Nextflow. `prepare` volvía a ejecutar `budget_tiles` sobre cuatro inventarios v6 ya terminados; el segundo exigió aproximadamente 145 s. El registro es `.claude/runs/r02-neural-next-20261019cz/STRUCTURE/V7_INTENTIONAL_STOP.json` (`e1b426ec295d37c48ac33ba27cdedeaee305d76bb0a86e506e4756735a93bc16`). Los intentos previos sumaban 455.031 s; el registro separa ambas cantidades.

El `.tgz` ejecutado tenía SHA-256 `1455fadc5ceaaf92a62cd6b597b1d28f6fec301e09923008dc5317d9200bfba6`. Su copia binaria fue sobrescrita antes de separar versiones y no se afirma retención byte a byte. Sí se preservaron y reautenticaron sus 28 archivos exactos en `package-v7`: `PACKAGE_MANIFEST.json` `3e3df6b37435a5c1489f29158777b1a52e10c7ebb70852b3de1324b14931e520` y `PACKAGE_SHA256.txt` `44dfb16993857b2c42668b7c677a2ff8051499fa14f6bb59802eacdd1bb31430`. Esta incidencia establece que un paquete ejecutado debe congelarse antes de crear la versión siguiente.

## Ejecución v8 y recuperación exacta

`V8_RECOVERY_ADMISSION.json` (`f54161c148eb615be54d12b1809855847269e5dee2e6d626f5e5dae775e6c27f`) permite reutilizar exactamente cuatro inventarios sólo si coinciden el archivo v6 `7e639172…`, el constructor v6 `a0f580a3…`, el estado sin entrenamiento/EVALUATE, las cuatro identidades, los cuatro hashes y la cobertura de 22 cromosomas. Sólo se calculan los contextos pendientes de 0.2 y 1.0 cM.

Paquete v8: `r02-structure-lazy-input-package-v8.tgz` `96916e78e88df147a4323bbdbb3fd625739f6986eff6ce195cdd9a28fa05b07a`; manifiesto `27fffa3fa24f411763471340815a0c22429357d7078c053ba12460f414c71962`; lista `41698cca8216736bba0bc9ea2a202f54be6c650bb669f3522cb86b2ad26bc5c3`.

V8 terminó los seis inventarios, el índice de 528 solicitudes y las dos mediciones. Falló únicamente `FINALIZE_PREPARATION` porque Groovy convirtió `+'\n'` de un bloque Python incrustado en un salto literal y produjo `SyntaxError`. `V8_FINALIZER_FAILURE.json` (`2e1d7a25ba6695e60063dfe0c6f1bf9f05bb2e93c056639a43933a93d898c336`) registra 451.677 s de servicio y 450.507 s del comando. No fue falta de memoria ni de espacio. Se recuperaron, sin repetir cálculo, los 16 JSON científicos exactos del trabajo `42/fe2690`; `V8_FINALIZATION_RECOVERY.json` es `6727f3dc9487c0644a0057e517b3627d640dff092dd57d04a444d8bdc3809d97`.

Las mediciones reales leen un lote de 8 personas. Para 0.2 cM en cromosoma 11 se observaron 108954 filas raras, 19535 comunes, 80581176 bytes de matrices NumPy, RSS máximo 268406784 bytes y 30.630 s, bajo el límite autenticado 109568. Para 1.0 cM en cromosoma 3 se observaron 187297 filas raras, 40573 comunes, 142716752 bytes de matrices, RSS máximo 347168768 bytes y 31.609 s, bajo el límite 187392. Ambos valores quedan muy por debajo del umbral de 2 GiB. La caché basada en recibos abrió 253 y 446 fragmentos respectivamente, pero no volvió a calcular SHA-256 sobre los 195593032185 bytes: esa limitación permanece explícita; el entrenamiento posterior conserva su lectura estricta separada.

## V9 terminada y validación independiente

La corrección movió la finalización a `bin/r02_structure_finalize_preparation.py`, limitó el manifiesto a `prepared-final/`, `materialized-final/` y `measurements-final/`, y eliminó Python incrustado de los flujos afectados. El paquete ejecutado `r02-structure-lazy-input-package-v9.tgz` tiene SHA-256 `06a13e228b3af586d21ca5a250fb6b7f29348bb42c1f4f9028aacaf4c3f46f1f`; `PACKAGE_MANIFEST.json` es `5975bc9fd4ee129f1a906fae0185ab19408916447b490794f620ae6a1e2d4782` y `PACKAGE_SHA256.txt` `6936c149c3360f7fdb5f3eb9a621e6a633a9d119aa42a0cc919cb4f53694b0e3`.

`dnabr-r02-structure-cz-v9.service` terminó en 67.747 s. La finalización reutilizó los productos autenticados de V8 y publicó exactamente 16 archivos. `FINAL.json` es `4f6c3beea8f5724766886b14f5d6d52c51ebc211b2a6ff20a49627b76600f228`; `prepared-final/index.json` es `1b7433d79197202be78676e30af637c2abbf7cbc3101c46a013a78805f30f7bb`. Después, Nextflow materializó una vez las 528 solicitudes en 42.855 s con salida cero. `ALL_TRAINER_ENTRIES.json` es `9e763c78da50ae0520f9e9e2bfded8d6366c4b35455aeaa7b57a734eea0237ae`.

`validate_v9_completion.py` (`9796650acf48ae209872d5fddb2662c979b092c8708092a8b4d62e7aec7295d2`) releyó sólo JSON y comprobó: seis contextos, 22 almacenes, 88 vistas, seis solicitudes por vista, 528 identidades únicas, 1056 archivos FIT/SELECT y sus hashes; 264 solicitudes Deep Sets y 264 de atención local; cuatro contextos recuperados y dos calculados; 704 archivos de etapa con límite 100000, 176 con 109568 y 176 con 187392; `reader_limits_sha256=d5fb2201…`; ninguna copia física de genotipos u objetivos. El recibo `V9_COMPLETION.json` es `d4c5dede457bd87630b71928aa81279d2290dae6b3ed15ada00ca5afcf5b40c6`.

## Alcance científico y límites

El resultado permite que una ejecución futura admitida localice las entradas FIT y SELECT de las 88 vistas y seis configuraciones sin duplicar matrices. Conserva `source_minor`, REF/ALT, faltantes, fase, las 1460 personas FIT, 476 SELECT, lotes de 8 personas y EVALUATE cerrado. No se importó el módulo de entrenamiento durante las mediciones, no se creó modelo, no se leyó la respuesta objetivo y no se calculó pérdida, selección ni resultado biológico. `CONTINUITY_V9_STATE.json` (`5accad617062147c040a7359b1c189837e8b0a889a76dfcf077388a8d8c5fb7a`) termina como `COMPLETE_MAIN_AND_528_TRAINER_ENTRIES_NO_TRAINING`.

Fuentes abiertas para esta revisión posterior: `FINAL.json`, `prepared-final/index.json`, los 528 `trainer-entries.json`, sus 1056 solicitudes FIT/SELECT, `ALL_TRAINER_ENTRIES.json`, `CONTINUITY_V9_STATE.json`, los cierres y registros de consola v8/v9, `V7_INTENTIONAL_STOP.json`, `V8_FINALIZER_FAILURE.json`, `V8_FINALIZATION_RECOVERY.json`, los manifiestos de paquetes v7-v9 y las dos mediciones publicadas. No se abrieron genotipos crudos durante la validación posterior.
