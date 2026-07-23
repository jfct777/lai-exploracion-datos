# DNABR QC

Pipeline de control de calidad y análisis exploratorio de datos genómicos,
implementado con Nextflow DSL2.

## Requisitos

- Nextflow
- Docker, Singularity o Apptainer
- Python 3 para los módulos de agregación y visualización

## Ejecución

Los datos de entrada y las rutas de salida se indican mediante parámetros:

```bash
nextflow run main.nf \
  --vcf_dir /ruta/a/vcf \
  --ref_fasta /ruta/a/referencia.fasta \
  --outdir /ruta/a/resultados
```

La configuración de recursos se encuentra en `conf/`. Los perfiles disponibles
para ejecución local o en un clúster se definen en `nextflow.config`.

Los datos clínicos, metadatos individuales y resultados de ejecución no forman
parte de este repositorio.
