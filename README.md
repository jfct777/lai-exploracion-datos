# DNABR QC

Pipeline de control de calidad y análisis exploratorio de datos genómicos,
implementado con Nextflow DSL2.

## Requisitos

- Nextflow
- Slurm
- Singularity o Apptainer
- Python 3 y R para los módulos estadísticos

## Ejecución

Las rutas de entrada, salida y referencia se indican mediante parámetros. Los
perfiles y recursos del clúster se definen en `nextflow.config`, `hpc.config` y
`conf/`.

```bash
nextflow run main.nf \
  -profile slurm_singularity \
  --vcf_dir /ruta/a/vcf \
  --ref_fasta /ruta/a/referencia.fasta \
  --outdir /ruta/a/resultados
```

Los datos clínicos, metadatos individuales, resultados y registros de ejecución
se mantienen fuera del repositorio.
