#!/usr/bin/env Rscript
# Preserve BED dosage/missingness; compute LD selection only in the open roles.
suppressPackageStartupMessages({library(SNPRelate); library(gdsfmt); library(jsonlite); library(digest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=6L || !identical(args[c(1,3,5)],c('--contract','--contract-sha256','--outdir'))) stop('Explicit contract/hash/output required')
hash <- function(p) digest(file=p,algo='sha256')
if (hash(args[2])!=args[4]) stop('Contract hash mismatch')
c <- fromJSON(args[2],simplifyVector=FALSE)
if (c$schema!='r02_lai_panel_gds_input_v1') stop('Wrong schema')
for (d in c$inputs) if (!file.exists(d$path) || hash(d$path)!=d$sha256) stop('Input authentication failed')
p <- c$ld
if (!is.numeric(p$r2) || p$r2<=0 || p$r2>=1 || p$window_bp<1 || p$threads<1 || p$method!='corr' || p$start!='first') stop('Invalid LD parameters')
out <- args[6]
if (file.exists(out)) stop('Output already exists')
dir.create(out,recursive=FALSE,mode='0700'); out<-normalizePath(out)
snpgdsBED2GDS(c$inputs$bed$path,c$inputs$fam$path,c$inputs$bim$path,file.path(out,'panel.gds'),family=FALSE,verbose=TRUE)
g<-snpgdsOpen(file.path(out,'panel.gds'),readonly=TRUE)
roles<-read.delim(c$inputs$roles$path,colClasses='character',check.names=FALSE)
ids<-read.gdsn(index.gdsn(g,'sample.id'))
if (anyDuplicated(ids) || !setequal(ids,roles$sample_id) || any(!roles$role %in% c('REF_TRAIN','SOURCE_VALID'))) stop('Unauthorized or mismatched sample universe')
sel<-snpgdsLDpruning(g,autosome.only=TRUE,remove.monosnp=TRUE,maf=p$maf,missing.rate=p$max_missing,
  method=p$method,slide.max.bp=p$window_bp,ld.threshold=sqrt(p$r2),start.pos=p$start,num.thread=p$threads,verbose=TRUE)
markers<-unname(unlist(sel)); if (length(markers)<=max(unlist(c$kinship$pc_counts))) stop('Insufficient markers')
saveRDS(markers,file.path(out,'snp_ids.rds'),compress=FALSE)
counts<-data.frame(chromosome=names(sel),retained_snps=vapply(sel,length,integer(1)))
write.table(counts,file.path(out,'ld_counts.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
snpgdsClose(g)
file.copy(c$inputs$roles$path,file.path(out,'roles.tsv'),overwrite=FALSE)
desc<-function(name) list(path=file.path(out,name),sha256=hash(file.path(out,name)))
contract<-list(schema='r02_lai_panel_kinship_v1',inputs=list(gds=desc('panel.gds'),roles=desc('roles.tsv'),snp_ids=desc('snp_ids.rds')),parameters=c$kinship)
write_json(contract,file.path(out,'kinship_contract.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
files<-list.files(out)
manifest<-list(schema='r02_lai_panel_gds_v1',status='COMPLETE_PRIVATE_GDS_LD_NOT_KINSHIP_VALIDATION',
  input_contract_sha256=args[4],inputs=c$inputs,ld=p,ld_correlation_threshold=sqrt(p$r2),samples=length(ids),markers=length(markers),
  software=list(R=as.character(getRversion()),SNPRelate=as.character(packageVersion('SNPRelate'))),
  source_test_genotypes_opened=FALSE,phase_not_preserved_in_gds=TRUE,phase_sources_unchanged=TRUE,
  outputs_sha256=setNames(lapply(files,function(f)hash(file.path(out,f))),files))
write_json(manifest,file.path(out,'manifest.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
