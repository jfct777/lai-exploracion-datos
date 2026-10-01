#!/usr/bin/env bash
# Produce two explicitly defined common-SNV GRMs with established tools.
# These are descriptive genotype similarities, not PC-Relate kinship estimates.
set -euo pipefail

vcf=""; sample_file=""; chrom=""; expected_source=""; expected_samples=""
output=""; threads=2; memory_mb=4096; min_maf=0.05; max_missing=0.02
ld_configs='primary:500:0.2,sensitivity:200:0.5'
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vcf) vcf=$2; shift 2 ;;
        --samples) sample_file=$2; shift 2 ;;
        --chrom) chrom=${2#chr}; shift 2 ;;
        --expected-source-samples) expected_source=$2; shift 2 ;;
        --expected-samples) expected_samples=$2; shift 2 ;;
        --output-dir) output=$2; shift 2 ;;
        --threads) threads=$2; shift 2 ;;
        --memory-mb) memory_mb=$2; shift 2 ;;
        --min-maf) min_maf=$2; shift 2 ;;
        --max-missing) max_missing=$2; shift 2 ;;
        --ld-configs) ld_configs=$2; shift 2 ;;
        *) printf 'Unknown common-GRM option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$vcf" && -n "$sample_file" && -n "$chrom" && -n "$output" && -n "$expected_source" && -n "$expected_samples" ]] || {
    printf 'Required: --vcf --samples --chrom --expected-source-samples --expected-samples --output-dir\n' >&2
    exit 2
}
[[ ! -e "$output" ]] || { printf 'Output exists; refuse overwrite: %s\n' "$output" >&2; exit 2; }
for executable in bcftools plink2 python3; do command -v "$executable" >/dev/null; done

# The original input is read-only. The following JSON records the selected
# cohort and all parameters before variant processing or statistical fitting.
python3 - "$vcf" "$sample_file" "$chrom" "$expected_source" "$expected_samples" "$output" "$threads" "$memory_mb" "$min_maf" "$max_missing" "$ld_configs" <<'PY'
import hashlib, json, pathlib, re, sys
import pysam
vcf, sample_path, chrom, source_n, analytical_n, out, threads, memory, maf, missing, ld = sys.argv[1:]
assert chrom.isdigit() and 1 <= int(chrom) <= 22, "Only autosomes permitted"
assert int(threads) > 0 and int(memory) >= 640, "Invalid resource settings"
assert 0 < float(maf) <= .5 and 0 <= float(missing) < 1, "Invalid common thresholds"
samples = pathlib.Path(sample_path).read_text().splitlines()
assert len(samples) == int(analytical_n) and len(set(samples)) == len(samples), "Wrong analytical cohort"
assert all(s and not any(c.isspace() for c in s) for s in samples), "Invalid ID"
def digest(values): return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
with pysam.VariantFile(vcf) as source:
    original = list(source.header.samples)
    assert len(original) == int(source_n) and set(samples) <= set(original), "Wrong source cohort"
    matches = [line for line in str(source.header).splitlines() if line.startswith("##dnabr_original_alleles=")]
    assert matches == ["##dnabr_original_alleles=v1"], "Input must preserve M01 original-allele certification"
    assert "ORIG_NALLELES" in source.header.info and "GT" in source.header.formats, "Missing original-count or GT field"
configs = []
for item in ld.split(","):
    name, window, r2 = item.split(":")
    assert re.fullmatch(r"[A-Za-z0-9_-]+", name), "Unsafe LD configuration name"
    assert int(window) > 0 and 0 < float(r2) < 1, "Invalid LD parameters"
    configs.append(dict(name=name, window_kb=int(window), step_variants=1, r2=float(r2)))
assert len(set(c["name"] for c in configs)) == len(configs), "Repeated LD configuration"
root = pathlib.Path(out)
root.mkdir(parents=True, exist_ok=False)
payload = dict(schema="r02_common_grm_v1", chromosome=chrom, input_vcf=vcf,
    source_sample_n=len(original), source_samples_sha256=digest(original),
    analytical_sample_n=len(samples), analytical_samples_sha256=digest(samples),
    min_maf=float(maf), max_missing=float(missing), ld_configurations=configs,
    frequency_scope="complete called GT in the fixed analytical cohort; descriptive full-cohort frequencies",
    missingness="partial GT becomes missing; no imputation; no new DP/GQ genotype thresholds",
    original_biallelic=True, site_filter="PASS", threads=int(threads), memory_mb=int(memory),
    scope="Descriptive common-genotype similarity, not recent-kinship estimation, population confirmation or held-out prediction",
    threshold_status="declared engineering QC; human LD settings from PLINK documentation, not DNABR optima",
    reference="https://www.cog-genomics.org/plink/2.0/ld")
(root / "preparation.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

# All full-source VCF operations are streamed. Only the frequency/missingness-
# filtered common subset is materialized; no complete genotype-only VCF copy.
bcftools view --threads "$threads" --samples-file "$sample_file" -f PASS \
    -m2 -M2 -v snps -i 'INFO/ORIG_NALLELES=2' -Ou "$vcf" \
    | bcftools +setGT -Ou -- -t './x' -n . \
    | bcftools +fill-tags -Ou -- -t AC,AN,F_MISSING \
    | bcftools view --threads "$threads" \
        -i "MAF>=${min_maf} && INFO/F_MISSING<=${max_missing}" -Ou \
    | bcftools annotate -x FORMAT,^FORMAT/GT --threads "$threads" \
        -Ob -o "$output/common.filtered.bcf"
bcftools index --threads "$threads" "$output/common.filtered.bcf"

plink2 --bcf "$output/common.filtered.bcf" --double-id --vcf-require-gt \
    --vcf-half-call missing --chr "$chrom" --snps-only just-acgt --max-alleles 2 \
    --set-all-var-ids '@:#:$r:$a' --maf "$min_maf" --geno "$max_missing" \
    --threads "$threads" --memory "$memory_mb" --make-pgen erase-phase erase-dosage \
    --out "$output/common"

IFS=',' read -r -a config_items <<< "$ld_configs"
for configuration in "${config_items[@]}"; do
    IFS=':' read -r name window r2 <<< "$configuration"
    config_dir="$output/$name"
    mkdir "$config_dir"
    plink2 --pfile "$output/common" --indep-pairwise "${window}kb" 1 "$r2" \
        --indep-order 2 --threads "$threads" --memory "$memory_mb" --out "$config_dir/ld"
    [[ -s "$config_dir/ld.prune.in" ]] || { printf 'LD pruning retained no variants\n' >&2; exit 3; }
    # A separate invocation is mandatory: --indep-pairwise does NOT itself
    # restrict other calculations in that invocation to its retained variants.
    plink2 --pfile "$output/common" --extract "$config_dir/ld.prune.in" \
        --make-grm-bin --threads "$threads" --memory "$memory_mb" --out "$config_dir/grm"
    python3 - "$output" "$name" "$chrom" "$window" "$r2" "$threads" "$memory_mb" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, name, chrom, window, r2, threads, memory = sys.argv[1:]
root, d = pathlib.Path(out), pathlib.Path(out) / name
prep = json.loads((root / "preparation.json").read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
command = ["plink2", "--pfile", str(root / "common"), "--extract", str(d / "ld.prune.in"),
    "--make-grm-bin", "--threads", threads, "--memory", memory, "--out", str(d / "grm")]
payload = dict(tool="PLINK2", version=subprocess.check_output(["plink2", "--version"], text=True).strip(),
    command=command, marker_selection=dict(original_biallelic=True, PASS=True,
        min_maf=prep["min_maf"], max_missing=prep["max_missing"], window_kb=int(window),
        step_variants=1, r2=float(r2), indep_order=2,
        retained_snps=sum(bool(x.strip()) for x in (d/"ld.prune.in").read_text().splitlines())),
    frequency_scope=prep["frequency_scope"], missingness=prep["missingness"], chromosome=chrom,
    analytical_samples_sha256=prep["analytical_samples_sha256"],
    filtered_common_pvar_sha256=sha(root/"common.pvar"), retained_markers_sha256=sha(d/"ld.prune.in"),
    preparation_sha256=sha(root/"preparation.json"),
    bcftools_version=subprocess.check_output(["bcftools", "--version"], text=True).splitlines()[0])
(d / "method.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
done
printf 'COMMON_GRM_COMPLETE chr%s: %s\n' "$chrom" "$output"
