#!/usr/bin/env python3
"""Prueba funcional de la ruta de reintento por falta de memoria en un worker de loky.

Cuando el kernel termina un worker de joblib/loky, sklearn propaga TerminatedWorkerError al proceso
padre. Solo el caso SIGKILL(-9) se convierte en el código reintentable 42; las demás señales conservan
su comportamiento normal.

Valida las 4 aristas de la ruta nueva:
  1. TerminatedWorkerError con 'SIGKILL(-9)'  -> main() sale exit 42 (reintentable).
  2. TerminatedWorkerError con 'SIGSEGV(-11)' -> NO se remapea; se re-lanza -> exit 1 (fail-closed).
  3. exit 42 dispara UN SOLO reintento a 256 GB/30 h; un 2o fallo -> terminate.
  4. exit 1 (fallo cientifico) NO reintenta.
Los dos primeros casos prueban el flujo real de main() en un subproceso. Los dos últimos comprueban
la estrategia de reintento y su configuración.

Correr en el contenedor del pipeline (stack coherente), NO en login node:
  srun -p cpu -c 2 --mem=4G -t 00:15:00 singularity exec -B "$HOME" <sif> \
    python3 tests/test_m23_oom_retry_policy.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT / "bin"
CONFIG = ROOT / "conf" / "auto_resources.config"
sys.path.insert(0, str(BIN_DIR))


def _make_twe(sig):
    """Construye un TerminatedWorkerError con el formato REAL de loky (joblib _format_exitcodes):
    'The exit codes of the workers are {SIG(-N)}'."""
    from joblib.externals.loky.process_executor import TerminatedWorkerError
    return TerminatedWorkerError(
        "A worker process managed by the executor was unexpectedly terminated. "
        f"The exit codes of the workers are {{{sig}}}")


# --------------------------------------------------------------------------------------------------
# MODO DRIVER: corre el main() REAL con _run parcheado para lanzar la excepcion pedida, y deja que el
# código de salida del proceso sea el resultado. _run se reemplaza antes de leer la matriz.
# --------------------------------------------------------------------------------------------------
if len(sys.argv) >= 3 and sys.argv[1] == "--drive":
    which = sys.argv[2]
    import rare_bench_cv as R

    tmp = tempfile.mkdtemp(prefix="m23drv_")

    def _raise_sigkill(_args):
        raise _make_twe("SIGKILL(-9)")

    def _raise_sigsegv(_args):
        raise _make_twe("SIGSEGV(-11)")

    def _raise_exit1(_args):
        raise SystemExit(1)

    R._run = {"SIGKILL": _raise_sigkill, "SIGSEGV": _raise_sigsegv, "EXIT1": _raise_exit1}[which]
    sys.argv = ["rare_bench_cv.py", "--mode", "fold", "--outdir", tmp]
    R.main()               # debe terminar via sys.exit(...) o propagar la excepcion
    raise SystemExit(97)   # si main() retorno sin salir, la ruta esta rota


# --------------------------------------------------------------------------------------------------
# MODO SUITE
# --------------------------------------------------------------------------------------------------
PY = sys.executable
SELF = Path(__file__).resolve()
FAILS = []


def check(name, cond, detail=""):
    print(f"[test] {'ok   ' if cond else 'FALLO'} {name}" + ("" if cond else f"  ::  {detail}"))
    if not cond:
        FAILS.append(name)


def drive(which):
    r = subprocess.run([PY, str(SELF), "--drive", which], capture_output=True, text=True)
    return r.returncode


import rare_bench_cv as R  # noqa: E402

# --- Arista 1-2 (predicado puro, funcion extraida para testeabilidad) ---
check("predicado: SIGKILL(-9) -> True", R._is_oom_worker_kill(_make_twe("SIGKILL(-9)")) is True)
check("predicado: SIGSEGV(-11) -> False", R._is_oom_worker_kill(_make_twe("SIGSEGV(-11)")) is False)
check("predicado: no-TerminatedWorkerError -> False", R._is_oom_worker_kill(RuntimeError("SIGKILL(-9)")) is False)

# --- Arista 1-2 (control-flow REAL de main(), exit code del proceso) ---
rc = drive("SIGKILL")
check("main(): OOM SIGKILL(-9) -> exit 42", rc == R.EXIT_OOM_WORKER, f"rc={rc} esperado={R.EXIT_OOM_WORKER}")
rc = drive("SIGSEGV")
check("main(): SIGSEGV(-11) -> exit 1 (no remapeado)", rc == 1, f"rc={rc}")
rc = drive("EXIT1")
check("main(): SystemExit(1) cientifico -> exit 1", rc == 1, f"rc={rc}")


# --- Arista 3-4 (politica de reintento: espejo de la closure de conf/auto_resources.config) ---
def retry_decision(exit_status, attempt, max_retries=1):
    """Espejo EXACTO de la closure errorStrategy de conf/auto_resources.config (anclada abajo)."""
    retriable = exit_status is None or exit_status in [137, 140, 143, 42]
    return "retry" if (retriable and attempt <= max_retries) else "terminate"


check("politica: exit 42 attempt 1 -> retry", retry_decision(42, 1) == "retry")
check("politica: exit 42 attempt 2 -> terminate (UN solo reintento)", retry_decision(42, 2) == "terminate")
check("politica: exit 1 -> terminate (no reintenta)", retry_decision(1, 1) == "terminate")
check("politica: exit 139 (SIGSEGV del proceso) -> terminate", retry_decision(139, 1) == "terminate")
check("politica: exit 137 (OOM del proceso) attempt 1 -> retry", retry_decision(137, 1) == "retry")

# --- Ancla al config REAL: el test falla si la politica cambia en disco ---
cfg = CONFIG.read_text()
check("config: errorStrategy lista 42 como reintentable",
      re.search(r"errorStrategy.*137,\s*140,\s*143,\s*42", cfg) is not None)
check("config: reintento a 256 GB",
      re.search(r"rare_bench_cv_fold_retry_memory\s*=\s*'256 GB'", cfg) is not None)
check("config: reintento a 30h",
      re.search(r"rare_bench_cv_fold_retry_time\s*=\s*'30h'", cfg) is not None)
check("config: un solo reintento (max_retries=1)",
      re.search(r"rare_bench_cv_fold_max_retries\s*=\s*1\b", cfg) is not None)

print("\n[test] === RESULTADO ===")
if FAILS:
    print(f"[test] FALLO ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("[test] OK: OOM-en-worker(SIGKILL -9) -> exit 42 -> reintento unico 256 GB/30 h; "
      "SIGSEGV/exit 1 fail-closed sin reintento.")
sys.exit(0)
