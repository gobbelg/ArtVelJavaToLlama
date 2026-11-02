@echo off
rem ============================================================================
rem sample_resources_windows.bat
rem
rem Samples CPU, RAM, the llama-server process, and (if an Nvidia GPU and
rem nvidia-smi are present) GPU use, VRAM, power and temperature, and appends one
rem CSV row per sample. Works on the Xeon + A2 PC and on the CPU-only i5 PC: when
rem there is no GPU the GPU columns are simply left empty, so the layout is the same.
rem
rem USAGE (run in its OWN command window, start it before the batch, Ctrl+C to stop):
rem   sample_resources_windows.bat [output.csv] [seconds] [process-name]
rem     output.csv    default resource_samples.csv (appended to; header written once)
rem     seconds       pause between samples, default 5 (each sample itself takes ~2 s)
rem     process-name  default llama-server (the .exe name without ".exe")
rem
rem COLUMNS
rem   sample_time_local        local clock time from typeperf (the run's manifest.json
rem                            uses UTC: convert, or compare with the PC clock)
rem   cpu_total_pct            whole-machine CPU use, 0-100
rem   mem_available_mb         free RAM in MB (smaller = more in use)
rem   llama_cpu_pct            llama-server CPU use; 100 = one core fully busy, so it can
rem                            exceed 100 on multi-core machines. Empty when it is not running
rem   llama_working_set_bytes  llama-server RAM in use (bytes). Empty when not running
rem   gpu_util_pct, gpu_mem_used_mib, gpu_mem_total_mib, gpu_power_w, gpu_temp_c
rem                            from nvidia-smi (first line only if there are several GPUs)
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

set "OUT=%~1"
if "%OUT%"=="" set "OUT=resource_samples.csv"
set "INTERVAL=%~2"
if "%INTERVAL%"=="" set "INTERVAL=5"
set "PROC=%~3"
if "%PROC%"=="" set "PROC=llama-server"
set /a PINGS=INTERVAL+1

rem --- find nvidia-smi (optional) ---
set "SMI="
where nvidia-smi >nul 2>&1 && set "SMI=1"
if not defined SMI if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "PATH=%PATH%;%ProgramFiles%\NVIDIA Corporation\NVSMI"
if not defined SMI where nvidia-smi >nul 2>&1 && set "SMI=1"

if not exist "%OUT%" echo sample_time_local,cpu_total_pct,mem_available_mb,llama_cpu_pct,llama_working_set_bytes,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,gpu_temp_c>"%OUT%"

echo Sampling every ~%INTERVAL% s to %OUT%   (process: %PROC%)
if defined SMI (echo GPU: nvidia-smi found, GPU columns will be filled.) else (echo GPU: nvidia-smi not found, GPU columns will be empty.)
echo Press Ctrl+C to stop.

:loop
set "T="
set "CPU="
set "MEM="
set "PCPU="
set "PWS="
set "G="

for /f "tokens=1-3 delims=," %%a in ('typeperf "\Processor(_Total)\%% Processor Time" "\Memory\Available MBytes" -sc 1 2^>nul ^| findstr /v /i "PDH command Error" ^| findstr /r "[0-9]"') do (
  set "T=%%~a"
  set "CPU=%%~b"
  set "MEM=%%~c"
)

rem Process counters only exist while llama-server is running; otherwise typeperf fails and these stay empty.
for /f "tokens=2-3 delims=," %%b in ('typeperf "\Process(%PROC%)\%% Processor Time" "\Process(%PROC%)\Working Set" -sc 1 2^>nul ^| findstr /v /i "PDH command Error" ^| findstr /r "[0-9]"') do (
  set "PCPU=%%~b"
  set "PWS=%%~c"
)
if defined PWS for /f "delims=." %%w in ("!PWS!") do set "PWS=%%w"

if defined SMI (
  for /f "usebackq delims=" %%g in (`nvidia-smi --query-gpu^=utilization.gpu^,memory.used^,memory.total^,power.draw^,temperature.gpu --format^=csv^,noheader^,nounits 2^>nul`) do (
    if not defined G set "G=%%g"
  )
)
if defined G (set "G=!G: =!") else (set "G=,,,,")

>>"%OUT%" echo !T!,!CPU!,!MEM!,!PCPU!,!PWS!,!G!

ping -n %PINGS% 127.0.0.1 >nul
goto loop
