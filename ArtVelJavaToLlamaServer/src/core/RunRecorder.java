package core;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Creates and fills one experiment run folder (experiment mode only):
 *
 *   <output.base-directory>/<run_id>/
 *       manifest.json            everything that defines the run (written at start,
 *                                rewritten at the end with passes and summary)
 *       summary.json             headline numbers (also appended to runs.csv)
 *       results.json             records, as in a normal run
 *       results_*.tsv            ICA/CCA ratio, velocities, stenosis, documents (+ telemetry)
 *       results.log, server.log  program log and llama-server console output
 *       raw_responses/<doc>.txt  raw model output per document and call
 *       prompt.txt               the system prompt actually used
 *       settings_machine.properties, settings_experiment.properties   copies
 *       corpus_manifest.tsv      document name, size, SHA-256
 *       document_list.txt        copy of experiment.document-list, if used
 *   <output.base-directory>/runs.csv   one row per run, for comparing runs
 *
 * Thread-safe: document callbacks may come from several threads.
 */
public final class RunRecorder {

    private static final String RUNS_CSV_HEADER = String.join(",",
            "run_id", "label", "status", "started_utc", "machine", "model_label", "model_file",
            "model_size_gb", "prompt_id", "prompt_sha256", "corpus_sha256", "documents",
            "ok", "no_findings", "failed", "too_long", "not_processed",
            "server_start_s", "processing_min", "total_min", "docs_per_min", "docs_per_min_incl_start",
            "median_latency_s", "p95_latency_s", "prompt_tokens", "completion_tokens",
            "prompt_tokens_per_s", "generation_tokens_per_s", "aggregate_generation_tokens_per_s",
            "ctx_size", "parallel_first_pass", "n_gpu_layers", "batch_size", "ubatch_size",
            "flash_attn", "threads", "temperature", "max_tokens", "run_dir");

    private final AppSettings settings;
    private final Path runDir;
    private final String runId;
    private final Instant started = Instant.now();
    private final long startedNanos = System.nanoTime();
    private final Map<String, Object> manifest = new LinkedHashMap<>();
    private final List<Object> passes = new ArrayList<>();
    private final Object lock = new Object();

    private final String promptSha;
    private final String corpusSha;
    private final int documentCount;
    private final String machine;
    private final long modelBytes;

    // summary accumulators (guarded by lock)
    private int ok, noFindings, failed, tooLong;
    private final List<Long> latenciesMs = new ArrayList<>();
    private long sumPromptTokens, sumCompletionTokens, sumPromptN, sumPredictedN;
    private double sumPromptMs, sumPredictedMs;
    private double serverStartSeconds, processingSeconds;
    private int firstPassParallel = -1;
    private int firstPassSlotSize = -1;
    private boolean finished;

    private RunRecorder(AppSettings settings, Path runDir, String runId, String promptSha,
            String corpusSha, int documentCount, String machine, long modelBytes) {
        this.settings = settings;
        this.runDir = runDir;
        this.runId = runId;
        this.promptSha = promptSha;
        this.corpusSha = corpusSha;
        this.documentCount = documentCount;
        this.machine = machine;
        this.modelBytes = modelBytes;
    }

    // -------------------------------------------------------------------------
    // Creation
    // -------------------------------------------------------------------------

    /**
     * Creates the run folder and writes the first manifest.
     *
     * @param effectivePrompt the system prompt that will be sent
     * @param builtInPrompt   the prompt built into the code (to record whether they match)
     * @param documents       the documents that will be processed
     */
    public static RunRecorder create(AppSettings settings, String effectivePrompt, String builtInPrompt,
            List<Path> documents) throws IOException {
        Path base = settings.outputBaseDirectory();
        if (base == null) {
            throw new IllegalStateException("output.base-directory is not set");
        }
        Files.createDirectories(base);

        String machine = settings.machineName().isEmpty() ? HardwareInfo.hostName() : settings.machineName();
        String stamp = DateTimeFormatter.ofPattern("yyMMdd_HHmmss").withZone(java.time.ZoneId.systemDefault())
                .format(Instant.now());
        String baseId = stamp + "_" + sanitize(machine) + "_" + sanitize(settings.modelLabel())
                + "_" + sanitize(settings.experimentLabel());
        String id = baseId;
        Path dir = base.resolve(id);
        int n = 2;
        while (Files.exists(dir)) {
            id = baseId + "-" + n++;
            dir = base.resolve(id);
        }
        Files.createDirectories(dir);
        Files.createDirectories(dir.resolve("raw_responses"));

        String promptSha = sha256(effectivePrompt.getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("prompt.txt"), effectivePrompt.getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("settings_machine.properties"),
                settings.machineFileText().getBytes(StandardCharsets.UTF_8));
        if (settings.experimentFileText() != null) {
            Files.write(dir.resolve("settings_experiment.properties"),
                    settings.experimentFileText().getBytes(StandardCharsets.UTF_8));
        }
        if (settings.documentList() != null && Files.isRegularFile(settings.documentList())) {
            Files.copy(settings.documentList(), dir.resolve("document_list.txt"));
        }

        // Corpus manifest: name, bytes, SHA-256 of each document; the corpus hash is the
        // hash of the "name<TAB>sha" lines, so two runs on identical documents match.
        StringBuilder cm = new StringBuilder("name\tbytes\tsha256\n");
        StringBuilder forHash = new StringBuilder();
        List<Path> sorted = new ArrayList<>(documents);
        Collections.sort(sorted);
        for (Path p : sorted) {
            String name = p.getFileName().toString();
            try {
                byte[] bytes = Files.readAllBytes(p);
                String sha = sha256(bytes);
                cm.append(name).append('\t').append(bytes.length).append('\t').append(sha).append('\n');
                forHash.append(name).append('\t').append(sha).append('\n');
            } catch (IOException e) {
                cm.append(name).append("\t-1\tUNREADABLE\n");
                forHash.append(name).append("\tUNREADABLE\n");
            }
        }
        Files.write(dir.resolve("corpus_manifest.tsv"), cm.toString().getBytes(StandardCharsets.UTF_8));
        String corpusSha = sha256(forHash.toString().getBytes(StandardCharsets.UTF_8));

        long modelBytes = -1;
        Path modelFile = settings.serverConfig().model();
        try {
            modelBytes = Files.size(modelFile);
        } catch (Exception e) {
            // leave -1
        }

        RunRecorder r = new RunRecorder(settings, dir, id, promptSha, corpusSha, documents.size(),
                machine, modelBytes);
        r.buildManifest(effectivePrompt, builtInPrompt, modelFile);
        r.writeManifest("RUNNING", null);
        return r;
    }

    private void buildManifest(String effectivePrompt, String builtInPrompt, Path modelFile) {
        manifest.put("run_id", runId);
        manifest.put("label", settings.experimentLabel());
        manifest.put("notes", settings.experimentNotes());
        manifest.put("started_utc", DateTimeFormatter.ISO_INSTANT.format(started));

        Map<String, Object> mach = new LinkedHashMap<>(HardwareInfo.describe());
        mach.put("machine_name", machine);
        mach.put("hardware_notes", settings.hardwareNotes());
        manifest.put("machine", mach);

        Map<String, Object> llama = new LinkedHashMap<>();
        llama.put("executable", settings.serverConfig().executable().toString());
        llama.put("version_output", HardwareInfo.llamaVersion(settings.serverConfig().executable()));
        llama.put("command_line_first_pass", settings.serverConfig().commandLine());
        manifest.put("llama_server", llama);

        Map<String, Object> model = new LinkedHashMap<>();
        model.put("label", settings.modelLabel());
        model.put("file", modelFile.toString());
        model.put("size_bytes", modelBytes);
        try {
            model.put("modified_utc", DateTimeFormatter.ISO_INSTANT.format(Files.getLastModifiedTime(modelFile).toInstant()));
        } catch (Exception e) {
            model.put("modified_utc", null);
        }
        if (settings.hashModelFile()) {
            try {
                model.put("sha256", sha256File(modelFile));
            } catch (Exception e) {
                model.put("sha256", "unavailable: " + e.getMessage());
            }
        } else {
            model.put("sha256", "not computed (experiment.hash-model-file = false)");
        }
        manifest.put("model", model);

        Map<String, Object> prompt = new LinkedHashMap<>();
        prompt.put("id", settings.promptId());
        prompt.put("file", settings.promptFile() == null ? null : settings.promptFile().toString());
        prompt.put("sha256", promptSha);
        prompt.put("chars", effectivePrompt.length());
        prompt.put("matches_builtin", effectivePrompt.equals(builtInPrompt));
        manifest.put("prompt", prompt);

        manifest.put("extraction_options", settings.extractionOptions().asMap());

        Map<String, Object> corpus = new LinkedHashMap<>();
        corpus.put("documents", documentCount);
        corpus.put("corpus_sha256", corpusSha);
        corpus.put("directory", settings.corpusDirectory() == null ? null : settings.corpusDirectory().toString());
        corpus.put("document_list", settings.documentList() == null ? null : settings.documentList().toString());
        manifest.put("corpus", corpus);

        Map<String, Object> sources = new LinkedHashMap<>();
        sources.put("machine_settings", settings.machineSource());
        sources.put("experiment_settings", settings.experimentSource());
        manifest.put("sources", sources);

        manifest.put("effective_settings", settings.effectiveValues());
    }

    // -------------------------------------------------------------------------
    // Paths for the caller
    // -------------------------------------------------------------------------

    public Path runDir()         { return runDir; }
    public String runId()        { return runId; }
    public Path resultsFile()    { return runDir.resolve("results.json"); }
    public Path serverLogFile()  { return runDir.resolve("server.log"); }

    // -------------------------------------------------------------------------
    // Recording
    // -------------------------------------------------------------------------

    /** One document finished (any status except NOT_PROCESSED); telemetry may be null. */
    public void documentDone(String status, DocumentTelemetry t) {
        synchronized (lock) {
            switch (status) {
                case "SUCCESS":             ok++; break;
                case "SUCCESS_NO_FINDINGS": ok++; noFindings++; break;
                case "FAILED":              failed++; break;
                case "TOO_LONG":            return; // counted at the end; its telemetry is discarded
                default:                    break;
            }
            if (t != null && t.modelCalls > 0) {
                latenciesMs.add(t.wallMs);
                sumPromptTokens += t.promptTokens;
                sumCompletionTokens += t.completionTokens;
                sumPromptN += t.promptN;
                sumPromptMs += t.promptMs;
                sumPredictedN += t.predictedN;
                sumPredictedMs += t.predictedMs;
            }
        }
    }

    /** Appends the raw model output of every call made for the document. */
    public void saveRaw(Path document, DocumentTelemetry t, int pass) {
        if (t == null || t.rawResponses.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder();
        for (DocumentTelemetry.RawResponse r : t.rawResponses) {
            sb.append("===== pass ").append(pass).append(" | ").append(r.stage)
                    .append(" | finish_reason=").append(r.finishReason)
                    .append(" | prompt_tokens=").append(r.promptTokens)
                    .append(" | completion_tokens=").append(r.completionTokens)
                    .append(" | request_ms=").append(r.requestMs).append(" =====\n");
            if (r.reasoning != null && !r.reasoning.isEmpty()) {
                sb.append("----- reasoning -----\n").append(r.reasoning).append("\n----- content -----\n");
            }
            sb.append(r.content == null ? "" : r.content).append("\n\n");
        }
        Path file = runDir.resolve("raw_responses").resolve(safeFileName(document.getFileName().toString()) + ".txt");
        synchronized (lock) {
            try {
                Files.write(file, sb.toString().getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (IOException e) {
                System.err.println("Unable to write " + file + ": " + e.getMessage());
            }
        }
    }

    /** Records one server pass: how long the server took to load and how long documents took. */
    public void recordPass(int pass, int parallel, int slotSize, double serverStartSeconds,
            double processingSeconds, int documentsAttempted, int succeeded, int tooLong, int failed) {
        synchronized (lock) {
            this.serverStartSeconds += serverStartSeconds;
            this.processingSeconds += processingSeconds;
            if (pass == 1) {
                firstPassParallel = parallel;
                firstPassSlotSize = slotSize;
            }
            Map<String, Object> p = new LinkedHashMap<>();
            p.put("pass", pass);
            p.put("parallel", parallel);
            p.put("slot_size_tokens", slotSize);
            p.put("server_start_seconds", serverStartSeconds);
            p.put("processing_seconds", processingSeconds);
            p.put("documents_attempted", documentsAttempted);
            p.put("succeeded", succeeded);
            p.put("too_long", tooLong);
            p.put("failed", failed);
            passes.add(p);
        }
    }

    /**
     * Writes summary.json, the final manifest and a runs.csv row. Call once, in a finally block.
     *
     * @param status       COMPLETED or ABORTED
     * @param tooLongFinal documents still too long after the last pass
     * @param notProcessed documents not reached because the run aborted
     */
    public void finish(String status, int tooLongFinal, int notProcessed) {
        synchronized (lock) {
            if (finished) {
                return;
            }
            finished = true;
            tooLong = tooLongFinal;
            double totalSeconds = (System.nanoTime() - startedNanos) / 1e9;
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("run_id", runId);
            s.put("status", status);
            s.put("documents", documentCount);
            s.put("ok", ok);
            s.put("ok_with_no_findings", noFindings);
            s.put("failed", failed);
            s.put("too_long", tooLong);
            s.put("not_processed", notProcessed);
            s.put("server_start_seconds", serverStartSeconds);
            s.put("processing_minutes", processingSeconds / 60.0);
            s.put("total_minutes", totalSeconds / 60.0);
            int done = ok + failed;
            double perMin = processingSeconds > 0 ? done * 60.0 / processingSeconds : Double.NaN;
            double perMinTotal = totalSeconds > 0 ? done * 60.0 / totalSeconds : Double.NaN;
            s.put("documents_per_minute_processing_only", perMin);
            s.put("documents_per_minute_including_server_start", perMinTotal);
            List<Long> sorted = new ArrayList<>(latenciesMs);
            Collections.sort(sorted);
            double median = percentile(sorted, 0.5) / 1000.0;
            double p95 = percentile(sorted, 0.95) / 1000.0;
            s.put("median_document_seconds", median);
            s.put("p95_document_seconds", p95);
            s.put("prompt_tokens_total", sumPromptTokens);
            s.put("completion_tokens_total", sumCompletionTokens);
            double promptTps = sumPromptMs > 0 ? sumPromptN * 1000.0 / sumPromptMs : Double.NaN;
            double genTps = sumPredictedMs > 0 ? sumPredictedN * 1000.0 / sumPredictedMs : Double.NaN;
            double aggTps = processingSeconds > 0 ? sumCompletionTokens / processingSeconds : Double.NaN;
            s.put("prompt_tokens_per_second_per_request", promptTps);
            s.put("generation_tokens_per_second_per_request", genTps);
            s.put("aggregate_generation_tokens_per_second", aggTps);
            writeFile("summary.json", SimpleJson.write(s));

            manifest.put("finished_utc", DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
            writeManifest(status, s);

            appendRunsCsv(status, perMin, perMinTotal, median, p95, promptTps, genTps, aggTps, totalSeconds,
                    notProcessed);
        }
    }

    private void writeManifest(String status, Map<String, Object> summary) {
        Map<String, Object> m = new LinkedHashMap<>(manifest);
        m.put("status", status);
        m.put("passes", new ArrayList<>(passes));
        if (summary != null) {
            m.put("summary", summary);
        }
        writeFile("manifest.json", SimpleJson.write(m));
    }

    private void appendRunsCsv(String status, double perMin, double perMinTotal, double median, double p95,
            double promptTps, double genTps, double aggTps, double totalSeconds, int notProcessed) {
        LlamaServerConfig c = settings.serverConfig();
        ExtractionOptions o = settings.extractionOptions();
        List<String> row = new ArrayList<>();
        row.add(runId);
        row.add(settings.experimentLabel());
        row.add(status);
        row.add(DateTimeFormatter.ISO_INSTANT.format(started));
        row.add(machine);
        row.add(settings.modelLabel());
        row.add(c.model().getFileName().toString());
        row.add(modelBytes < 0 ? "" : num(modelBytes / 1e9, 2));
        row.add(settings.promptId());
        row.add(promptSha.substring(0, 12));
        row.add(corpusSha.substring(0, 12));
        row.add(Integer.toString(documentCount));
        row.add(Integer.toString(ok));
        row.add(Integer.toString(noFindings));
        row.add(Integer.toString(failed));
        row.add(Integer.toString(tooLong));
        row.add(Integer.toString(notProcessed));
        row.add(num(serverStartSeconds, 1));
        row.add(num(processingSeconds / 60.0, 2));
        row.add(num(totalSeconds / 60.0, 2));
        row.add(num(perMin, 2));
        row.add(num(perMinTotal, 2));
        row.add(num(median, 1));
        row.add(num(p95, 1));
        row.add(Long.toString(sumPromptTokens));
        row.add(Long.toString(sumCompletionTokens));
        row.add(num(promptTps, 1));
        row.add(num(genTps, 1));
        row.add(num(aggTps, 1));
        row.add(Integer.toString(c.ctxSize()));
        row.add(firstPassParallel < 0 ? "" : Integer.toString(firstPassParallel));
        row.add(Integer.toString(c.gpuLayers()));
        row.add(Integer.toString(c.batchSize()));
        row.add(Integer.toString(c.ubatchSize()));
        row.add(c.flashAttn());
        row.add(Integer.toString(c.threads()));
        row.add(num(o.temperature, 2));
        row.add(Integer.toString(o.maxTokens));
        row.add(runDir.toAbsolutePath().toString());

        StringBuilder line = new StringBuilder();
        for (int i = 0; i < row.size(); i++) {
            if (i > 0) {
                line.append(',');
            }
            line.append(csv(row.get(i)));
        }
        line.append('\n');
        Path csv = settings.outputBaseDirectory().resolve("runs.csv");
        try {
            boolean needsHeader = !Files.exists(csv) || Files.size(csv) == 0;
            StringBuilder out = new StringBuilder();
            if (needsHeader) {
                out.append(RUNS_CSV_HEADER).append('\n');
            }
            out.append(line);
            Files.write(csv, out.toString().getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("Unable to append to " + csv + ": " + e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void writeFile(String name, String text) {
        try {
            Files.write(runDir.resolve(name), text.getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            System.err.println("Unable to write " + runDir.resolve(name) + ": " + e.getMessage());
        }
    }

    private static double percentile(List<Long> sorted, double p) {
        if (sorted.isEmpty()) {
            return Double.NaN;
        }
        int idx = (int) Math.ceil(p * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, idx)));
    }

    private static String num(double v, int decimals) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return "";
        }
        return String.format(Locale.ROOT, "%." + decimals + "f", v);
    }

    private static String csv(String s) {
        if (s == null) {
            return "";
        }
        if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }

    private static String sanitize(String s) {
        String t = s == null ? "" : s.replaceAll("[^A-Za-z0-9._-]+", "-");
        t = t.replaceAll("^-+|-+$", "");
        if (t.isEmpty()) {
            t = "x";
        }
        return t.length() > 40 ? t.substring(0, 40) : t;
    }

    private static String safeFileName(String s) {
        return s.replaceAll("[\\\\/:*?\"<>|]", "_");
    }

    private static String sha256(byte[] data) {
        try {
            return hex(MessageDigest.getInstance("SHA-256").digest(data));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static String sha256File(Path file) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] buf = new byte[1 << 20];
        try (java.io.InputStream in = Files.newInputStream(file)) {
            int n;
            while ((n = in.read(buf)) > 0) {
                md.update(buf, 0, n);
            }
        }
        return hex(md.digest());
    }

    private static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder();
        for (byte x : b) {
            sb.append(String.format("%02x", x));
        }
        return sb.toString();
    }
}
