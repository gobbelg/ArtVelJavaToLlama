package config;

import server.LlamaServerConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

/**
 * All values that differ from one computer to the next, or that you vary between
 * experiments, read from external properties files chosen at start-up. The same
 * build then runs unchanged on every computer and for every experiment.
 *
 * TWO LAYERS
 *   MACHINE file     what is fixed for one computer: llama-server paths, GPU layers,
 *                    threads, corpus and output folders. Required (or the built-in VA
 *                    defaults are used).
 *   EXPERIMENT file  what you are testing: model, prompt, sampling, ctx-size, parallel,
 *                    limits. Optional. Any key may appear in either file; a key in the
 *                    experiment file overrides the same key in the machine file.
 *
 * Giving an experiment file turns on EXPERIMENT MODE (see MainVascular): no console
 * prompts, a fresh run folder under output.base-directory with a manifest, per-document
 * timing, raw model responses and a summary, and a row appended to runs.csv.
 * Without one, the program behaves as before (interactive prompts, one output file).
 *
 * HOW THE FILES ARE CHOSEN (first match wins, for each file separately)
 *   machine:     --config=PATH, -Dvascular.config=PATH, env VASCULAR_CONFIG, then a
 *                console prompt (Enter = built-in VA Xeon + A2 defaults)
 *   experiment:  --experiment=PATH, -Dvascular.experiment=PATH, env VASCULAR_EXPERIMENT;
 *                otherwise none (no prompt)
 *   (--config and --experiment are consumed by VascularRunner before the run type is
 *   read; in Eclipse use Run Configurations > Arguments > VM arguments for -D options.)
 *
 * FILE FORMAT: one  key = value  per line. Lines starting with # or ! are comments.
 * This is NOT java.util.Properties: backslashes are NOT escape characters, so a
 * Windows path such as P:\ORD\Models\x.gguf can be pasted as-is (forward slashes
 * also work). Surrounding double quotes are optional. A relative path is resolved
 * against the folder containing the file that sets it; a leading ~ means the user's
 * home folder. Unknown keys produce a warning; bad or missing values are all
 * reported together in a single error.
 *
 * Keys (see the sample files for explanations):
 *   server.executable  server.model            (required)
 *   server.host  server.port  server.n-gpu-layers  server.ctx-size
 *   server.batch-size  server.ubatch-size  server.flash-attn  server.threads
 *   server.cont-batching  server.extra-args  server.startup-timeout-minutes
 *   run.default-parallel  run.auto-second-pass  run.second-pass-parallel  run.model-name
 *   run.machine-name  run.hardware-notes
 *   corpus.directory  corpus.extension
 *   http.request-timeout-minutes  http.connect-timeout-seconds
 *   output.base-directory                      (required in experiment mode)
 *   experiment.label  experiment.notes  experiment.model-label
 *   experiment.document-list  experiment.hash-model-file
 *   prompt.file  prompt.id
 *   sampling.temperature  sampling.top-k  sampling.top-p  sampling.repeat-penalty  sampling.seed
 *   generation.max-tokens  generation.retry-max-tokens  generation.repair-max-tokens
 *   generation.disable-thinking  generation.use-grammar
 *   budget.chars-per-token  budget.prompt-overhead-tokens  budget.safety-margin-tokens
 *   harness.sweep-values  harness.slot-size  harness.sample-size  harness.warm-up
 *
 * Still in code on purpose: the JSON schema / grammar structure (it must match the
 * prompt's vocabulary and the validators). The prompt TEXT is external when prompt.file is set.
 */
public final class AppSettings {

    public static final String CONFIG_PROPERTY = "vascular.config";
    public static final String CONFIG_ENV = "VASCULAR_CONFIG";
    public static final String EXPERIMENT_PROPERTY = "vascular.experiment";
    public static final String EXPERIMENT_ENV = "VASCULAR_EXPERIMENT";

    private static final Set<String> KNOWN_KEYS = new TreeSet<>(Arrays.asList(
            "server.executable", "server.model", "server.host", "server.port",
            "server.n-gpu-layers", "server.ctx-size", "server.batch-size",
            "server.ubatch-size", "server.flash-attn", "server.threads",
            "server.cont-batching", "server.extra-args", "server.startup-timeout-minutes",
            "run.default-parallel", "run.auto-second-pass", "run.second-pass-parallel",
            "run.model-name", "run.machine-name", "run.hardware-notes",
            "corpus.directory", "corpus.extension",
            "http.request-timeout-minutes", "http.connect-timeout-seconds",
            "output.base-directory",
            "experiment.label", "experiment.notes", "experiment.model-label",
            "experiment.document-list", "experiment.hash-model-file",
            "prompt.file", "prompt.id",
            "sampling.temperature", "sampling.top-k", "sampling.top-p",
            "sampling.repeat-penalty", "sampling.seed",
            "generation.max-tokens", "generation.retry-max-tokens",
            "generation.repair-max-tokens", "generation.disable-thinking",
            "generation.use-grammar",
            "budget.chars-per-token", "budget.prompt-overhead-tokens",
            "budget.safety-margin-tokens",
            "harness.sweep-values", "harness.slot-size", "harness.sample-size",
            "harness.warm-up"));

    /** The Xeon Silver 4310 + A2 values that used to be hard-coded. */
    private static final String VA_DEFAULTS =
            "server.executable = P:/ORD_Nguyen_202206062D/Glenn/LlamaAppInstall_261008/llama-b11514-bin-win-cuda-12.4-x64/llama-server.exe\n"
          + "server.model = O:/OMHSP_CONDUIT/NLP/Glenn/LLMs/Models/gemma-4-E4B-it-IQ4_NL/gemma-4-E4B-it-IQ4_NL.gguf\n"
          + "server.n-gpu-layers = 999\n"
          + "server.ctx-size = 49152\n"
          + "server.batch-size = 2048\n"
          + "server.ubatch-size = 512\n"
          + "server.flash-attn = on\n"
          + "server.threads = 12\n"
          + "server.extra-args = --load-mode mlock\n"
          + "run.default-parallel = 4\n"
          + "corpus.directory = P:/ORD_Nguyen_202206062D/Glenn/DataAnalysis/LLMs/TestCorpus2\n";

    /** One setting with the file that set it (so relative paths resolve per file). */
    private static final class Entry {
        final String value;
        final Path baseDir;
        Entry(String value, Path baseDir) {
            this.value = value;
            this.baseDir = baseDir;
        }
    }

    private final String source;
    private final String machineSource;
    private final String experimentSource;       // null when there is no experiment file
    private final String machineText;
    private final String experimentText;
    private final Map<String, String> effectiveValues;
    private final LlamaServerConfig serverConfig;
    private final int defaultParallel;
    private final boolean autoSecondPass;
    private final int secondPassParallel;
    private final String modelName;
    private final String machineName;
    private final String hardwareNotes;
    private final Path corpusDirectory;
    private final String documentExtension;
    private final Duration requestTimeout;
    private final Duration connectTimeout;
    private final Path outputBaseDirectory;
    private final String experimentLabel;
    private final String experimentNotes;
    private final String modelLabel;
    private final Path documentList;
    private final boolean hashModelFile;
    private final Path promptFile;
    private final String promptText;
    private final String promptId;
    private final ExtractionOptions extractionOptions;
    private final int[] sweepValues;
    private final int harnessSlotSize;
    private final int harnessSampleSize;
    private final boolean harnessWarmUp;

    // -------------------------------------------------------------------------
    // Selecting and loading
    // -------------------------------------------------------------------------

    /**
     * Removes "--config=PATH" / "--config PATH" and "--experiment=PATH" /
     * "--experiment PATH" from the command line, storing them in the
     * vascular.config and vascular.experiment system properties, and returns the
     * remaining arguments. Call it first in main(), before arguments are interpreted.
     */
    public static String[] consumeConfigArgument(String[] args) {
        if (args == null) {
            return new String[0];
        }
        List<String> rest = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.startsWith("--config=")) {
                System.setProperty(CONFIG_PROPERTY, a.substring("--config=".length()));
            } else if (a.equals("--config") && i + 1 < args.length) {
                System.setProperty(CONFIG_PROPERTY, args[++i]);
            } else if (a.startsWith("--experiment=")) {
                System.setProperty(EXPERIMENT_PROPERTY, a.substring("--experiment=".length()));
            } else if (a.equals("--experiment") && i + 1 < args.length) {
                System.setProperty(EXPERIMENT_PROPERTY, args[++i]);
            } else {
                rest.add(a);
            }
        }
        return rest.toArray(new String[0]);
    }

    /**
     * The machine settings file chosen by --config / -Dvascular.config / VASCULAR_CONFIG,
     * else typed at the console prompt (Enter = null = built-in VA defaults).
     * Returns null for the built-in defaults.
     */
    public static Path resolveMachineFile(Scanner console) {
        String path = clean(System.getProperty(CONFIG_PROPERTY));
        if (path.isEmpty()) {
            path = clean(System.getenv(CONFIG_ENV));
        }
        if (path.isEmpty() && console != null) {
            System.out.println("Settings file (.properties) for this computer. "
                    + "Press Enter for the built-in VA Xeon + A2 defaults.");
            System.out.print("Settings file: ");
            path = clean(console.nextLine());
        }
        return path.isEmpty() ? null : Paths.get(expandHome(path));
    }

    /** Resolves the machine file (and optional experiment file) and loads them. */
    public static AppSettings load(Scanner console) throws IOException {
        Path machine = resolveMachineFile(console);
        String exp = clean(System.getProperty(EXPERIMENT_PROPERTY));
        if (exp.isEmpty()) {
            exp = clean(System.getenv(EXPERIMENT_ENV));
        }
        Path experiment = exp.isEmpty() ? null : Paths.get(expandHome(exp));
        return fromFiles(machine, experiment);
    }

    public static AppSettings builtInVaDefaults() {
        return parse(VA_DEFAULTS, null, "built-in VA defaults (Xeon Silver 4310 + A2)", null, null, null);
    }

    public static AppSettings fromFile(Path file) throws IOException {
        return fromFiles(file, null);
    }

    /** machine may be null (built-in VA defaults); experiment may be null (no experiment layer). */
    public static AppSettings fromFiles(Path machine, Path experiment) throws IOException {
        String machineText = VA_DEFAULTS;
        Path machineBase = null;
        String machineSource = "built-in VA defaults (Xeon Silver 4310 + A2)";
        if (machine != null) {
            machineText = readSettingsFile(machine);
            machineBase = machine.toAbsolutePath().getParent();
            machineSource = machine.toAbsolutePath().toString();
        }
        String expText = null;
        Path expBase = null;
        String expSource = null;
        if (experiment != null) {
            expText = readSettingsFile(experiment);
            expBase = experiment.toAbsolutePath().getParent();
            expSource = experiment.toAbsolutePath().toString();
        }
        return parse(machineText, machineBase, machineSource, expText, expBase, expSource);
    }

    private static String readSettingsFile(Path file) throws IOException {
        if (!Files.isRegularFile(file)) {
            throw new IOException("Settings file not found: " + file.toAbsolutePath());
        }
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }

    // -------------------------------------------------------------------------
    // Parsing
    // -------------------------------------------------------------------------

    static AppSettings parse(String text, Path baseDir, String source) {
        return parse(text, baseDir, source, null, null, null);
    }

    static AppSettings parse(String machineText, Path machineBase, String machineSource,
            String experimentText, Path experimentBase, String experimentSource) {
        Map<String, Entry> m = new LinkedHashMap<>();
        List<String> errors = new ArrayList<>();
        readInto(m, machineText, machineBase, machineSource, errors);
        if (experimentText != null) {
            readInto(m, experimentText, experimentBase, experimentSource, errors);
        }
        String source = experimentSource == null ? machineSource
                : machineSource + "  +  experiment " + experimentSource;

        Reader r = new Reader(m, errors);

        Path executable = r.path("server.executable", null, true);
        Path model = r.path("server.model", null, true);
        String host = r.string("server.host", "127.0.0.1");
        int port = r.integer("server.port", 8080, 1, 65535);
        int gpuLayers = r.integer("server.n-gpu-layers", 0, 0, 10000);
        int ctx = r.integer("server.ctx-size", 49152, 256, 4_000_000);
        int batch = r.integer("server.batch-size", 2048, 1, 1_000_000);
        int ubatch = r.integer("server.ubatch-size", 512, 1, 1_000_000);
        String flash = r.string("server.flash-attn", "auto");
        if (!Arrays.asList("on", "off", "auto").contains(flash)) {
            errors.add("server.flash-attn must be on, off or auto: " + flash);
        }
        int threads = r.integer("server.threads", 0, 0, 1024);
        boolean contBatching = r.bool("server.cont-batching", true);
        List<String> extra = splitArgs(r.string("server.extra-args", ""));
        int startup = r.integer("server.startup-timeout-minutes", 15, 1, 600);

        int parallel = r.integer("run.default-parallel", 4, 1, 256);
        boolean autoSecond = r.bool("run.auto-second-pass", true);
        int secondParallel = r.integer("run.second-pass-parallel", 1, 1, 256);
        String modelName = r.string("run.model-name", "local-model");
        String machineName = r.string("run.machine-name", "");
        String hardwareNotes = r.string("run.hardware-notes", "");

        Path corpus = r.path("corpus.directory", null, false);
        String ext = r.string("corpus.extension", ".txt");
        if (!ext.startsWith(".")) {
            ext = "." + ext;
        }

        int reqMin = r.integer("http.request-timeout-minutes", 20, 1, 1440);
        int connSec = r.integer("http.connect-timeout-seconds", 30, 1, 3600);

        boolean experimentMode = experimentText != null;
        Path outputBase = r.path("output.base-directory", null, false);
        String label = r.string("experiment.label", "run");
        String notes = r.string("experiment.notes", "");
        String modelLabel = r.string("experiment.model-label", "");
        Path docList = r.path("experiment.document-list", null, false);
        boolean hashModel = r.bool("experiment.hash-model-file", false);

        Path promptFile = r.path("prompt.file", null, false);
        String promptText = null;
        if (promptFile != null) {
            try {
                if (!Files.isRegularFile(promptFile)) {
                    errors.add("prompt.file not found: " + promptFile);
                } else {
                    promptText = new String(Files.readAllBytes(promptFile), StandardCharsets.UTF_8);
                    if (promptText.startsWith("\uFEFF")) {
                        promptText = promptText.substring(1);
                    }
                    promptText = promptText.replace("\r\n", "\n");
                    if (promptText.trim().isEmpty()) {
                        errors.add("prompt.file is empty: " + promptFile);
                    }
                }
            } catch (IOException e) {
                errors.add("prompt.file could not be read: " + promptFile + " (" + e.getMessage() + ")");
            }
        }
        String defaultPromptId = "builtin";
        if (promptFile != null) {
            String fn = promptFile.getFileName().toString();
            int dot = fn.lastIndexOf('.');
            defaultPromptId = dot > 0 ? fn.substring(0, dot) : fn;
        }
        String promptId = r.string("prompt.id", defaultPromptId);

        ExtractionOptions defaults = ExtractionOptions.defaults();
        double temperature = r.decimal("sampling.temperature", defaults.temperature, 0, 5);
        int topK = r.integer("sampling.top-k", defaults.topK, 0, 100000);
        double topP = r.decimal("sampling.top-p", defaults.topP, 0, 1);
        double repeatPenalty = r.decimal("sampling.repeat-penalty", defaults.repeatPenalty, 0, 10);
        int seed = r.integer("sampling.seed", defaults.seed, -1, Integer.MAX_VALUE);
        int maxTokens = r.integer("generation.max-tokens", defaults.maxTokens, 16, 1_000_000);
        int retryMax = r.integer("generation.retry-max-tokens", defaults.retryMaxTokens, 16, 1_000_000);
        int repairMax = r.integer("generation.repair-max-tokens", defaults.repairMaxTokens, 16, 1_000_000);
        boolean disableThinking = r.bool("generation.disable-thinking", defaults.disableThinking);
        boolean useGrammar = r.bool("generation.use-grammar", defaults.useGrammar);
        double charsPerToken = r.decimal("budget.chars-per-token", defaults.charsPerToken, 0.5, 20);
        int overhead = r.integer("budget.prompt-overhead-tokens", defaults.promptOverheadTokens, 0, 100000);
        int margin = r.integer("budget.safety-margin-tokens", defaults.safetyMarginTokens, 0, 100000);

        int[] sweep = r.intList("harness.sweep-values", new int[] {2, 3, 4});
        int slotSize = r.integer("harness.slot-size", 12288, 256, 4_000_000);
        int sample = r.integer("harness.sample-size", 20, 1, 100000);
        boolean warm = r.bool("harness.warm-up", true);

        if (ctx / Math.max(1, parallel) < 256) {
            errors.add("server.ctx-size / run.default-parallel leaves under 256 tokens per slot");
        }
        if (experimentMode) {
            if (outputBase == null) {
                errors.add("output.base-directory is required when an experiment file is used");
            }
            if (corpus == null && docList == null) {
                errors.add("set corpus.directory or experiment.document-list "
                        + "(an experiment run does not prompt for documents)");
            }
            if (docList != null && !Files.isRegularFile(docList)) {
                errors.add("experiment.document-list not found: " + docList);
            }
        }
        if (!errors.isEmpty()) {
            StringBuilder sb = new StringBuilder("Problems in settings " + source + ":");
            for (String e : errors) {
                sb.append("\n  - ").append(e);
            }
            throw new IllegalArgumentException(sb.toString());
        }

        if (modelLabel.isEmpty()) {
            String fn = model.getFileName().toString();
            modelLabel = fn.toLowerCase().endsWith(".gguf") ? fn.substring(0, fn.length() - 5) : fn;
        }

        LlamaServerConfig sc = new LlamaServerConfig(executable, host, port, model, gpuLayers,
                ctx, parallel, contBatching, batch, ubatch, threads, flash, extra, startup);
        ExtractionOptions opts = new ExtractionOptions(maxTokens, retryMax, repairMax, temperature,
                topK, topP, repeatPenalty, seed, disableThinking, useGrammar,
                charsPerToken, overhead, margin);
        return new AppSettings(source, machineSource, experimentSource, machineText, experimentText,
                r.used, sc, parallel, autoSecond, secondParallel, modelName, machineName,
                hardwareNotes, corpus, ext, Duration.ofMinutes(reqMin), Duration.ofSeconds(connSec),
                outputBase, label, notes, modelLabel, docList, hashModel, promptFile, promptText,
                promptId, opts, sweep, slotSize, sample, warm);
    }

    private static void readInto(Map<String, Entry> m, String text, Path baseDir, String source,
            List<String> errors) {
        if (text.startsWith("\uFEFF")) {
            text = text.substring(1);
        }
        Set<String> seenInThisFile = new TreeSet<>();
        int lineNo = 0;
        for (String raw : text.split("\\r?\\n", -1)) {
            lineNo++;
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("!")) {
                continue;
            }
            int eq = line.indexOf('=');
            if (eq < 1) {
                errors.add(source + " line " + lineNo + ": expected key = value, found: " + line);
                continue;
            }
            String key = line.substring(0, eq).trim();
            String value = unquote(line.substring(eq + 1).trim());
            if (!KNOWN_KEYS.contains(key)) {
                System.err.println("WARNING: unknown setting '" + key + "' on line " + lineNo
                        + " of " + source + " (ignored)");
                continue;
            }
            if (!seenInThisFile.add(key)) {
                errors.add(source + " line " + lineNo + ": '" + key + "' is set more than once");
            }
            m.put(key, new Entry(value, baseDir));
        }
    }

    private AppSettings(String source, String machineSource, String experimentSource,
            String machineText, String experimentText, Map<String, String> effectiveValues,
            LlamaServerConfig serverConfig, int defaultParallel, boolean autoSecondPass,
            int secondPassParallel, String modelName, String machineName, String hardwareNotes,
            Path corpusDirectory, String documentExtension, Duration requestTimeout,
            Duration connectTimeout, Path outputBaseDirectory, String experimentLabel,
            String experimentNotes, String modelLabel, Path documentList, boolean hashModelFile,
            Path promptFile, String promptText, String promptId, ExtractionOptions extractionOptions,
            int[] sweepValues, int harnessSlotSize, int harnessSampleSize, boolean harnessWarmUp) {
        this.source = source;
        this.machineSource = machineSource;
        this.experimentSource = experimentSource;
        this.machineText = machineText;
        this.experimentText = experimentText;
        this.effectiveValues = effectiveValues;
        this.serverConfig = serverConfig;
        this.defaultParallel = defaultParallel;
        this.autoSecondPass = autoSecondPass;
        this.secondPassParallel = secondPassParallel;
        this.modelName = modelName;
        this.machineName = machineName;
        this.hardwareNotes = hardwareNotes;
        this.corpusDirectory = corpusDirectory;
        this.documentExtension = documentExtension;
        this.requestTimeout = requestTimeout;
        this.connectTimeout = connectTimeout;
        this.outputBaseDirectory = outputBaseDirectory;
        this.experimentLabel = experimentLabel;
        this.experimentNotes = experimentNotes;
        this.modelLabel = modelLabel;
        this.documentList = documentList;
        this.hashModelFile = hashModelFile;
        this.promptFile = promptFile;
        this.promptText = promptText;
        this.promptId = promptId;
        this.extractionOptions = extractionOptions;
        this.sweepValues = sweepValues;
        this.harnessSlotSize = harnessSlotSize;
        this.harnessSampleSize = harnessSampleSize;
        this.harnessWarmUp = harnessWarmUp;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public String source()                 { return source; }
    public String machineSource()          { return machineSource; }
    /** Null when no experiment file was given. */
    public String experimentSource()       { return experimentSource; }
    /** True when an experiment file was given (non-interactive, run folder, telemetry). */
    public boolean experimentMode()        { return experimentSource != null; }
    public String machineFileText()        { return machineText; }
    /** Null when no experiment file was given. */
    public String experimentFileText()     { return experimentText; }
    /** Every known key with the value in effect (after defaults), for the run manifest. */
    public Map<String, String> effectiveValues() { return new LinkedHashMap<>(effectiveValues); }
    /** Server settings using run.default-parallel. */
    public LlamaServerConfig serverConfig(){ return serverConfig; }
    public int defaultParallel()           { return defaultParallel; }
    public boolean autoSecondPass()        { return autoSecondPass; }
    public int secondPassParallel()        { return secondPassParallel; }
    public String modelName()              { return modelName; }
    /** Empty when run.machine-name is not set; the run folder then uses the host name. */
    public String machineName()            { return machineName; }
    public String hardwareNotes()          { return hardwareNotes; }
    /** May be null if the file does not set corpus.directory; the caller then prompts. */
    public Path corpusDirectory()          { return corpusDirectory; }
    public String documentExtension()      { return documentExtension; }
    public Duration requestTimeout()       { return requestTimeout; }
    public Duration connectTimeout()       { return connectTimeout; }
    /** Null unless output.base-directory is set. */
    public Path outputBaseDirectory()      { return outputBaseDirectory; }
    public String experimentLabel()        { return experimentLabel; }
    public String experimentNotes()        { return experimentNotes; }
    /** experiment.model-label, or the model file name without .gguf. */
    public String modelLabel()             { return modelLabel; }
    /** Null unless experiment.document-list is set. */
    public Path documentList()             { return documentList; }
    public boolean hashModelFile()         { return hashModelFile; }
    /** Null when the built-in prompt is used. */
    public Path promptFile()               { return promptFile; }
    /** The prompt loaded from prompt.file (LF line endings), or null for the built-in prompt. */
    public String promptText()             { return promptText; }
    public String promptId()               { return promptId; }
    public ExtractionOptions extractionOptions() { return extractionOptions; }
    public int[] sweepValues()             { return sweepValues.clone(); }
    public int harnessSlotSize()           { return harnessSlotSize; }
    public int harnessSampleSize()         { return harnessSampleSize; }
    public boolean harnessWarmUp()         { return harnessWarmUp; }

    /** The system prompt to use: prompt.file if set, otherwise the one built into the code. */
    public String effectivePrompt(String builtInPrompt) {
        return promptText != null ? promptText : builtInPrompt;
    }

    /** Multi-line summary, printed at start-up so a log shows which profile ran. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append("Settings:        ").append(source).append('\n');
        sb.append("Server command:  ").append(String.join(" ", serverConfig.commandLine())).append('\n');
        sb.append("Slots:           ").append(serverConfig.describe()).append('\n');
        sb.append("Corpus:          ").append(corpusDirectory == null ? "(prompt)" : corpusDirectory)
                .append("  (*").append(documentExtension).append(")\n");
        sb.append("Passes:          default parallel ").append(defaultParallel)
                .append(", second pass ").append(autoSecondPass ? "parallel " + secondPassParallel : "off").append('\n');
        sb.append("HTTP timeouts:   request ").append(requestTimeout.toMinutes()).append(" min, connect ")
                .append(connectTimeout.getSeconds()).append(" s; server start-up ")
                .append(serverConfig.startupTimeoutMinutes()).append(" min\n");
        sb.append("Prompt:          ").append(promptFile == null ? "built-in" : promptFile)
                .append("  (id ").append(promptId).append(")\n");
        sb.append("Generation:      max_tokens ").append(extractionOptions.maxTokens)
                .append(", temperature ").append(extractionOptions.temperature)
                .append(", top_k ").append(extractionOptions.topK)
                .append(extractionOptions.seed >= 0 ? ", seed " + extractionOptions.seed : "")
                .append(", thinking ").append(extractionOptions.disableThinking ? "disabled" : "allowed")
                .append(", ").append(extractionOptions.useGrammar ? "grammar" : "json_schema");
        if (experimentMode()) {
            sb.append("\nExperiment:      ").append(experimentLabel).append("  (model ").append(modelLabel)
                    .append("); results under ").append(outputBaseDirectory);
        }
        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static String clean(String s) {
        return s == null ? "" : unquote(s.trim());
    }

    private static String unquote(String s) {
        if (s.length() >= 2 && s.startsWith("\"") && s.endsWith("\"")) {
            return s.substring(1, s.length() - 1);
        }
        return s;
    }

    private static String expandHome(String p) {
        if (p.equals("~") || p.startsWith("~/") || p.startsWith("~\\")) {
            return System.getProperty("user.home") + p.substring(1);
        }
        return p;
    }

    /** Whitespace-separated arguments; double-quoted segments may contain spaces. */
    static List<String> splitArgs(String s) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuote = false;
        boolean have = false;
        for (char ch : s.toCharArray()) {
            if (ch == '"') {
                inQuote = !inQuote;
                have = true;
            } else if (Character.isWhitespace(ch) && !inQuote) {
                if (have) {
                    out.add(cur.toString());
                    cur.setLength(0);
                    have = false;
                }
            } else {
                cur.append(ch);
                have = true;
            }
        }
        if (have) {
            out.add(cur.toString());
        }
        return out;
    }

    /** Typed access to the merged settings; problems are appended to the shared error list. */
    private static final class Reader {
        private final Map<String, Entry> m;
        private final List<String> errors;
        /** key -> value in effect, in the order the keys were read. */
        final Map<String, String> used = new LinkedHashMap<>();

        Reader(Map<String, Entry> m, List<String> errors) {
            this.m = m;
            this.errors = errors;
        }

        String string(String key, String dflt) {
            Entry e = m.get(key);
            String v = e == null ? dflt : e.value;
            used.put(key, v);
            return v;
        }

        int integer(String key, int dflt, int min, int max) {
            int result = dflt;
            Entry e = m.get(key);
            if (e != null && !e.value.isEmpty()) {
                try {
                    int n = Integer.parseInt(e.value.trim());
                    if (n < min || n > max) {
                        errors.add(key + " must be between " + min + " and " + max + ": " + e.value);
                    } else {
                        result = n;
                    }
                } catch (NumberFormatException ex) {
                    errors.add(key + " must be a whole number: " + e.value);
                }
            }
            used.put(key, Integer.toString(result));
            return result;
        }

        double decimal(String key, double dflt, double min, double max) {
            double result = dflt;
            Entry e = m.get(key);
            if (e != null && !e.value.isEmpty()) {
                try {
                    double d = Double.parseDouble(e.value.trim());
                    if (Double.isNaN(d) || d < min || d > max) {
                        errors.add(key + " must be a number between " + min + " and " + max + ": " + e.value);
                    } else {
                        result = d;
                    }
                } catch (NumberFormatException ex) {
                    errors.add(key + " must be a number: " + e.value);
                }
            }
            used.put(key, Double.toString(result));
            return result;
        }

        boolean bool(String key, boolean dflt) {
            boolean result = dflt;
            Entry e = m.get(key);
            if (e != null && !e.value.isEmpty()) {
                String v = e.value;
                if (v.equalsIgnoreCase("true") || v.equalsIgnoreCase("yes") || v.equalsIgnoreCase("on")) {
                    result = true;
                } else if (v.equalsIgnoreCase("false") || v.equalsIgnoreCase("no") || v.equalsIgnoreCase("off")) {
                    result = false;
                } else {
                    errors.add(key + " must be true or false: " + v);
                }
            }
            used.put(key, Boolean.toString(result));
            return result;
        }

        int[] intList(String key, int[] dflt) {
            int[] result = dflt;
            Entry e = m.get(key);
            if (e != null && !e.value.isEmpty()) {
                String[] parts = e.value.split("[,\\s]+");
                int[] out = new int[parts.length];
                boolean ok = true;
                for (int i = 0; i < parts.length && ok; i++) {
                    try {
                        out[i] = Integer.parseInt(parts[i]);
                        if (out[i] < 1 || out[i] > 256) {
                            ok = false;
                        }
                    } catch (NumberFormatException ex) {
                        ok = false;
                    }
                }
                if (ok) {
                    result = out;
                } else {
                    errors.add(key + " must be a comma-separated list of numbers 1-256: " + e.value);
                }
            }
            used.put(key, Arrays.toString(result).replace("[", "").replace("]", "").replace(" ", ""));
            return result;
        }

        /** A path; relative paths resolve against the folder of the file that set them. */
        Path path(String key, Path dflt, boolean required) {
            Entry e = m.get(key);
            if (e == null || e.value.isEmpty()) {
                if (required && dflt == null) {
                    errors.add(key + " is required");
                }
                used.put(key, dflt == null ? "" : dflt.toString());
                return dflt;
            }
            try {
                Path p = Paths.get(expandHome(e.value));
                if (!p.isAbsolute() && e.baseDir != null) {
                    p = e.baseDir.resolve(p).normalize();
                }
                used.put(key, p.toString());
                return p;
            } catch (RuntimeException ex) {
                errors.add(key + " is not a valid path: " + e.value);
                used.put(key, e.value);
                return dflt;
            }
        }
    }
}
