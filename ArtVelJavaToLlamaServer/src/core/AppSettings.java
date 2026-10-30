package core;

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
 * All values that differ from one computer to the next, read from ONE external
 * properties file chosen at start-up. The same build then runs unchanged on the
 * Xeon + A2 machine, the small CPU-only PC and the Mac Studio.
 *
 * HOW THE FILE IS CHOSEN (first match wins):
 *   1. a command-line argument  --config=PATH   (consumed by VascularRunner)
 *   2. the Java system property -Dvascular.config=PATH   (Eclipse: Run Configurations > Arguments > VM arguments)
 *   3. the environment variable VASCULAR_CONFIG
 *   4. a prompt on the console; pressing Enter uses the built-in VA Xeon + A2 defaults
 *
 * FILE FORMAT: one  key = value  per line. Lines starting with # or ! are comments.
 * This is NOT java.util.Properties: backslashes are NOT escape characters, so a
 * Windows path such as P:\ORD\Models\x.gguf can be pasted as-is (forward slashes
 * also work). Surrounding double quotes are optional. A relative path is resolved
 * against the folder containing the settings file; a leading ~ means the user's
 * home folder. Unknown keys produce a warning; bad or missing values are all
 * reported together in a single error.
 *
 * Keys (see the sample files for explanations):
 *   server.executable  server.model            (required)
 *   server.host  server.port  server.n-gpu-layers  server.ctx-size
 *   server.batch-size  server.ubatch-size  server.flash-attn  server.threads
 *   server.cont-batching  server.extra-args  server.startup-timeout-minutes
 *   run.default-parallel  run.auto-second-pass  run.second-pass-parallel  run.model-name
 *   corpus.directory  corpus.extension
 *   http.request-timeout-minutes  http.connect-timeout-seconds
 *   harness.sweep-values  harness.slot-size  harness.sample-size  harness.warm-up
 *
 * What stays in code on purpose: the prompt and MAX_TOKENS. They are properties
 * of the task and the model, not of the computer.
 */
public final class AppSettings {

    public static final String CONFIG_PROPERTY = "vascular.config";
    public static final String CONFIG_ENV = "VASCULAR_CONFIG";

    private static final Set<String> KNOWN_KEYS = new TreeSet<>(Arrays.asList(
            "server.executable", "server.model", "server.host", "server.port",
            "server.n-gpu-layers", "server.ctx-size", "server.batch-size",
            "server.ubatch-size", "server.flash-attn", "server.threads",
            "server.cont-batching", "server.extra-args", "server.startup-timeout-minutes",
            "run.default-parallel", "run.auto-second-pass", "run.second-pass-parallel",
            "run.model-name", "corpus.directory", "corpus.extension",
            "http.request-timeout-minutes", "http.connect-timeout-seconds",
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

    private final String source;
    private final LlamaServerConfig serverConfig;
    private final int defaultParallel;
    private final boolean autoSecondPass;
    private final int secondPassParallel;
    private final String modelName;
    private final Path corpusDirectory;
    private final String documentExtension;
    private final Duration requestTimeout;
    private final Duration connectTimeout;
    private final int[] sweepValues;
    private final int harnessSlotSize;
    private final int harnessSampleSize;
    private final boolean harnessWarmUp;

    // -------------------------------------------------------------------------
    // Selecting and loading
    // -------------------------------------------------------------------------

    /**
     * Removes a "--config=PATH" (or "--config PATH") argument from the command line,
     * storing PATH in the vascular.config system property, and returns the remaining
     * arguments. Call it first in main(), before arguments are interpreted.
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
            } else {
                rest.add(a);
            }
        }
        return rest.toArray(new String[0]);
    }

    /** Resolves the settings file (property, environment variable, then prompt) and loads it. */
    public static AppSettings load(Scanner console) throws IOException {
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
        return path.isEmpty() ? builtInVaDefaults() : fromFile(Paths.get(expandHome(path)));
    }

    public static AppSettings builtInVaDefaults() {
        return parse(VA_DEFAULTS, null, "built-in VA defaults (Xeon Silver 4310 + A2)");
    }

    public static AppSettings fromFile(Path file) throws IOException {
        if (!Files.isRegularFile(file)) {
            throw new IOException("Settings file not found: " + file.toAbsolutePath());
        }
        byte[] bytes = Files.readAllBytes(file);
        String text = new String(bytes, StandardCharsets.UTF_8);
        return parse(text, file.toAbsolutePath().getParent(), file.toAbsolutePath().toString());
    }

    // -------------------------------------------------------------------------
    // Parsing
    // -------------------------------------------------------------------------

    static AppSettings parse(String text, Path baseDir, String source) {
        Map<String, String> m = new LinkedHashMap<>();
        List<String> errors = new ArrayList<>();
        if (text.startsWith("\uFEFF")) {
            text = text.substring(1);
        }
        int lineNo = 0;
        for (String raw : text.split("\\r?\\n", -1)) {
            lineNo++;
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("!")) {
                continue;
            }
            int eq = line.indexOf('=');
            if (eq < 1) {
                errors.add("line " + lineNo + ": expected key = value, found: " + line);
                continue;
            }
            String key = line.substring(0, eq).trim();
            String value = unquote(line.substring(eq + 1).trim());
            if (!KNOWN_KEYS.contains(key)) {
                System.err.println("WARNING: unknown setting '" + key + "' on line " + lineNo
                        + " of " + source + " (ignored)");
                continue;
            }
            if (m.containsKey(key)) {
                errors.add("line " + lineNo + ": '" + key + "' is set more than once");
            }
            m.put(key, value);
        }

        Reader r = new Reader(m, baseDir, errors);

        Path executable = r.path("server.executable", null);
        Path model = r.path("server.model", null);
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

        Path corpus = r.path("corpus.directory", null);
        String ext = r.string("corpus.extension", ".txt");
        if (!ext.startsWith(".")) {
            ext = "." + ext;
        }

        int reqMin = r.integer("http.request-timeout-minutes", 20, 1, 1440);
        int connSec = r.integer("http.connect-timeout-seconds", 30, 1, 3600);

        int[] sweep = r.intList("harness.sweep-values", new int[] {2, 3, 4});
        int slotSize = r.integer("harness.slot-size", 12288, 256, 4_000_000);
        int sample = r.integer("harness.sample-size", 20, 1, 100000);
        boolean warm = r.bool("harness.warm-up", true);

        if (ctx / Math.max(1, parallel) < 256) {
            errors.add("server.ctx-size / run.default-parallel leaves under 256 tokens per slot");
        }
        if (!errors.isEmpty()) {
            StringBuilder sb = new StringBuilder("Problems in settings file " + source + ":");
            for (String e : errors) {
                sb.append("\n  - ").append(e);
            }
            throw new IllegalArgumentException(sb.toString());
        }

        LlamaServerConfig sc = new LlamaServerConfig(executable, host, port, model, gpuLayers,
                ctx, parallel, contBatching, batch, ubatch, threads, flash, extra, startup);
        return new AppSettings(source, sc, parallel, autoSecond, secondParallel, modelName,
                corpus, ext, Duration.ofMinutes(reqMin), Duration.ofSeconds(connSec),
                sweep, slotSize, sample, warm);
    }

    private AppSettings(String source, LlamaServerConfig serverConfig, int defaultParallel,
            boolean autoSecondPass, int secondPassParallel, String modelName,
            Path corpusDirectory, String documentExtension, Duration requestTimeout,
            Duration connectTimeout, int[] sweepValues, int harnessSlotSize,
            int harnessSampleSize, boolean harnessWarmUp) {
        this.source = source;
        this.serverConfig = serverConfig;
        this.defaultParallel = defaultParallel;
        this.autoSecondPass = autoSecondPass;
        this.secondPassParallel = secondPassParallel;
        this.modelName = modelName;
        this.corpusDirectory = corpusDirectory;
        this.documentExtension = documentExtension;
        this.requestTimeout = requestTimeout;
        this.connectTimeout = connectTimeout;
        this.sweepValues = sweepValues;
        this.harnessSlotSize = harnessSlotSize;
        this.harnessSampleSize = harnessSampleSize;
        this.harnessWarmUp = harnessWarmUp;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public String source()                 { return source; }
    /** Server settings using run.default-parallel. */
    public LlamaServerConfig serverConfig(){ return serverConfig; }
    public int defaultParallel()           { return defaultParallel; }
    public boolean autoSecondPass()        { return autoSecondPass; }
    public int secondPassParallel()        { return secondPassParallel; }
    public String modelName()              { return modelName; }
    /** May be null if the file does not set corpus.directory; the caller then prompts. */
    public Path corpusDirectory()          { return corpusDirectory; }
    public String documentExtension()      { return documentExtension; }
    public Duration requestTimeout()       { return requestTimeout; }
    public Duration connectTimeout()       { return connectTimeout; }
    public int[] sweepValues()             { return sweepValues.clone(); }
    public int harnessSlotSize()           { return harnessSlotSize; }
    public int harnessSampleSize()         { return harnessSampleSize; }
    public boolean harnessWarmUp()         { return harnessWarmUp; }

    /** Multi-line summary, printed at start-up so a log shows which computer profile ran. */
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
                .append(serverConfig.startupTimeoutMinutes()).append(" min");
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

    /** Typed access to the parsed map; problems are appended to the shared error list. */
    private static final class Reader {
        private final Map<String, String> m;
        private final Path baseDir;
        private final List<String> errors;

        Reader(Map<String, String> m, Path baseDir, List<String> errors) {
            this.m = m;
            this.baseDir = baseDir;
            this.errors = errors;
        }

        String string(String key, String dflt) {
            String v = m.get(key);
            return v == null ? dflt : v;
        }

        int integer(String key, int dflt, int min, int max) {
            String v = m.get(key);
            if (v == null || v.isEmpty()) {
                return dflt;
            }
            try {
                int n = Integer.parseInt(v.trim());
                if (n < min || n > max) {
                    errors.add(key + " must be between " + min + " and " + max + ": " + v);
                    return dflt;
                }
                return n;
            } catch (NumberFormatException e) {
                errors.add(key + " must be a whole number: " + v);
                return dflt;
            }
        }

        boolean bool(String key, boolean dflt) {
            String v = m.get(key);
            if (v == null || v.isEmpty()) {
                return dflt;
            }
            if (v.equalsIgnoreCase("true") || v.equalsIgnoreCase("yes") || v.equalsIgnoreCase("on")) {
                return true;
            }
            if (v.equalsIgnoreCase("false") || v.equalsIgnoreCase("no") || v.equalsIgnoreCase("off")) {
                return false;
            }
            errors.add(key + " must be true or false: " + v);
            return dflt;
        }

        int[] intList(String key, int[] dflt) {
            String v = m.get(key);
            if (v == null || v.isEmpty()) {
                return dflt;
            }
            String[] parts = v.split("[,\\s]+");
            int[] out = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                try {
                    out[i] = Integer.parseInt(parts[i]);
                    if (out[i] < 1 || out[i] > 256) {
                        throw new NumberFormatException();
                    }
                } catch (NumberFormatException e) {
                    errors.add(key + " must be a comma-separated list of numbers 1-256: " + v);
                    return dflt;
                }
            }
            return out;
        }

        /** Required when dflt is null and the key is absent (an error is recorded; null returned). */
        Path path(String key, Path dflt) {
            String v = m.get(key);
            if (v == null || v.isEmpty()) {
                if (dflt == null && (key.startsWith("server."))) {
                    errors.add(key + " is required");
                }
                return dflt;
            }
            try {
                Path p = Paths.get(expandHome(v));
                if (!p.isAbsolute() && baseDir != null) {
                    p = baseDir.resolve(p).normalize();
                }
                return p;
            } catch (RuntimeException e) {
                errors.add(key + " is not a valid path: " + v);
                return dflt;
            }
        }
    }
}
